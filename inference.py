"""
inference.py

Executes SAM3, CLAHE, MNS filters
and plotting
"""
import torch
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Import standard official Hugging Face components
from transformers import Sam3Model, Sam3Processor

# ==========================================
# 1. SAM3 model set up
# ==========================================

def load_sam3_model(bpe_path: str, confidence: float, device):
    """
    Loads in the SAM3 model into the GPU or CPU
    """
    logging.info("Loading in SAM 3...")

    model_id = "facebook/sam3"

    if device.type == "cuda":
        # Apply high-performance hardware configurations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load the foundational weights
        model = Sam3Model.from_pretrained(model_id).to(device)
        processor = Sam3Processor.from_pretrained(model_id)

        # Attach the model instance to the processor object to preserve custom pipeline compatibility
        processor.model = model

        # Apply torch.compile to optimize the underlying computational execution graph
        logging.info("Compiling model graph using torch.compile...")
        model = torch.compile(model)
    else:
        # Load the model directly into CPU space without optimization overhead
        model = Sam3Model.from_pretrained(model_id).to(device)
        processor = Sam3Processor.from_pretrained(model_id)
        processor.model = model

    return model, processor

# ==========================================
# 2. Helper functions
# ==========================================

def yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    """
    Converts YOLO normalized (center_x, center_y, width, height)
    to absolute pixel coordinates (x_min, y_min, x_max, y_max).
    """

    # Un-normalize coords
    x_c, y_c, w, h = x_c * img_w, y_c * img_h, w * img_w, h * img_h

    # Convert Center-WH to TopLeft-BottomRight
    x1 = x_c - (w / 2)
    x2 = x_c + (w / 2)
    y1 = y_c - (h / 2)
    y2 = y_c + (h / 2)

    return [x1, y1, x2, y2]

def apply_clahe(image):
    """
    Applies CLAHE to an image
    """

    # Convert from BGR to LAB color space and split
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (only to the luminosity)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the channels back and return the image in BGR color space
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_nms(boxes, scores, confidence, iou_threshold=0.40, return_indices=False):
    """
    Deduplicate overlapping predictions
    across passes using NMS

    [BASELINE] Pure-IoU NMS via cv2.dnn.NMSBoxes. This is the currently validated
    citrus default (Recall 0.74 / Precision 0.75). Kept intact so the harness can
    A/B it against apply_nms_dualgate without changing the reference behavior.

    return_indices: if True, also return the indices into the INPUT arrays that
                    survived, so a caller can subselect a parallel list (e.g.
                    masks). Default False keeps the legacy 2-tuple return.
    """
    if len(boxes) == 0:
        empty_idx = np.array([], dtype=int)
        return (np.array([]), np.array([]), empty_idx) if return_indices else (np.array([]), np.array([]))

    cv2_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cv2_boxes.append([float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)])

    indices = cv2.dnn.NMSBoxes(
        cv2_boxes,
        scores.astype(float).tolist(),
        score_threshold=confidence,
        nms_threshold=iou_threshold
    )

    if len(indices) > 0:
        indices = indices.flatten()
        if return_indices:
            return boxes[indices], scores[indices], indices
        return boxes[indices], scores[indices]

    empty_idx = np.array([], dtype=int)
    return (np.array([]), np.array([]), empty_idx) if return_indices else (np.array([]), np.array([]))

def apply_nms_dualgate(boxes, scores, confidence,
                       iou_threshold=0.40,
                       iom_threshold=0.90,
                       min_size_ratio_for_containment=0.40,
                       max_concentric_offset_ratio=0.43,
                       use_concentric=False,
                       return_indices=False,
                       masks=None,
                       gate_mode="dual"):
    """
    Dual-Gate NMS for the citrus pipeline. Ported from the PixMo/CountBench engine
    and re-tuned for small, same-color, clustered fruit.

    gate_mode: additive, backward-compatible. "dual" (default) keeps Gate A (IoU)
    OR Gate B (IoM containment, size-ratio-guarded) exactly as before --
    byte-identical to the original behavior. "iou_only" disables Gate B (pure
    lateral-duplicate IoU suppression). "iom_only" disables Gate A AND drops
    Gate B's size-ratio guard (pure containment suppression: iom > iom_threshold
    alone) -- useful for datasets like CARPK where uniform-size objects sit in
    dense grids and a box fully swallowed by another is never a distinct real
    object regardless of relative size, so the guard (meant to protect a small
    fruit genuinely nested in a citrus cluster box) would only hide the
    duplicate. The concentric sub-gate (opt-in) still adds to whichever gate is
    active.

    Gate A (IoU): lateral-duplicate suppression. Held at the citrus-validated 0.40
                  so swapping IoU->dual-gate is a *controlled, additive* change and
                  not a hidden relaxation of the IoU threshold (the source engine ran
                  IoU at 0.95, which would have kept far more overlapping boxes).

    Gate B (IoM = intersection / min-area): removes a box largely *contained* by a
                  bigger one (an individual fruit swallowed by a cluster proposal),
                  but only when guarded:
                    - containment sub-gate: size_ratio >= min_size_ratio_for_containment
                      -> the two boxes are comparably sized, so this is a real nested
                      duplicate, not a small fruit sitting inside a big cluster box.
                    - concentric sub-gate (OPT-IN, default OFF): suppresses a smaller
                      box centered inside a larger one. On CARPK-style scenes this kills
                      nested junk; on THIS dataset it can delete a real fruit that
                      happens to sit dead-center in a cluster box. Leave off unless the
                      overlays show concentric false positives the size guard misses.

    A box removed here is therefore always either an IoU duplicate (identical to the
    baseline's intent) or a size-guarded nested duplicate.

    masks: additive, backward-compatible. None (default) keeps the box-based gates
    byte-for-byte. When given (a list of per-instance boolean arrays in this image's
    frame, index-aligned with boxes), Gate A IoU and Gate B IoM/size-ratio are
    measured on the masks instead of the boxes; the boxes still bound which pairs
    can overlap and the concentric sub-gate stays box-geometry-based. Combine with
    return_indices=True to subselect the surviving masks.
    """
    empty_idx = np.array([], dtype=int)
    if len(boxes) == 0:
        return (np.array([]), np.array([]), empty_idx) if return_indices else (np.array([]), np.array([]))

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    valid = np.where(scores >= confidence)[0]
    if len(valid) == 0:
        return (np.array([]), np.array([]), empty_idx) if return_indices else (np.array([]), np.array([]))
    boxes, scores = boxes[valid], scores[valid]
    if masks is not None:
        masks = [masks[int(v)] for v in valid]
        mask_areas = np.array([float(np.count_nonzero(m)) for m in masks])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)

        if masks is not None:
            # Mask-mode overlap: recompute intersection and areas from the masks.
            # A pair whose boxes don't touch can't have mask overlap, so the
            # box intersection bounds which pairs need the (expensive) mask AND.
            mask_inter = np.zeros_like(inter)
            for jj, j in enumerate(rest):
                if inter[jj] > 0:
                    mask_inter[jj] = float(np.count_nonzero(np.logical_and(masks[i], masks[j])))
            inter = mask_inter
            area_i, area_rest = mask_areas[i], mask_areas[rest]
        else:
            area_i, area_rest = areas[i], areas[rest]

        # --- GATE A: IoU (lateral jitter / cross-pass duplicates) ---
        union = area_i + area_rest - inter
        iou = np.zeros_like(inter)
        m = union > 0
        iou[m] = inter[m] / union[m]

        # --- GATE B: IoM containment with size-ratio guard ---
        min_a = np.minimum(area_i, area_rest)
        max_a = np.maximum(area_i, area_rest)
        iom = np.zeros_like(inter)
        m2 = min_a > 0
        iom[m2] = inter[m2] / min_a[m2]
        size_ratio = np.zeros_like(inter)
        m3 = max_a > 0
        size_ratio[m3] = min_a[m3] / max_a[m3]

        iou_violation = iou > iou_threshold
        pure_containment_violation = iom > iom_threshold
        containment_violation = pure_containment_violation & (size_ratio >= min_size_ratio_for_containment)
        if gate_mode == "iou_only":
            suppress = iou_violation
        elif gate_mode == "iom_only":
            # No size-ratio guard: a fully-contained box is always a duplicate/
            # fragment here, never a genuinely distinct smaller object (unlike
            # the citrus dual-gate case the guard was built for).
            suppress = pure_containment_violation
        else:
            suppress = iou_violation | containment_violation

        # --- OPT-IN concentric sub-gate (recall-risky on small clustered fruit) ---
        if use_concentric:
            i_larger = areas[i] >= areas[rest]
            larger_w = np.where(i_larger, x2[i] - x1[i], x2[rest] - x1[rest])
            larger_h = np.where(i_larger, y2[i] - y1[i], y2[rest] - y1[rest])
            half_diag = 0.5 * np.sqrt(larger_w ** 2 + larger_h ** 2)
            center_dist = np.sqrt((cx[i] - cx[rest]) ** 2 + (cy[i] - cy[rest]) ** 2)
            offset = np.full_like(center_dist, np.inf)
            m4 = half_diag > 0
            offset[m4] = center_dist[m4] / half_diag[m4]
            is_concentric = offset <= max_concentric_offset_ratio
            suppress = suppress | ((iom > iom_threshold) & is_concentric)

        order = rest[np.where(~suppress)[0]]

    if return_indices:
        # map survivors back to indices into the ORIGINAL input arrays
        original_idx = valid[np.array(keep, dtype=int)] if keep else empty_idx
        return boxes[keep], scores[keep], original_idx
    return boxes[keep], scores[keep]

def plot_graph_scene(image_pil, graph, output_path="output.jpg"):
    """
    Renders the master dictionary database color coded
    by verification classification.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_pil)

    for node in graph.nodes.values():
        xmin, ymin, xmax, ymax = node.box

        # Color coding schema: Green for approved fruit, Red for confirmed leaf
        color = "lime" if node.classification == "fruit" else "red"
        linestyle = "-" if node.classification == "fruit" else "--"

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle)
        ax.add_patch(rect)

        #txt = f"F:{node.scores['fruit_verification']:.2f}|L:{node.scores['leaf_verification']:.2f}"
        #ax.text(xmin, ymin - 4, txt, color='white', fontsize=6, weight='bold', backgroundcolor='black')

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 3. Run SAM3 inference
# ==========================================

def run_raw_inference(processor, image_np, confidence, prompt, pos_boxes=None, neg_boxes=None,
                      disable_size_filter=False, return_masks=False):
    """
    Executes a single forward pass through SAM3 supporting exemplars.

    return_masks:
        False (default) -> returns (boxes, scores). Backward-compatible: every
                           existing caller (pipeline.py engines, legacy main.py)
                           keeps getting a 2-tuple and is unaffected.
        True            -> returns (boxes, scores, masks), where masks is a list
                           of per-instance boolean numpy arrays (H×W in this
                           image's frame), aligned index-for-index with boxes.
                           Used by the agent so association can run on mask-IoU
                           / mask-IoM rather than box overlap.
    """
    # --- Load in the image ---
    img_pil = Image.fromarray(image_np)
    w, h = img_pil.size
    image_area = w * h

    model = processor.model

    # --- Positive & Negative exemplars Inference ---
    input_boxes_list = []
    input_labels_list = []

    if pos_boxes is not None and len(pos_boxes) > 0:
        for box in pos_boxes.tolist():
            input_boxes_list.append(box.tolist() if hasattr(box, "tolist") else list(box))
            input_labels_list.append(1)

    if neg_boxes is not None and len(neg_boxes) > 0:
        for box in neg_boxes.tolist():
            input_boxes_list.append(box.tolist() if hasattr(box, "tolist") else list(box))
            input_labels_list.append(0)

    # --- Vectorized inputs dict ---
    input_kwargs = {
        "images": img_pil,
        "text": prompt,
        "return_tensors": "pt"
    }

    if input_boxes_list:
        input_kwargs["input_boxes"] = [input_boxes_list]
        input_kwargs["input_boxes_labels"] = [input_labels_list]

    inputs = processor(**input_kwargs).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process outputs using the standard HF instance segmentation utility
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(confidence),
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    # --- Get the bounding boxes and scores ---
    raw_boxes = results["boxes"].cpu().numpy() if hasattr(results["boxes"], "cpu") else np.array(results["boxes"])
    raw_scores = results["scores"].cpu().numpy() if hasattr(results["scores"], "cpu") else np.array(results["scores"])

    # --- Masks (optional), kept index-aligned with boxes through filtering ---
    raw_masks = None
    if return_masks and "masks" in results and results["masks"] is not None:
        rm = results["masks"]
        rm = rm.cpu().numpy() if hasattr(rm, "cpu") else np.array(rm)
        # rm shape (N, H, W) probabilities or logits -> boolean at 0.5
        raw_masks = [np.asarray(rm[k]) > 0.5 for k in range(len(rm))]

    # --- Filter out the boxes that are just too large ---
    if prompt == "tree canopy" or disable_size_filter:
        MAX_AREA_THRESHOLD = 1.00 * image_area  # Allow up to 100% frame coverage
        MIN_DIM_THRESHOLD = 0                   # No minimum limit for the macro pass
    else:
        MAX_AREA_THRESHOLD = 0.50 * image_area  # Restrict micro-fruit boxes to 30% max
        MIN_DIM_THRESHOLD = 10                  # Silently drop micro-noise under 10px

    filtered_boxes = []
    filtered_scores = []
    filtered_masks = []

    for idx, (b, s) in enumerate(zip(raw_boxes, raw_scores)):
        xmin, ymin, xmax, ymax = b
        box_width = xmax - xmin
        box_height = ymax - ymin
        box_area = box_width * box_height

        # Only keep the box if it's smaller than our max threshold
        #if box_width < MIN_DIM_THRESHOLD or box_height < MIN_DIM_THRESHOLD:
         #   continue
        if box_area < MAX_AREA_THRESHOLD:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            if raw_masks is not None:
                filtered_masks.append(raw_masks[idx])
        else:
            logging.warning(f"Discarded massive hallucinated box: Area {box_area:.0f}px vs Image Total {image_area:.0f}px")

    boxes = np.array(filtered_boxes) if filtered_boxes else np.empty((0, 4))
    scores = np.array(filtered_scores) if filtered_scores else np.empty((0,))

    if return_masks:
        return boxes, scores, filtered_masks
    return boxes, scores

# ==========================================
# 4. Run verificartion
# ==========================================
def verify_box_semantics(processor, image_np, box, scale_factor=1.1,
                         fruit_prompt="fruit", leaf_prompt="green leaf"):
    """
    Verify whether the bounding box is a fruit or a leaf
    using bicubic upsampling and macro prompt tuning.

    fruit_prompt / leaf_prompt: short noun-phrase text prompts, matching the
    style used everywhere else SAM3 is queried in this file (e.g. the "apple",
    "tree canopy", "green leaf" prompts elsewhere). SAM3 mean-pools all text
    tokens into one query vector (design doc §5.2), so a long descriptive
    sentence like "a close-up macro photo of a round green fruit" dilutes the
    query across unrelated words and tends to produce near-zero scores on
    BOTH channels — observed directly in a real MinneApple run: 9/9 verified
    candidates had s_fruit and s_leaf both under 0.011, i.e. essentially
    noise-floor, on the old prompt wording. Pass the SAME concept word used
    for detection (e.g. fruit_prompt="apple") for maximum consistency; the
    generic "fruit"/"green leaf" defaults here are a safe fallback when the
    caller doesn't have a specific concept to hand.
    """
    # --- Unpack original coordinates and compute dimensions ---
    xmin_orig, ymin_orig, xmax_orig, ymax_orig = box
    w = xmax_orig - xmin_orig
    h = ymax_orig - ymin_orig

    # Find the absolute center point of the candidate
    cx = xmin_orig + (w / 2)
    cy = ymin_orig + (h / 2)

    # --- Apply a tight neighborhood scale expansion factor ---
    new_w = w * scale_factor
    new_h = h * scale_factor

    # --- Convert box coordinates to integers for array slicing and clamp to image boundaries ---
    img_h, img_w = image_np.shape[:2]

    xmin = int(max(0, cx - (new_w / 2)))
    ymin = int(max(0, cy - (new_h / 2)))
    xmax = int(min(img_w, cx + (new_w / 2)))
    ymax = int(min(img_h, cy + (new_h / 2)))

    # Guard against invalid or edge-case zero-sized dimensions
    if xmax <= xmin or ymax <= ymin:
        return 0.0, 0.0

    # --- Crop the candidate region directly out of the master image matrix ---
    crop_np = image_np[ymin:ymax, xmin:xmax]
    if crop_np.size == 0:
        return 0.0, 0.0

    # --- High-fidelity upsampling to prevent patch pixelation ---
    target_size = (256, 256)
    crop_resized = cv2.resize(crop_np, target_size, interpolation=cv2.INTER_CUBIC)
    img_pil = Image.fromarray(crop_resized)

    model = processor.model

    # --- Run the localized Fruit Audit ---
    inputs_fruit = processor(images=img_pil, text=fruit_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_fruit = model(**inputs_fruit)
    res_fruit = processor.post_process_instance_segmentation(out_fruit, threshold=0.001, target_sizes=inputs_fruit.get("original_sizes").tolist())[0]
    fruit_score = float(res_fruit["scores"][0]) if len(res_fruit["scores"]) > 0 else 0.0

    # --- Run the localized Leaf Audit ---
    inputs_leaf = processor(images=img_pil, text=leaf_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_leaf = model(**inputs_leaf)
    res_leaf = processor.post_process_instance_segmentation(out_leaf, threshold=0.001, target_sizes=inputs_leaf.get("original_sizes").tolist())[0]
    leaf_score = float(res_leaf["scores"][0]) if len(res_leaf["scores"]) > 0 else 0.0

    return fruit_score, leaf_score
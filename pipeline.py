"""
pipeline.py

Controls the execution of the loop using a full-frame spatial cross-reference gate.
"""

import numpy as np
import logging
import inference

# ==========================================
#   Diagnostics-Carrying Return Type
# ==========================================

class PassStats(int):
    """
    Behaves as the integer count of newly-accepted fruit nodes (so every existing
    caller that does `int(execute_pass(...))` or drops it straight into an f-string
    keeps working), while also carrying per-stage diagnostics for decompose-before-
    tuning analysis.

    Attributes (default 0):
        raw_proposals       - candidate boxes returned by SAM3 before NMS
        post_nms            - survivors after NMS (baseline IoU or dual-gate)
        post_verify         - candidates that reached the verification gate
        duplicates_rejected - candidates dropped by inter-pass dedup (vs prior fruit)
        accepted            - alias for the int value itself
    """
    def __new__(cls, value, raw_proposals=0, post_nms=0, post_verify=0, duplicates_rejected=0):
        obj = super().__new__(cls, value)
        obj.raw_proposals = raw_proposals
        obj.post_nms = post_nms
        obj.post_verify = post_verify
        obj.duplicates_rejected = duplicates_rejected
        obj.accepted = int(value)
        return obj

    def as_row(self):
        """One-line dict for CSV / table logging."""
        return {
            "raw_proposals": self.raw_proposals,
            "post_nms": self.post_nms,
            "post_verify": self.post_verify,
            "duplicates_rejected": self.duplicates_rejected,
            "accepted": self.accepted,
        }

# ==========================================
# 1. Helper functions
# ==========================================

def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0

    return interArea / float(boxAArea + boxBArea - interArea)

def compute_ioc(candidate_box, leaf_box):
    """
    Computes Intersection over Candidate (IoC).
    Determines what percentage of the small candidate box is
    overlapped by the larger background leaf mask.
    """
    xA = max(candidate_box[0], leaf_box[0])
    yA = max(candidate_box[1], leaf_box[1])
    xB = min(candidate_box[2], leaf_box[2])
    yB = min(candidate_box[3], leaf_box[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Calculate ONLY the area of the small candidate box
    candidate_area = (candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1])

    return inter_area / float(candidate_area) if candidate_area > 0 else 0.0

def register_and_verify_candidates(candidate_boxes, candidate_scores, leaf_boxes, graph, pass_number, iou_threshold=0.40):
    """
    Dedicated registration function. Cross-references fruit candidates against
    globally detected leaf maps to apply semantic verdicts without cropping.

    Returns (added_nodes, duplicates_rejected). The verification gate below is
    UNCHANGED -- only an inter-pass duplicate counter was added for diagnostics.
    """
    added_nodes = 0
    duplicates_rejected = 0

    for box, score in zip(candidate_boxes, candidate_scores):
        # --- Inter-Pass Deduplication: Skip if this box overlaps an existing valid object ---
        is_duplicate = False
        for existing_node in graph.nodes.values():
            # Only deduplicate against confirmed target detections ("fruit")
            if existing_node.classification == "fruit":
                # Defensive check: safely grab coordinate array whether named .bbox or .box
                existing_box = getattr(existing_node, 'bbox', getattr(existing_node, 'box', None))

                if existing_box is not None and compute_iou(box, existing_box) > iou_threshold:
                    is_duplicate = True
                    break

        if is_duplicate:
            logging.info("Cross-pass duplicate detected. Skipping node registration.")
            duplicates_rejected += 1
            continue

        # --- REGISTER: Initialize the candidate node in our database ---
        node_id = graph.add_candidate(box, score, found_in_pass=pass_number)

        # --- VERIFY: Find the maximum containment within the global leaf detection map ---
        max_leaf_ioc = 0.0
        if len(leaf_boxes) > 0:
            for l_box in leaf_boxes:
                ioc = compute_ioc(box, l_box)
                if ioc > max_leaf_ioc:
                    max_leaf_ioc = ioc

        # --- Save spatial scores and transition classification verdicts ---
        fruit_verification_score = float(score)
        leaf_verification_score = float(max_leaf_ioc)

        # --- OCCLUSION-AWARE LOGIC GATE ---
        graph.update_verdict(node_id, fruit_score=fruit_verification_score, leaf_score=leaf_verification_score)

        # 1. Clear Foliage Veto: Strong containment or leaf profile dominant
        if leaf_verification_score > 0.40 and (leaf_verification_score > fruit_verification_score or leaf_verification_score > 0.80):
            graph.nodes[node_id].classification = "leaf"

        # 2. Ambiguous Noise/Clutter Guard (Active on High-Recall Late Passes)
        elif pass_number >= 3 and fruit_verification_score < 0.35:
            graph.nodes[node_id].classification = "spurious"

        # 3. Clean Match Earned
        elif fruit_verification_score >= 0.25:
            graph.nodes[node_id].classification = "fruit"

        else:
            graph.nodes[node_id].classification = "spurious"

        added_nodes += 1

    return added_nodes, duplicates_rejected

# ==========================================
# 2. Coordinate Space Translations
# ==========================================
def initialize_canopy_roi(processor, img_np, graph):
    """
    [PASS 0] Locates the main tree canopy bounding box globally.
    Gates subsequent micro discovery loops to strip out soil and sky noise.
    """
    if hasattr(graph, "tree_roi") and graph.tree_roi is not None:
        return graph.tree_roi

    logging.info("Executing Pass 0: Initializing Canopy ROI Anchor...")
    img_h, img_w = img_np.shape[:2]

    # Run a global sweep with a high threshold to find the primary tree structure
    tree_boxes, _ = global_engine(processor, img_np, conf=0.40, prompt="tree canopy")

    if len(tree_boxes) > 0:
        xmin_t = int(max(0, np.min(tree_boxes[:, 0])))
        ymin_t = int(max(0, np.min(tree_boxes[:, 1])))
        xmax_t = int(min(img_w, np.max(tree_boxes[:, 2])))
        ymax_t = int(min(img_h, np.max(tree_boxes[:, 3])))
        graph.tree_roi = [xmin_t, ymin_t, xmax_t, ymax_t]
        logging.info(f"Canopy ROI locked at coordinates: {graph.tree_roi}")
    else:
        # Fallback to full frame if template mapping fails
        graph.tree_roi = [0, 0, img_w, img_h]
        logging.warning("Canopy anchor failed. Falling back to full image canvas.")

    return graph.tree_roi

def get_roi_relative_exemplars(graph, roi):
    """
    Translates absolute coordinates from the historical database down
    into local relative crop coordinates for the isolated tile engine process.
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    pos_boxes, neg_boxes = graph.get_exemplars()

    local_pos = []
    if pos_boxes.size > 0:
        for b in pos_boxes.tolist():
            if b[0] >= roi_x1 and b[1] >= roi_y1 and b[2] <= roi_x2 and b[3] <= roi_y2:
                local_pos.append([b[0] - roi_x1, b[1] - roi_y1, b[2] - roi_x1, b[3] - roi_y1])

    local_neg = []
    if neg_boxes.size > 0:
        for b in neg_boxes.tolist():
            if b[0] >= roi_x1 and b[1] >= roi_y1 and b[2] <= roi_x2 and b[3] <= roi_y2:
                local_neg.append([b[0] - roi_x1, b[1] - roi_y1, b[2] - roi_x1, b[3] - roi_y1])

    pos_arr = np.array(local_pos) if local_pos else None
    neg_arr = np.array(local_neg) if local_neg else None
    return pos_arr, neg_arr

def translate_roi_to_global(boxes, roi):
    """
    Transforms localized window boundaries back to the parent image coordinate plane.
    """
    if len(boxes) == 0:
        return np.empty((0, 4))
    roi_x1, roi_y1, _, _ = roi
    global_boxes = []
    for box in boxes:
        global_boxes.append([box[0] + roi_x1, box[1] + roi_y1, box[2] + roi_x1, box[3] + roi_y1])
    return np.array(global_boxes)

# ==========================================
# 3. Inference Block
# ==========================================

def run_inference_block(processor, image_np, conf, prompt, pos_boxes=None, neg_boxes=None, disable_size_filter=False):
    """
    Atomic block to run one SAM 3 pass
    """
    return inference.run_raw_inference(
        processor, image_np, conf, prompt=prompt, pos_boxes=pos_boxes, neg_boxes=neg_boxes, disable_size_filter=disable_size_filter
    )

# ==========================================
# 4. Execution Engines
# ==========================================

def global_engine(processor, image_np, conf, prompt, pos_boxes=None, neg_boxes=None, disable_size_filter=False):
    """
    Run inference on the entire picture
    """
    return run_inference_block(processor, image_np, conf, prompt, pos_boxes, neg_boxes, disable_size_filter)

def tiled_engine(processor, image_pil, confidence, clahe, prompt, pos_boxes=None, neg_boxes=None, global_leaf_boxes=None, disable_size_filter=False):
    """
    Run tiled inference
    """
    master_w, master_h = image_pil.size
    tile_size = min(master_w, master_h) // 2
    tile_size = max(160, min(tile_size, 640)) # Clamp between 160px and 640px bounds

    overlap = int(tile_size * 0.2) # Dynamic 20% spatial overlap safety zone
    stride = tile_size - overlap

    # --- Calculate tile sliding increments ---
    x_offsets = list(range(0, max(1, master_w - tile_size + 1), stride))
    y_offsets = list(range(0, max(1, master_h - tile_size + 1), stride))

    if x_offsets[-1] + tile_size < master_w:
        x_offsets.append(master_w - tile_size)
    if y_offsets[-1] + tile_size < master_h:
        y_offsets.append(master_h - tile_size)

    tiled_boxes_list = []
    tiled_scores_list = []

    for y_off in y_offsets:
        for x_off in x_offsets:
            tile_crop = image_pil.crop((x_off, y_off, x_off + tile_size, y_off + tile_size))
            tile_np = np.array(tile_crop)

            if clahe:
                tile_np = inference.apply_clahe(tile_np)

            # --- Map global memory tracking coordinates into this local window ---
            local_pos = []
            if pos_boxes is not None and len(pos_boxes) > 0:
                for b in pos_boxes:
                    if b[0] >= x_off and b[1] >= y_off and b[2] <= x_off + tile_size and b[3] <= y_off + tile_size:
                        local_pos.append([b[0] - x_off, b[1] - y_off, b[2] - x_off, b[3] - y_off])

            local_neg = []
            if neg_boxes is not None and len(neg_boxes) > 0:
                for b in neg_boxes:
                    if b[0] >= x_off and b[1] >= y_off and b[2] <= x_off + tile_size and b[3] <= y_off + tile_size:
                        local_neg.append([b[0] - x_off, b[1] - y_off, b[2] - x_off, b[3] - y_off])

            # --- DYNAMIC OPTIMIZATION: Filter and sample global leaves for this tile ---
            local_leaves = []
            if global_leaf_boxes is not None and len(global_leaf_boxes) > 0:
                for b in global_leaf_boxes:
                    if b[0] >= x_off and b[1] >= y_off and b[2] <= x_off + tile_size and b[3] <= y_off + tile_size:
                        local_leaves.append([b[0] - x_off, b[1] - y_off, b[2] - x_off, b[3] - y_off])

            # Memory Safety Guard: Cap at the top 15 local background inhibitors
            if len(local_leaves) > 15:
                local_leaves = local_leaves[:15]

            # Combine historical graph negative feedback with our active global leaf inhibitors
            combined_neg = local_neg + local_leaves

            local_pos_arr = np.array(local_pos) if local_pos else None
            local_neg_arr = np.array(combined_neg) if combined_neg else None

            # --- Run inference on the cropped tile using the atomic pass block ---
            tile_boxes, tile_scores = run_inference_block(
                processor, tile_np, confidence, prompt, pos_boxes=local_pos_arr, neg_boxes=local_neg_arr, disable_size_filter=disable_size_filter
            )

            # --- Translate the local boxes to global space ---
            if len(tile_boxes) > 0:
                for box, score in zip(tile_boxes, tile_scores):
                    xmin_l, ymin_l, xmax_l, ymax_l = box
                    tiled_boxes_list.append([xmin_l + x_off, ymin_l + y_off, xmax_l + x_off, ymax_l + y_off])
                    tiled_scores_list.append(score)

    boxes = np.array(tiled_boxes_list) if tiled_boxes_list else np.empty((0, 4))
    scores = np.array(tiled_scores_list) if tiled_scores_list else np.empty((0,))
    return boxes, scores

# ==========================================
# 5. Master Pipeline Control Thread
# ==========================================

def execute_pass(processor, image_pil, graph, conf, clahe, tiling, pass_number, prompt,
                 disable_size_filter=False, nms_mode="dualgate", use_concentric=False):
    """
    Runs one full pass of SAM3 pipeline
    Propose -> Register -> Verify -> Feedback

    nms_mode:
        "iou"      -> baseline cv2 IoU NMS (the validated 0.74/0.75 reference)
        "dualgate" -> IoU + size-guarded IoM containment NMS (default)
    use_concentric: forwarded to the dual-gate concentric sub-gate (default OFF;
        only the harness/ablation should turn it on).

    Returns a PassStats (an int subclass) so existing callers in app.py are unaffected.
    """
    img_np = np.array(image_pil)
    img_h, img_w = img_np.shape[:2]

    # --- RUN OR RETRIEVE CANOPY GATE [PASS 0] ---
    roi = initialize_canopy_roi(processor, img_np, graph)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    # Crop the PIL image tightly to the tree zone for the processing track
    roi_image_pil = image_pil.crop((roi_x1, roi_y1, roi_x2, roi_y2))
    roi_img_np = np.array(roi_image_pil)

    # --- GENERATE GLOBAL LEAF MAP (Pass 1 Cache Optimization) ---
    # Check if the persistent graph state already holds the background leaf matrix
    if not hasattr(graph, "cached_leaf_boxes") or graph.cached_leaf_boxes is None or pass_number == 1:
        logging.info("Generating global leaf map...")
        leaf_boxes, _ = global_engine(processor, roi_img_np, conf, prompt="green leaf")
        graph.cached_leaf_boxes = leaf_boxes
    else:
        logging.info("Foliage layer unchanged. Reusing cached global leaf map coordinates.")
        leaf_boxes = graph.cached_leaf_boxes

    # --- READ PAST CONTEXT FEEDBACK FROM THE GRAPH ---
    pos_boxes, neg_boxes = get_roi_relative_exemplars(graph, roi) if pass_number > 1 else (None, None)

    master_boxes = []
    master_scores = []

    # --- RUN COMPOSABLE PROPOSAL GENERATION ---
    if tiling:
        logging.info("Tiled inference started...")
        # Only pass background leaves if pass_number > 1
        current_leaf_inhibitors = leaf_boxes if pass_number > 1 else None

        candidate_boxes, candidate_scores = tiled_engine(
            processor, roi_image_pil, conf, clahe, prompt,
            pos_boxes=pos_boxes, neg_boxes=neg_boxes, global_leaf_boxes=current_leaf_inhibitors,
            disable_size_filter=disable_size_filter
        )
        logging.info("Tiled inference ended...")
    else:
        logging.info("Global inference started...")
        if pass_number > 1:
            sampled_leaves = leaf_boxes
            if len(leaf_boxes) > 20:
                indices = np.random.choice(len(leaf_boxes), 20, replace=False)
                sampled_leaves = leaf_boxes[indices]

            combined_neg_global = neg_boxes.tolist() if neg_boxes is not None else []
            if len(sampled_leaves) > 0:
                combined_neg_global.extend(sampled_leaves.tolist())

            global_neg_arr = np.array(combined_neg_global) if combined_neg_global else None
        else:
            global_neg_arr = None

        img_to_process = inference.apply_clahe(roi_img_np) if clahe else roi_img_np
        candidate_boxes, candidate_scores = global_engine(
            processor, img_to_process, conf, prompt, pos_boxes=pos_boxes, neg_boxes=global_neg_arr,
            disable_size_filter=disable_size_filter
        )
        logging.info("Global inference ended...")

    if len(candidate_boxes) > 0:
        master_boxes.append(candidate_boxes)
        master_scores.append(candidate_scores)

    if not master_boxes:
        logging.info("No new structures located in this generation sweep.")
        return PassStats(0)

    all_boxes = np.vstack(master_boxes)
    all_scores = np.concatenate(master_scores)
    raw_proposal_count = len(all_boxes)

    # Clean duplicates locally within current ROI coordinate context
    if nms_mode == "dualgate":
        roi_boxes_final, roi_scores_final = inference.apply_nms_dualgate(
            all_boxes, all_scores, conf, use_concentric=use_concentric
        )
    else:
        roi_boxes_final, roi_scores_final = inference.apply_nms(all_boxes, all_scores, conf)
    post_nms_count = len(roi_boxes_final)

    # --- STEP 5: TRANSLATE REGIONS BACK TO FULL-FRAME GLOBAL PLANE ---
    global_candidate_boxes = translate_roi_to_global(roi_boxes_final, roi)
    global_leaf_boxes = translate_roi_to_global(leaf_boxes, roi)

    # --- GLOBAL CROSS-REFERENCE VERIFICATION ---
    added_nodes, duplicates_rejected = register_and_verify_candidates(
        candidate_boxes=global_candidate_boxes,
        candidate_scores=roi_scores_final,
        leaf_boxes=global_leaf_boxes,
        graph=graph,
        pass_number=pass_number,
        iou_threshold=0.40
    )

    return PassStats(
        added_nodes,
        raw_proposals=raw_proposal_count,
        post_nms=post_nms_count,
        post_verify=post_nms_count,
        duplicates_rejected=duplicates_rejected,
    )
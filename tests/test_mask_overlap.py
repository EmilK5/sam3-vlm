"""
Tests for the opt-in mask-based overlap switch (cfg.overlap_mode="mask"):

  - pipeline.compute_mask_iou on box-cropped masks (frame-independent).
  - inference.apply_nms_dualgate(masks=...): Gate A/B measured on instance masks
    (box path byte-for-byte unchanged when masks=None).
  - register_and_verify_candidates: cross-pass dedup via mask IoU when both the
    candidate and the existing node carry masks.
  - execute_pass end-to-end in mask mode: masks flow SAM3 -> NMS -> node.mask,
    and cross-pass dedup reinforces instead of duplicating.

inference/pipeline import torch/transformers at module top; both are stubbed in
sys.modules so everything runs CPU-only (inference's own functions under test
are pure numpy).
"""

import dataclasses
import sys
import types

import numpy as np
import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph


# ----------------------- real inference w/ stubbed torch+transformers -----------------------

@pytest.fixture
def inference_module():
    saved = {n: sys.modules.get(n) for n in ("torch", "transformers", "inference")}
    sys.modules["torch"] = types.ModuleType("torch")
    tf = types.ModuleType("transformers")
    tf.Sam3Model = object
    tf.Sam3Processor = object
    sys.modules["transformers"] = tf
    sys.modules.pop("inference", None)

    import inference

    try:
        yield inference
    finally:
        sys.modules.pop("inference", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture
def pipeline_module():
    """Real pipeline against a stub `inference` whose run_raw_inference returns one
    box (+ box-shaped mask when asked) and whose NMS delegates to the real
    apply_nms_dualgate (imported with stubbed torch/transformers)."""
    saved = {n: sys.modules.get(n) for n in ("torch", "transformers", "inference", "pipeline")}
    sys.modules["torch"] = types.ModuleType("torch")
    tf = types.ModuleType("transformers")
    tf.Sam3Model = object
    tf.Sam3Processor = object
    sys.modules["transformers"] = tf
    sys.modules.pop("inference", None)

    import inference as real_inference
    real_nms_dualgate = real_inference.apply_nms_dualgate

    inf = types.ModuleType("inference")

    def run_raw_inference(processor, image_np, confidence, prompt, pos_boxes=None,
                          neg_boxes=None, disable_size_filter=False, return_masks=False):
        if prompt == "green leaf":
            boxes, scores = np.empty((0, 4)), np.empty((0,))
            return (boxes, scores, []) if return_masks else (boxes, scores)
        boxes = np.array([[10.0, 10.0, 30.0, 30.0]])
        scores = np.array([0.9])
        if return_masks:
            h, w = image_np.shape[:2]
            m = np.zeros((h, w), dtype=bool)
            m[10:30, 10:30] = True
            return boxes, scores, [m]
        return boxes, scores

    inf.run_raw_inference = run_raw_inference
    inf.apply_nms_dualgate = real_nms_dualgate
    inf.apply_nms = real_nms_dualgate
    inf.apply_clahe = lambda img: img
    sys.modules["inference"] = inf
    sys.modules.pop("pipeline", None)

    import pipeline

    try:
        yield pipeline
    finally:
        sys.modules.pop("pipeline", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


# ----------------------- compute_mask_iou -----------------------

def test_mask_iou_identical_masks(pipeline_module):
    box = [10, 10, 20, 20]
    mask = np.ones((10, 10), dtype=bool)
    assert pipeline_module.compute_mask_iou(box, mask, box, mask) == pytest.approx(1.0)


def test_mask_iou_disjoint_masks_in_same_box(pipeline_module):
    # Same bounding box, complementary halves: box IoU would be 1.0, mask IoU 0.
    box = [0, 0, 10, 10]
    left = np.zeros((10, 10), dtype=bool)
    left[:, :5] = True
    right = np.zeros((10, 10), dtype=bool)
    right[:, 5:] = True
    assert pipeline_module.compute_mask_iou(box, left, box, right) == 0.0


def test_mask_iou_partial_overlap_and_disjoint_boxes(pipeline_module):
    a = np.ones((10, 10), dtype=bool)
    b = np.ones((10, 10), dtype=bool)
    # b shifted right by 5: inter 50, union 150
    assert pipeline_module.compute_mask_iou([0, 0, 10, 10], a, [5, 0, 15, 10], b) == pytest.approx(50 / 150)
    assert pipeline_module.compute_mask_iou([0, 0, 10, 10], a, [50, 50, 60, 60], b) == 0.0


# ----------------------- apply_nms_dualgate with masks -----------------------

def _strip_masks(frame_hw, box, cols):
    """A full-frame mask covering `cols` (slice) of `box`."""
    m = np.zeros(frame_hw, dtype=bool)
    x1, y1, x2, y2 = box
    m[y1:y2, cols] = True
    return m


def test_mask_mode_keeps_boxes_whose_masks_do_not_overlap(inference_module):
    boxes = np.array([[0, 0, 20, 20], [0, 2, 20, 22]], dtype=float)
    scores = np.array([0.9, 0.8])

    # Box mode: IoU ~0.82 -> the second box is suppressed (unchanged behavior).
    kept_boxes, _ = inference_module.apply_nms_dualgate(boxes, scores, 0.5)
    assert len(kept_boxes) == 1

    # Mask mode: complementary vertical strips -> mask IoU 0 -> both survive.
    masks = [_strip_masks((40, 40), (0, 0, 20, 20), slice(0, 10)),
             _strip_masks((40, 40), (0, 2, 20, 22), slice(10, 20))]
    kept_boxes, kept_scores, idx = inference_module.apply_nms_dualgate(
        boxes, scores, 0.5, masks=masks, return_indices=True)
    assert len(kept_boxes) == 2
    assert sorted(idx.tolist()) == [0, 1]


def test_mask_mode_suppresses_duplicate_masks(inference_module):
    boxes = np.array([[0, 0, 20, 20], [0, 2, 20, 22]], dtype=float)
    scores = np.array([0.9, 0.8])
    same = _strip_masks((40, 40), (0, 2, 20, 20), slice(0, 20))  # shared region
    kept_boxes, _, idx = inference_module.apply_nms_dualgate(
        boxes, scores, 0.5, masks=[same, same], return_indices=True)
    assert len(kept_boxes) == 1
    assert idx.tolist() == [0]            # the higher-score one wins


def test_mask_list_stays_aligned_after_confidence_filter(inference_module):
    # The middle box falls below conf and must take its mask with it.
    boxes = np.array([[0, 0, 20, 20], [0, 0, 20, 20], [0, 2, 20, 22]], dtype=float)
    scores = np.array([0.9, 0.3, 0.8])
    masks = [_strip_masks((40, 40), (0, 0, 20, 20), slice(0, 10)),
             _strip_masks((40, 40), (0, 0, 20, 20), slice(0, 20)),   # would suppress both
             _strip_masks((40, 40), (0, 2, 20, 22), slice(10, 20))]
    kept_boxes, _, idx = inference_module.apply_nms_dualgate(
        boxes, scores, 0.5, masks=masks, return_indices=True)
    assert sorted(idx.tolist()) == [0, 2]  # disjoint masks -> both kept, 0.3 dropped


# ----------------------- dedup via mask IoU -----------------------

def _register(pipeline_module, graph, cfg, box, mask, pass_number):
    return pipeline_module.register_and_verify_candidates(
        np.array([box]), np.array([0.9]), leaf_boxes=np.empty((0, 4)),
        graph=graph, pass_number=pass_number, cfg=cfg,
        candidate_masks=[mask],
    )


def test_dedup_uses_mask_iou_when_masks_present(pipeline_module):
    cfg = dataclasses.replace(Config(), verifier_mode="off", overlap_mode="mask")
    box = [0.0, 0.0, 10.0, 10.0]
    left = np.zeros((10, 10), dtype=bool)
    left[:, :5] = True
    right = np.zeros((10, 10), dtype=bool)
    right[:, 5:] = True

    # Same box, disjoint masks: two distinct objects -> both registered.
    g = OrchardGraph()
    _register(pipeline_module, g, cfg, box, left, 1)
    added, dup = _register(pipeline_module, g, cfg, box, right, 2)
    assert (added, dup) == (1, 0) and len(g.nodes) == 2

    # Same box, same mask: a true cross-pass duplicate -> reinforced.
    g2 = OrchardGraph()
    _register(pipeline_module, g2, cfg, box, left, 1)
    added, dup = _register(pipeline_module, g2, cfg, box, left, 2)
    assert (added, dup) == (0, 1) and len(g2.nodes) == 1
    assert list(g2.nodes.values())[0].support == 2


# ----------------------- execute_pass end-to-end in mask mode -----------------------

def test_execute_pass_mask_mode_stores_and_dedups_masks(pipeline_module):
    cfg = dataclasses.replace(Config(), verifier_mode="off", overlap_mode="mask")
    graph = OrchardGraph()
    img = Image.new("RGB", (100, 100))

    n1 = pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False,
                                      tiling=False, pass_number=1, prompt="green fruit",
                                      cfg=cfg, roi_override=(0, 0, 100, 100))
    assert int(n1) == 1
    node = list(graph.nodes.values())[0]
    assert node.mask is not None
    assert node.mask.shape == (20, 20) and node.mask.all()   # cropped to its box

    # Second pass re-detects the same instance -> mask dedup reinforces the track.
    n2 = pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False,
                                      tiling=False, pass_number=2, prompt="green fruit",
                                      cfg=cfg, roi_override=(0, 0, 100, 100))
    assert int(n2) == 0
    assert len(graph.nodes) == 1
    assert node.support == 2


def test_box_mode_leaves_node_masks_unset(pipeline_module):
    graph = OrchardGraph()
    img = Image.new("RGB", (100, 100))
    pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False,
                                 tiling=False, pass_number=1, prompt="green fruit",
                                 cfg=Config(), roi_override=(0, 0, 100, 100))
    assert all(n.mask is None for n in graph.nodes.values())

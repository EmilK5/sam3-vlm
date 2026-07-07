"""
Regression test for --iou-threshold/--iom-threshold (Config.nms_iou_threshold /
nms_iom_threshold): pipeline.execute_pass must forward them into
inference.apply_nms_dualgate's own iou_threshold/iom_threshold, not silently
keep the hardcoded 0.40/0.90 defaults.

inference/pipeline import torch/transformers at module top; both are stubbed so
this runs CPU-only.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture
def pipeline_module():
    saved = {n: sys.modules.get(n) for n in ("torch", "transformers", "inference", "pipeline")}
    sys.modules["torch"] = types.ModuleType("torch")
    tf = types.ModuleType("transformers")
    tf.Sam3Model = object
    tf.Sam3Processor = object
    sys.modules["transformers"] = tf
    sys.modules.pop("inference", None)
    sys.modules.pop("pipeline", None)

    import pipeline

    try:
        yield pipeline
    finally:
        sys.modules.pop("inference", None)
        sys.modules.pop("pipeline", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_execute_pass_forwards_nms_thresholds_to_apply_nms_dualgate(pipeline_module, monkeypatch):
    from graph import OrchardGraph
    from PIL import Image
    from config import Config

    captured = {}

    def _fake_global_engine(processor, img, conf, prompt, pos_boxes=None, neg_boxes=None,
                           disable_size_filter=False, return_masks=False):
        # One candidate box so apply_nms_dualgate actually runs (not short-circuited
        # by the empty-boxes early return).
        return np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.9])

    def _fake_apply_nms_dualgate(boxes, scores, confidence, **kw):
        captured["iou_threshold"] = kw.get("iou_threshold")
        captured["iom_threshold"] = kw.get("iom_threshold")
        captured["gate_mode"] = kw.get("gate_mode")
        return boxes, scores

    monkeypatch.setattr(pipeline_module, "global_engine", _fake_global_engine)
    monkeypatch.setattr(pipeline_module.inference, "apply_nms_dualgate", _fake_apply_nms_dualgate)

    graph = OrchardGraph()
    img = Image.new("RGB", (64, 64))
    pipeline_module.execute_pass(
        processor=None, image_pil=img, graph=graph, conf=0.35, clahe=False,
        tiling=False, pass_number=1, prompt="car", cfg=Config(), use_canopy_roi=False,
        gate_mode="iom_only", nms_iou_threshold=0.22, nms_iom_threshold=0.77,
    )

    assert captured["iou_threshold"] == 0.22
    assert captured["iom_threshold"] == 0.77
    assert captured["gate_mode"] == "iom_only"


def test_execute_pass_forwards_cross_pass_dedup_settings(pipeline_module, monkeypatch):
    """execute_pass's cross_pass_dedup_metric/_threshold must reach
    register_and_verify_candidates as dedup_metric/iou_threshold -- a
    DIFFERENT mechanism from the intra-pass nms_iou_threshold/nms_iom_threshold
    covered above (inter-pass dedup vs. intra-pass NMS)."""
    from graph import OrchardGraph
    from PIL import Image
    from config import Config

    captured = {}

    def _fake_global_engine(processor, img, conf, prompt, pos_boxes=None, neg_boxes=None,
                           disable_size_filter=False, return_masks=False):
        return np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.9])

    def _fake_register(candidate_boxes, candidate_scores, leaf_boxes, graph, pass_number,
                       iou_threshold=0.40, dedup_metric="iou", **kw):
        captured["iou_threshold"] = iou_threshold
        captured["dedup_metric"] = dedup_metric
        return 1, 0

    monkeypatch.setattr(pipeline_module, "global_engine", _fake_global_engine)
    monkeypatch.setattr(pipeline_module, "register_and_verify_candidates", _fake_register)

    graph = OrchardGraph()
    img = Image.new("RGB", (64, 64))
    pipeline_module.execute_pass(
        processor=None, image_pil=img, graph=graph, conf=0.35, clahe=False,
        tiling=False, pass_number=1, prompt="car", cfg=Config(), use_canopy_roi=False,
        cross_pass_dedup_metric="iom", cross_pass_dedup_threshold=0.85,
    )

    assert captured["iou_threshold"] == 0.85
    assert captured["dedup_metric"] == "iom"

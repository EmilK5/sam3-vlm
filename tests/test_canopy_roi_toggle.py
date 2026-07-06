"""
Regression test for use_canopy_roi (--canopy-roi in eval/run_eval.py):
pipeline.initialize_canopy_roi must skip the "tree canopy" SAM3 sweep entirely
when use_canopy=False, anchoring the ROI to the full frame instead -- for
datasets with no canopy concept (CARPK, CountBench, PixMo), where that sweep
could only ever return a spurious/empty match and wastes a SAM3 call.

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


class _FakeGraph:
    tree_roi = None


def test_use_canopy_false_skips_sam3_and_uses_full_frame(pipeline_module, monkeypatch):
    graph = _FakeGraph()
    img_np = np.zeros((50, 80, 3), dtype=np.uint8)  # H=50, W=80

    def _boom(*a, **kw):
        raise AssertionError("global_engine (SAM3) must not be called when use_canopy=False")
    monkeypatch.setattr(pipeline_module, "global_engine", _boom)

    roi = pipeline_module.initialize_canopy_roi(processor=None, img_np=img_np, graph=graph, use_canopy=False)
    assert roi == [0, 0, 80, 50]
    assert graph.tree_roi == [0, 0, 80, 50]


def test_use_canopy_true_default_still_calls_sam3(pipeline_module, monkeypatch):
    graph = _FakeGraph()
    img_np = np.zeros((50, 80, 3), dtype=np.uint8)
    called = {}

    def _fake_global_engine(processor, img, conf, prompt):
        called["prompt"] = prompt
        return np.empty((0, 4)), np.empty((0,))
    monkeypatch.setattr(pipeline_module, "global_engine", _fake_global_engine)

    roi = pipeline_module.initialize_canopy_roi(processor=None, img_np=img_np, graph=graph)
    assert called["prompt"] == "tree canopy"
    assert roi == [0, 0, 80, 50]  # empty tree_boxes -> fallback to full frame anyway


def test_execute_pass_use_canopy_roi_false_never_prompts_tree_canopy(pipeline_module, monkeypatch):
    """execute_pass's own use_canopy_roi=False must reach initialize_canopy_roi
    and must not count the (skipped) canopy sweep as a SAM3 call."""
    from graph import OrchardGraph
    from PIL import Image
    from config import Config

    prompts_seen = []

    def _fake_global_engine(processor, img, conf, prompt, pos_boxes=None, neg_boxes=None,
                           disable_size_filter=False, return_masks=False):
        prompts_seen.append(prompt)
        return np.empty((0, 4)), np.empty((0,))
    monkeypatch.setattr(pipeline_module, "global_engine", _fake_global_engine)

    graph = OrchardGraph()
    img = Image.new("RGB", (64, 64))
    stats = pipeline_module.execute_pass(
        processor=None, image_pil=img, graph=graph, conf=0.35, clahe=False,
        tiling=False, pass_number=1, prompt="car", cfg=Config(), use_canopy_roi=False,
    )
    assert "tree canopy" not in prompts_seen
    assert graph.tree_roi == [0, 0, 64, 64]
    # n_sam_calls: leaf map (1) + global proposal (1); canopy sweep skipped, so
    # NOT 3 -- this is what would silently regress if use_canopy_roi stopped
    # being forwarded from execute_pass into initialize_canopy_roi.
    assert stats.n_sam_calls == 2


def test_execute_pass_default_still_counts_canopy_call(pipeline_module, monkeypatch):
    """Default (use_canopy_roi=True, unset): unchanged baseline behavior --
    canopy runs on a fresh graph and is counted as a real SAM3 call."""
    from graph import OrchardGraph
    from PIL import Image
    from config import Config

    prompts_seen = []

    def _fake_global_engine(processor, img, conf, prompt, pos_boxes=None, neg_boxes=None,
                           disable_size_filter=False, return_masks=False):
        prompts_seen.append(prompt)
        return np.empty((0, 4)), np.empty((0,))
    monkeypatch.setattr(pipeline_module, "global_engine", _fake_global_engine)

    graph = OrchardGraph()
    img = Image.new("RGB", (64, 64))
    stats = pipeline_module.execute_pass(
        processor=None, image_pil=img, graph=graph, conf=0.35, clahe=False,
        tiling=False, pass_number=1, prompt="green fruit", cfg=Config(),
    )
    assert "tree canopy" in prompts_seen
    # canopy (1) + leaf map (1) + global proposal (1)
    assert stats.n_sam_calls == 3

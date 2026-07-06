"""
Regression test for the iom_only containment-guard bug: a box fully contained
inside another (iom ~= 1.0) was surviving apply_nms_dualgate(gate_mode="iom_only")
whenever its area was less than min_size_ratio_for_containment (default 0.40) of
the containing box -- e.g. a thin sliver box inside a CARPK car box. The
size-ratio guard exists to protect a genuinely distinct small object nested in a
larger box (the citrus fruit-in-cluster case), which only applies to the "dual"
gate_mode; "iom_only" now suppresses on containment alone, with no size guard.

inference imports torch/transformers at module top; both are stubbed so this
runs CPU-only against the real (pure-numpy) apply_nms_dualgate.
"""

import sys
import types

import numpy as np
import pytest


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


def test_iom_only_suppresses_small_box_fully_inside_large_box(inference_module):
    # Big box area 20000, thin sliver fully inside it, area 2000 -> size_ratio 0.1,
    # well below the 0.40 containment guard used by "dual".
    boxes = [[100, 100, 200, 300], [190, 100, 200, 300]]
    scores = [0.96, 0.72]

    kept_boxes, _ = inference_module.apply_nms_dualgate(
        boxes, scores, confidence=0.3, gate_mode="iom_only")
    assert len(kept_boxes) == 1
    assert kept_boxes[0].tolist() == [100, 100, 200, 300]


def test_dual_gate_mode_unchanged_for_same_scenario(inference_module):
    # "dual" (the citrus-validated default) must keep protecting a small,
    # comparably-undersized nested box exactly as before -- byte-identical.
    boxes = [[100, 100, 200, 300], [190, 100, 200, 300]]
    scores = [0.96, 0.72]

    kept_boxes, _ = inference_module.apply_nms_dualgate(
        boxes, scores, confidence=0.3, gate_mode="dual")
    assert len(kept_boxes) == 2

"""
Tests for the LookROIA sensing action (step 7.1; untiled since v2 step 8.1).

LookROIA expands a VLM-proposed ROI, guards it, and runs ONE untiled SAM3 query
restricted to it via pipeline.execute_pass. The ROI is a SENSING TARGET ONLY and
must never become a graph node. execute_pass pulls in torch, so we stub `pipeline`
in sys.modules (same pattern as test_actions.py) to keep this CPU-only.
"""

import sys
import types

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.actions import (
    LookROIA, ActionContext, execute,
    _expand_and_clamp, _roi_iou,
)


# ----------------------- fake pipeline.execute_pass -----------------------

@pytest.fixture
def fake_pipeline():
    """Fake `pipeline` whose execute_pass records its call kwargs and adds `n_new`
    fruit nodes with a distinctive width-5 box (so we can prove no node ever came
    from a LookROIA's ROI coordinates)."""
    saved = sys.modules.get("pipeline")
    mod = types.ModuleType("pipeline")
    calls = []

    def execute_pass(**kw):
        calls.append(kw)
        for j in range(execute_pass.n_new):
            nid = kw["graph"].add_candidate([j, 0, j + 5, 5], 0.9, kw["pass_number"])
            kw["graph"].nodes[nid].classification = "fruit"
        return execute_pass.n_new

    execute_pass.n_new = 3
    mod.execute_pass = execute_pass
    mod.calls = calls
    sys.modules["pipeline"] = mod
    try:
        yield mod
    finally:
        if saved is None:
            sys.modules.pop("pipeline", None)
        else:
            sys.modules["pipeline"] = saved


def _ctx(**overrides):
    base = dict(
        processor=object(), image_pil=Image.new("RGB", (200, 200)),
        graph=OrchardGraph(), cfg=Config(), oracle=None, query_set=None,
        discovery=DiscoveryCurve(), partition=[(0, 0, 200, 200)],
    )
    base.update(overrides)
    return ActionContext(**base)


# ----------------------- pure helpers -----------------------

def test_expand_and_clamp_grows_by_margin():
    # 40x40 box, 10% margin -> +4 px each side.
    assert _expand_and_clamp((100, 100, 140, 140), 0.10, 200, 200) == (96, 96, 144, 144)


def test_expand_and_clamp_clips_to_image():
    # A big margin runs off the top-left and bottom-right; clamp to [0, W/H].
    assert _expand_and_clamp((0, 0, 20, 20), 0.5, 200, 200) == (0, 0, 30, 30)
    assert _expand_and_clamp((180, 180, 200, 200), 0.5, 200, 200) == (170, 170, 200, 200)


def test_roi_iou_basic():
    assert _roi_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _roi_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert _roi_iou((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)


# ----------------------- sensing -----------------------

def test_look_senses_expanded_roi_untiled(fake_pipeline):
    ctx = _ctx()
    n = execute(LookROIA(region=(40, 40, 120, 120)), ctx)
    assert isinstance(n, int) and n == 3
    call = fake_pipeline.calls[-1]
    assert call["roi_override"] == (32, 32, 128, 128)   # +10% margin, clamped
    assert call["tiling"] is False                       # v2: one untiled query
    assert ctx.n_passes == 1                             # counts as a sensing pass
    assert ctx.sensed_rois == [(32, 32, 128, 128)]       # recorded for dedup


def test_look_roi_box_never_becomes_a_node(fake_pipeline):
    # Grounding invariant: neither the proposed box nor the expanded ROI may
    # appear as a candidate; every node must originate from the SAM3 stub.
    ctx = _ctx()
    execute(LookROIA(region=(40, 40, 120, 120)), ctx)
    node_boxes = [tuple(node.box) for node in ctx.graph.nodes.values()]
    assert (32.0, 32.0, 128.0, 128.0) not in node_boxes    # expanded ROI
    assert (40.0, 40.0, 120.0, 120.0) not in node_boxes    # original proposal
    assert node_boxes and all((b[2] - b[0]) == 5.0 for b in node_boxes)  # all from stub


# ----------------------- guards (no-op, no model call) -----------------------

def test_look_too_small_is_noop(fake_pipeline):
    ctx = _ctx()
    n = execute(LookROIA(region=(100, 100, 110, 110)), ctx)   # ~12px after margin < 32
    assert n == 0
    assert fake_pipeline.calls == []            # execute_pass never called
    assert len(ctx.graph.nodes) == 0
    assert ctx.n_passes == 0 and ctx.sensed_rois == []


def test_look_past_max_depth_is_noop(fake_pipeline):
    # 40x40 -> 48x48 after margin: side 48 >= min_size 32, but area 2304 <
    # frame/16 = 2500, so it trips the depth (area-floor) guard, not min-size.
    ctx = _ctx()
    n = execute(LookROIA(region=(100, 100, 140, 140)), ctx)
    assert n == 0
    assert fake_pipeline.calls == []
    assert len(ctx.graph.nodes) == 0


def test_look_duplicate_roi_is_noop(fake_pipeline):
    ctx = _ctx()
    execute(LookROIA(region=(40, 40, 120, 120)), ctx)          # senses ROI R
    n2 = execute(LookROIA(region=(41, 41, 121, 121)), ctx)     # overlaps R (IoU>0.7)
    assert n2 == 0
    assert len(fake_pipeline.calls) == 1                       # only the first sensed
    assert len(ctx.graph.nodes) == 3

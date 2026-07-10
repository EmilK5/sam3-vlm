"""
Tests for step 8.1 (v2 action space): a LookROIA is exactly ONE untiled SAM3
pass over the margin-expanded ROI; the legacy actions (TileQueryA, SubdivideA,
VerifyA) are gone; and the heuristic fallback emits only LookROIA/StopA,
stopping on a saturated discovery curve.

pipeline.execute_pass pulls in torch, so it is stubbed in sys.modules (same
pattern as test_actions.py). All boxes are global-frame xyxy pixels.
"""

import sys
import types

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent import actions, policy_heuristic
from agent.actions import LookROIA, StopA, ActionContext, execute


# ----------------------- fake pipeline.execute_pass -----------------------

@pytest.fixture
def fake_pipeline():
    saved = sys.modules.get("pipeline")
    mod = types.ModuleType("pipeline")
    calls = []

    def execute_pass(**kw):
        calls.append(kw)
        nid = kw["graph"].add_candidate([1, 1, 6, 6], 0.9, kw["pass_number"])
        kw["graph"].nodes[nid].classification = "fruit"
        return 1

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


# ----------------------- look = ONE untiled pass -----------------------

def test_look_is_exactly_one_untiled_pass_on_the_expanded_roi(fake_pipeline):
    ctx = _ctx()
    n = execute(LookROIA(region=(40, 40, 120, 120)), ctx)
    assert n == 1
    assert len(fake_pipeline.calls) == 1                 # exactly one execute_pass
    call = fake_pipeline.calls[0]
    assert call["tiling"] is False                       # v2: no tiling inside an ROI
    assert call["roi_override"] == (32, 32, 128, 128)    # +10% margin, clamped


def test_look_guards_still_noop_without_model_calls(fake_pipeline):
    ctx = _ctx()
    assert execute(LookROIA(region=(100, 100, 110, 110)), ctx) == 0  # min-size
    execute(LookROIA(region=(40, 40, 120, 120)), ctx)                # senses
    assert execute(LookROIA(region=(41, 41, 121, 121)), ctx) == 0    # duplicate
    assert len(fake_pipeline.calls) == 1


# ----------------------- legacy actions removed -----------------------

def test_legacy_action_names_are_gone_from_the_module():
    for name in ("TileQueryA", "SubdivideA", "VerifyA",
                 "subdivide_region", "_execute_subdivide", "_execute_verify"):
        assert not hasattr(actions, name)


# ----------------------- heuristic fallback: look/stop only -----------------------

def _phi(**over):
    base = dict(
        K=0, n_t=0, D=[], U=0.0, ids=[], centers=[], area=[], classification=[],
        w=[], s=[], k=[], delta=[], tiling_status=False, remaining_budget=12,
    )
    base.update(over)
    return base


def test_heuristic_emits_only_look_or_stop():
    cfg = Config()
    partition = [(0, 0, 100, 100)]
    scenarios = [
        _phi(D=[], K=0),                                  # nothing sensed yet
        _phi(D=[5, 5, 5], U=9.0, K=2,                     # still discovering
             ids=["a", "b"], centers=[[10, 10], [80, 80]],
             area=[100.0, 100.0], classification=["fruit", "unresolved"],
             w=[1.0, 0.2], s=[0.9, 0.4], k=[2, 1], delta=[0.0, 0.5]),
        _phi(D=[0, 0, 0], U=0.1, K=1,                     # saturated + settled
             ids=["a"], centers=[[10, 10]], area=[100.0],
             classification=["fruit"], w=[1.0], s=[0.9], k=[3], delta=[0.0]),
    ]
    for phi in scenarios:
        assert isinstance(policy_heuristic.choose(phi, partition, cfg), (LookROIA, StopA))


def test_heuristic_stops_on_saturated_curve():
    phi = _phi(D=[0, 0, 0], U=0.1, K=1,
               ids=["a"], centers=[[10, 10]], area=[100.0],
               classification=["fruit"], w=[1.0], s=[0.9], k=[3], delta=[0.0])
    action = policy_heuristic.choose(phi, [(0, 0, 100, 100)], Config())
    assert isinstance(action, StopA)


def test_heuristic_look_lies_inside_the_anchor_region():
    anchor = (10, 20, 110, 220)
    action = policy_heuristic.choose(_phi(D=[1], K=0, U=9.0), [anchor], Config())
    assert isinstance(action, LookROIA)
    x1, y1, x2, y2 = action.region
    assert anchor[0] <= x1 < x2 <= anchor[2]
    assert anchor[1] <= y1 < y2 <= anchor[3]

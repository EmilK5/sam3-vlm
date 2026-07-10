"""
Tests for agent/actions.py (v2 action space: QueryA / LookROIA / StopA).

QueryA calls pipeline.execute_pass, which pulls in torch/inference; we stub
`pipeline` in sys.modules with a fake execute_pass so dispatch is testable on
CPU. LookROIA-specific behavior (guards, grounding) lives in test_look_roi.py
and test_actions_v2.py.
"""

import sys
import types

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.actions import QueryA, StopA, ActionContext, execute


# ----------------------- fake pipeline.execute_pass -----------------------

@pytest.fixture
def fake_pipeline():
    """Install a fake `pipeline` module whose execute_pass records its call and
    adds `n_new` fruit nodes to the graph."""
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
        processor=object(), image_pil=Image.new("RGB", (100, 100)),
        graph=OrchardGraph(), cfg=Config(), oracle=None, query_set=None,
        discovery=DiscoveryCurve(), partition=[(0, 0, 100, 100)],
    )
    base.update(overrides)
    return ActionContext(**base)


# ----------------------- QueryA dispatch -----------------------

def test_query_action_restricts_to_region_and_returns_int(fake_pipeline):
    ctx = _ctx()
    n = execute(QueryA(region=(0, 0, 50, 50), prompt="green fruit", conf=0.3), ctx)
    assert isinstance(n, int) and n == 3
    assert fake_pipeline.calls[-1]["roi_override"] == (0, 0, 50, 50)
    assert fake_pipeline.calls[-1]["tiling"] is False
    assert len(ctx.graph.nodes) == 3


def test_pass_number_advances_only_with_sensing_passes(fake_pipeline):
    ctx = _ctx()
    execute(QueryA(region=(0, 0, 50, 50), prompt="p", conf=0.3), ctx)
    assert fake_pipeline.calls[-1]["pass_number"] == 1   # first sensing pass
    # Non-sensing actions must NOT advance the pass number (they used to, via
    # the discovery-curve length, skewing pass-dependent pipeline behavior).
    execute(StopA(estimate_name="N_obs"), ctx)           # non-sensing
    execute(QueryA(region=(0, 0, 60, 60), prompt="p", conf=0.3), ctx)
    assert fake_pipeline.calls[-1]["pass_number"] == 2


# ----------------------- StopA + unknown -----------------------

def test_stop_returns_zero():
    assert execute(StopA(estimate_name="N_obs"), _ctx()) == 0


def test_unknown_action_type_raises():
    with pytest.raises(TypeError):
        execute(object(), _ctx())

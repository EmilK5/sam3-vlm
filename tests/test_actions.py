"""
Tests for agent/actions.py.

QueryA/TileQueryA call pipeline.execute_pass, which pulls in torch/inference; we
stub `pipeline` in sys.modules with a fake execute_pass so dispatch is testable on
CPU. Subdivide/Verify need no models (Verify uses the torch-free MockOracle path).
"""

import sys
import types

import numpy as np
import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent import actions
from agent.actions import (
    QueryA, TileQueryA, SubdivideA, VerifyA, StopA, ActionContext,
    execute, subdivide_region,
)


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


# ----------------------- subdivide_region (pure) -----------------------

def test_subdivide_region_quadrants():
    quads = subdivide_region((0, 0, 100, 100))
    assert quads == [(0, 0, 50, 50), (50, 0, 100, 50), (0, 50, 50, 100), (50, 50, 100, 100)]


# ----------------------- QueryA / TileQueryA dispatch -----------------------

def test_query_action_restricts_to_region_and_returns_int(fake_pipeline):
    ctx = _ctx()
    n = execute(QueryA(region=(0, 0, 50, 50), prompt="green fruit", conf=0.3), ctx)
    assert isinstance(n, int) and n == 3
    assert fake_pipeline.calls[-1]["roi_override"] == (0, 0, 50, 50)
    assert fake_pipeline.calls[-1]["tiling"] is False
    assert len(ctx.graph.nodes) == 3


def test_tilequery_action_is_global_tiled(fake_pipeline):
    ctx = _ctx()
    n = execute(TileQueryA(prompt="green fruit", conf=0.3), ctx)
    assert n == 3
    assert fake_pipeline.calls[-1]["roi_override"] is None
    assert fake_pipeline.calls[-1]["tiling"] is True


def test_pass_number_advances_only_with_sensing_passes(fake_pipeline):
    ctx = _ctx()
    execute(QueryA(region=(0, 0, 50, 50), prompt="p", conf=0.3), ctx)
    assert fake_pipeline.calls[-1]["pass_number"] == 1   # first sensing pass
    # Non-sensing actions must NOT advance the pass number (they used to, via
    # the discovery-curve length, skewing pass-dependent pipeline behavior).
    execute(SubdivideA(region=(0, 0, 100, 100)), ctx)
    ctx.discovery.append(0)                              # runner records every action
    execute(QueryA(region=(0, 0, 25, 25), prompt="p", conf=0.3), ctx)
    assert fake_pipeline.calls[-1]["pass_number"] == 2


# ----------------------- SubdivideA (no model) -----------------------

def test_subdivide_updates_partition_without_calling_models():
    # processor that explodes if touched, and no fake pipeline installed
    exploding = types.SimpleNamespace()
    ctx = _ctx(processor=exploding)
    n = execute(SubdivideA(region=(0, 0, 100, 100)), ctx)
    assert n == 0
    assert ctx.partition == [(0, 0, 50, 50), (50, 0, 100, 50), (0, 50, 50, 100), (50, 50, 100, 100)]


def test_subdivide_unknown_region_raises():
    ctx = _ctx(partition=[(0, 0, 100, 100)])
    with pytest.raises(ValueError):
        execute(SubdivideA(region=(10, 10, 20, 20)), ctx)


# ----------------------- VerifyA (real FM+V-IP via MockOracle) -----------------------

def test_verify_action_classifies_named_nodes():
    import os
    from verifier.queries import load_query_set
    from verifier.oracle import MockOracle
    qs = load_query_set(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queries", "green_citrus.json"))

    graph = OrchardGraph()
    nid = graph.add_candidate([10, 10, 50, 50], 0.9, 1)  # unresolved
    ctx = _ctx(graph=graph, oracle=MockOracle("target"), query_set=qs)

    n = execute(VerifyA(node_ids=[nid]), ctx)
    assert n == 0
    assert graph.nodes[nid].classification == "fruit"
    assert graph.nodes[nid].vip_chain is not None


# ----------------------- StopA + unknown -----------------------

def test_stop_returns_zero():
    assert execute(StopA(estimate_name="N_obs"), _ctx()) == 0


def test_unknown_action_type_raises():
    with pytest.raises(TypeError):
        execute(object(), _ctx())

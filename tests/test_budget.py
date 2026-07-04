"""
Tests for agent/budget.py (CostMeter) and its wiring into agent.actions.execute.

Query/TileQuery are exercised through a stubbed `pipeline.execute_pass`; the stub
returns an int-like carrying n_tiles so tile cost can be metered on CPU.
"""

import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.budget import CostMeter
from agent.actions import (
    QueryA, TileQueryA, SubdivideA, VerifyA, StopA, ActionContext, execute,
)


# ----------------------- CostMeter.total -----------------------

def test_cost_meter_total_uses_cfg_ratios():
    cfg = Config()  # c_sam=1, c_tile=.25, c_verify=.5, c_inspect=.5, c_orch=.05
    m = CostMeter(n_sam=2, n_tile=4, n_verify=3, n_inspect=1, n_orch=6)
    expected = 2 + 0.25 * 4 + 0.5 * 3 + 0.5 * 1 + 0.05 * 6
    assert m.total(cfg) == pytest.approx(expected)  # 2 + 1 + 1.5 + 0.5 + 0.3 = 5.3


def test_cost_meter_starts_at_zero():
    m = CostMeter()
    assert m.as_dict() == {"n_sam": 0, "n_tile": 0, "n_verify": 0, "n_inspect": 0, "n_orch": 0}


# ----------------------- stubbed pipeline -----------------------

class _Stats(int):
    def __new__(cls, val, n_tiles=0):
        obj = super().__new__(cls, val)
        obj.n_tiles = n_tiles
        return obj


@pytest.fixture
def fake_pipeline():
    saved = sys.modules.get("pipeline")
    mod = types.ModuleType("pipeline")

    def execute_pass(**kw):
        return _Stats(0, n_tiles=execute_pass.n_tiles)

    execute_pass.n_tiles = 4
    mod.execute_pass = execute_pass
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
        discovery=DiscoveryCurve(), partition=[(0, 0, 100, 100)], cost=CostMeter(),
    )
    base.update(overrides)
    return ActionContext(**base)


# ----------------------- per-action metering -----------------------

def test_query_first_pass_counts_query_plus_leaf_map(fake_pipeline):
    ctx = _ctx()
    execute(QueryA(region=(0, 0, 50, 50), prompt="p", conf=0.3), ctx)  # pass 1
    assert ctx.cost.n_sam == 2   # query + leaf map
    assert ctx.cost.n_tile == 0
    assert ctx.cost.n_orch == 1


def test_query_later_pass_counts_one_sam(fake_pipeline):
    ctx = _ctx()
    ctx.discovery.append(3)      # one prior pass -> next is pass 2
    execute(QueryA(region=(0, 0, 50, 50), prompt="p", conf=0.3), ctx)
    assert ctx.cost.n_sam == 1
    assert ctx.cost.n_orch == 1


def test_tilequery_counts_tiles(fake_pipeline):
    ctx = _ctx()
    ctx.discovery.append(3)      # pass 2 so no leaf-map term
    execute(TileQueryA(prompt="p", conf=0.3), ctx)
    assert ctx.cost.n_tile == 4  # from the stub's n_tiles
    assert ctx.cost.n_sam == 0
    assert ctx.cost.n_orch == 1


def test_verify_counts_oracle_calls_batched():
    from verifier.queries import load_query_set
    from verifier.oracle import MockOracle
    qs = load_query_set(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queries", "green_citrus.json"))
    graph = OrchardGraph()
    ids = [graph.add_candidate([10, 10, 50, 50], 0.9, 1),
           graph.add_candidate([60, 60, 90, 90], 0.9, 1)]
    ctx = _ctx(graph=graph, oracle=MockOracle("target"), query_set=qs)  # batched default

    execute(VerifyA(node_ids=ids), ctx)
    assert ctx.cost.n_verify == 2   # one oracle call per node (batched)
    assert ctx.cost.n_orch == 1


def test_subdivide_and_stop_only_count_orchestration():
    ctx = _ctx()
    execute(SubdivideA(region=(0, 0, 100, 100)), ctx)
    execute(StopA(estimate_name="N_obs"), ctx)
    assert ctx.cost.n_orch == 2
    assert (ctx.cost.n_sam, ctx.cost.n_tile, ctx.cost.n_verify) == (0, 0, 0)


# ----------------------- scripted sequence, hand-computed total -----------------------

def test_scripted_sequence_matches_hand_computed_total(fake_pipeline):
    cfg = Config()
    ctx = _ctx(cfg=cfg)

    execute(QueryA(region=(0, 0, 50, 50), prompt="p", conf=0.3), ctx)   # pass1: n_sam += 2, orch=1
    ctx.discovery.append(3)
    execute(TileQueryA(prompt="p", conf=0.3), ctx)                      # pass2: n_tile += 4, orch=2
    ctx.discovery.append(2)
    execute(SubdivideA(region=(0, 0, 100, 100)), ctx)                   # orch=3
    execute(StopA(estimate_name="N_obs"), ctx)                          # orch=4

    assert (ctx.cost.n_sam, ctx.cost.n_tile, ctx.cost.n_verify, ctx.cost.n_orch) == (2, 4, 0, 4)
    # total = 2 + 0.25*4 + 0.5*0 + 0.05*4 = 2 + 1 + 0.2 = 3.2
    assert ctx.cost.total(cfg) == pytest.approx(3.2)

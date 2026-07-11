"""
Step 9.4 spec tests: the active arm never emits LookROIA, and recovers recall via
tiling. Covers: QueryA.tiling threads to execute_pass; bootstrap_tiled_pass runs a
tiled seed pass (the generic cascade's recall floor) as a 4th bootstrap record; a v3
refine builds a TILED QueryA (cfg.refine_tiling), overridable off; and an invalid VLM
response STOPS the arm -- never a LookROIA. CPU-only: `pipeline` is stubbed in
sys.modules / an execute_fn is injected; no SAM3, no network.
"""

import dataclasses
import sys
import types

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.actions import QueryA, LookROIA, StopA, ActionContext, execute
from agent.belief import DiscoveryCurve, summarize
from agent.budget import CostMeter
from agent.history import EpisodeHistory
from agent import runner, policy_vlm_v3


FRAME = Image.new("RGB", (120, 120))
TREE_ROI = [10, 10, 110, 110]


# ----------------------- QueryA.tiling threads to execute_pass -----------------------

@pytest.fixture
def fake_pipeline():
    saved = sys.modules.get("pipeline")
    mod = types.ModuleType("pipeline")
    mod.calls = []

    def execute_pass(**kw):
        mod.calls.append(kw)
        return 0

    mod.execute_pass = execute_pass
    sys.modules["pipeline"] = mod
    try:
        yield mod
    finally:
        if saved is None:
            sys.modules.pop("pipeline", None)
        else:
            sys.modules["pipeline"] = saved


def _ctx():
    return ActionContext(processor=object(), image_pil=FRAME, graph=OrchardGraph(),
                         cfg=Config(), discovery=DiscoveryCurve(), cost=CostMeter(),
                         partition=[tuple(TREE_ROI)])


def test_query_tiling_flag_threads_to_execute_pass(fake_pipeline):
    ctx = _ctx()
    execute(QueryA(region=tuple(TREE_ROI), prompt="green fruit", conf=0.4, tiling=True), ctx)
    assert fake_pipeline.calls[-1]["tiling"] is True
    execute(QueryA(region=tuple(TREE_ROI), prompt="green fruit", conf=0.4), ctx)
    assert fake_pipeline.calls[-1]["tiling"] is False        # default unchanged


# ----------------------- bootstrap tiled seed floor -----------------------

def test_bootstrap_tiled_pass_adds_tiled_seed_record():
    seen = []

    def fake_execute(action, ctx):
        if isinstance(action, StopA):
            ctx.last_pass_stats = None
            return 0
        seen.append((type(action).__name__, getattr(action, "tiling", None)))
        ctx.last_pass_stats = {"n_new": 0, "n_redetected": 0, "n_detections": 0}
        return 0

    ctx = ActionContext(image_pil=FRAME, graph=OrchardGraph(), cfg=Config(),
                        discovery=DiscoveryCurve(), partition=[tuple(TREE_ROI)], cost=CostMeter())
    # budget = 2 so the two bootstrap passes (global + tiled seed) fill it and the
    # policy loop does not run (keeps the record list to the bootstrap sequence).
    result = runner.run_episode(FRAME, ctx, policy=lambda p, part, c: StopA("N_obs"),
                                max_actions=2, execute_fn=fake_execute,
                                bootstrap_global_pass=True, bootstrap_tiled_pass=True)
    actions = [r["x"]["action"] for r in result["history"]]
    assert actions == ["canopy_roi", "leaf_map", "global_pass", "tiled_seed_pass"]
    # the global bootstrap ran untiled; the seed-floor pass ran tiled.
    assert seen == [("QueryA", False), ("QueryA", True)]


# ----------------------- refine builds a tiled QueryA -----------------------

class _Client:
    def __init__(self, content):
        self._c = content
        self.chat = self
        self.completions = self

    def create(self, **kw):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._c))])


def _graph_phi():
    cfg = Config()
    g = OrchardGraph()
    g.tree_roi = list(TREE_ROI)
    nid = g.add_candidate([20, 20, 40, 40], 0.9, 1)
    g.nodes[nid].classification = "fruit"
    return cfg, g, summarize(g, DiscoveryCurve(), budget=12, cfg=cfg)


def _hist():
    h = EpisodeHistory()
    h.append({"action": "global_pass", "region": TREE_ROI, "prompt": "green fruit", "conf": 0.65},
             {"n_new": 1, "n_redetected": 0, "n_detections": 1, "new_nodes": [],
              "totals": {"K": 1, "N_obs": 1}})
    return h


def test_refine_builds_tiled_query():
    cfg, g, phi = _graph_phi()
    client = _Client('{"action": "refine", "args": {"prompt": "round fruit", "threshold": 0.4}}')
    action = policy_vlm_v3.choose(phi, _hist(), g, cfg, client=client, image=FRAME)
    assert isinstance(action, QueryA) and action.tiling is True    # cfg.refine_tiling default


def test_refine_tiling_config_off():
    cfg, g, phi = _graph_phi()
    cfg = dataclasses.replace(cfg, refine_tiling=False)
    client = _Client('{"action": "refine", "args": {"prompt": "round fruit", "threshold": 0.4}}')
    action = policy_vlm_v3.choose(phi, _hist(), g, cfg, client=client, image=FRAME)
    assert isinstance(action, QueryA) and action.tiling is False


# ----------------------- LookROIA is banned: invalid -> Stop -----------------------

def test_invalid_response_stops_never_lookroia():
    cfg, g, phi = _graph_phi()
    client = _Client('not json at all')
    action = policy_vlm_v3.choose(phi, _hist(), g, cfg, client=client, image=FRAME)
    assert isinstance(action, StopA)
    assert not isinstance(action, LookROIA)

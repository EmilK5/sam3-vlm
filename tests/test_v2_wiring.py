"""
Step 8.5 wiring tests: config cleanup + new fields, RouterOracle wiring in
run_eval.build_verifier, and an offline end-to-end v2 episode.

The end-to-end run uses the REAL agent.actions.execute against a stubbed
`pipeline` module (no torch) and a fake VLM client, so a full episode --
bootstrap trio + look + stop -- is exercised on CPU with no network.
"""

import os
import sys
import types
import json

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.budget import CostMeter
from agent.actions import ActionContext
from agent import runner
from eval import run_eval

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


# ----------------------- config: removed + added fields -----------------------

def test_orphaned_knobs_removed():
    cfg = Config()
    assert not hasattr(cfg, "small_area")   # old tile-menu heuristic only
    assert not hasattr(cfg, "c0")


def test_new_config_fields():
    cfg = Config()
    assert cfg.sam3_presence_tau == 0.5     # promoted from the 8.4 getattr default
    assert cfg.oracle_kind == "qwen"        # default oracle; "router" opt-in


# ----------------------- build_verifier oracle wiring -----------------------

def test_build_verifier_default_builds_qwen_oracle():
    from verifier.oracle import QwenOracle
    _cfg, oracle, _qs = run_eval.build_verifier("vip", query_file=GREEN_CITRUS)
    assert isinstance(oracle, QwenOracle)


def test_build_verifier_router_wraps_qwen_and_takes_processor():
    from verifier.oracle import RouterOracle, QwenOracle
    stub_processor = object()
    _cfg, oracle, _qs = run_eval.build_verifier(
        "vip", query_file=GREEN_CITRUS, oracle_kind="router", processor=stub_processor)
    assert isinstance(oracle, RouterOracle)
    assert isinstance(oracle.vlm_oracle, QwenOracle)
    assert oracle.processor is stub_processor
    # the promoted config threshold reaches the SAM3 channel.
    assert oracle._sam3.tau == 0.5


def test_build_verifier_ioc_has_no_oracle_regardless_of_kind():
    _cfg, oracle, qs = run_eval.build_verifier("ioc", oracle_kind="router")
    assert oracle is None and qs is None


# ----------------------- offline end-to-end v2 episode -----------------------

class _Stats(int):
    """Mirrors the PassStats fields agent.actions._meter_query reads."""
    def __new__(cls, val, n_sam_calls=0, n_tiles=0, n_verify_calls=0):
        obj = super().__new__(cls, val)
        obj.n_sam_calls = n_sam_calls
        obj.n_tiles = n_tiles
        obj.n_verify_calls = n_verify_calls
        return obj


@pytest.fixture
def fake_pipeline():
    """`pipeline.execute_pass` that adds 2 fruit nodes per sensing pass and meters
    one SAM3 call, so cost accrues and counts are non-trivial."""
    saved = sys.modules.get("pipeline")
    mod = types.ModuleType("pipeline")

    def execute_pass(**kw):
        for j in range(2):
            nid = kw["graph"].add_candidate([j, 0, j + 5, 5], 0.9, kw["pass_number"])
            kw["graph"].nodes[nid].classification = "fruit"
        return _Stats(2, n_sam_calls=1)

    mod.execute_pass = execute_pass
    sys.modules["pipeline"] = mod
    try:
        yield mod
    finally:
        if saved is None:
            sys.modules.pop("pipeline", None)
        else:
            sys.modules["pipeline"] = saved


class _FakeVLM:
    def __init__(self, contents):
        self._it = iter(contents)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = next(self._it)
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_offline_v2_episode_end_to_end(fake_pipeline):
    cfg = Config()
    img = Image.new("RGB", (100, 100))
    graph = OrchardGraph()
    graph.tree_roi = [0, 0, 100, 100]
    ctx = ActionContext(processor=object(), image_pil=img, graph=graph, cfg=cfg,
                        oracle=None, query_set=None, discovery=DiscoveryCurve(),
                        partition=[(0, 0, 100, 100)], cost=CostMeter())

    fake = _FakeVLM([
        '{"action": "look", "args": {"region": [10, 10, 90, 90]}}',
        '{"action": "stop", "args": {"estimate_name": "N_obs"}}',
    ])
    pol = runner.make_vlm_policy(ctx, vlm_client=fake)
    result = runner.run_episode(img, ctx, pol, max_actions=6,
                                bootstrap_global_pass=True, auto_stop=True)

    actions = [r["x"]["action"] for r in result["history"]]
    assert actions[:3] == ["canopy_roi", "leaf_map", "global_pass"]   # bootstrap trio first
    assert "LookROIA" in actions                                       # the VLM look sensed
    assert actions[-1] == "StopA"                                      # ended on the model's stop

    # counts sane: bootstrap (2) + one look (2) = 4 fruit observed.
    assert result["counts"]["N_obs"] == 4
    # cost metered (two sensing passes each metered a SAM3 call + orchestration).
    assert result["cost"] > 0.0
    # history is the full x/y record, 1:1 with the log, and JSON-serializable.
    assert len(result["history"]) == len(result["log"])
    json.dumps(result["history"])


def test_offline_v2_episode_auto_stops_on_saturation(fake_pipeline):
    # A VLM that always looks: the runner must auto-stop on discovery saturation
    # even though the policy never emits stop.
    import dataclasses
    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0)
    img = Image.new("RGB", (100, 100))
    graph = OrchardGraph()
    graph.tree_roi = [0, 0, 100, 100]
    ctx = ActionContext(processor=object(), image_pil=img, graph=graph, cfg=cfg,
                        oracle=None, query_set=None, discovery=DiscoveryCurve(),
                        partition=[(0, 0, 100, 100)], cost=CostMeter())

    # The stub adds nodes only on the bootstrap; make later looks discover nothing
    # by exhausting the dedup guard is hard here, so instead drive saturation with
    # an executor that stops discovering after the bootstrap.
    calls = {"n": 0}
    orig = fake_pipeline.execute_pass

    def diminishing(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return orig(**kw)          # bootstrap discovers 2
        return _Stats(0, n_sam_calls=1)  # later looks discover nothing -> saturates

    fake_pipeline.execute_pass = diminishing

    fake = _FakeVLM(['{"action": "look", "args": {"region": [%d, 10, %d, 90]}}' % (i, i + 60)
                     for i in range(1, 20)])
    pol = runner.make_vlm_policy(ctx, vlm_client=fake)
    result = runner.run_episode(img, ctx, pol, max_actions=20,
                                bootstrap_global_pass=True, auto_stop=True)

    actions = [r["x"]["action"] for r in result["history"]]
    assert "StopA" not in actions          # policy never stopped
    assert len(result["history"]) < 20 + 2  # auto-stopped before the budget

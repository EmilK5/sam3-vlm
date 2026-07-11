"""
Step 9.3 spec tests: per-prompt dedup feedback threads detection -> observation ->
policy. Two links:
  (1) actions._execute_query stashes {n_new, n_redetected, n_detections} on the ctx
      from the PassStats it gets back (n_redetected == duplicates_rejected).
  (2) runner.run_episode copies that onto every observation record y, so the v3
      policy sees, per prompt, how many already-known objects it re-found -- the
      signal that lets it judge a wording, not just enumerate past ones.
CPU-only: `pipeline` is stubbed in sys.modules (so no torch/inference import) and
the runner's execute_fn is injected; no SAM3, no network.
"""

import sys
import types

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.actions import QueryA, StopA, ActionContext, execute
from agent.belief import DiscoveryCurve
from agent.budget import CostMeter
from agent import runner


class _Stats(int):
    """Int-subclass stand-in for pipeline.PassStats (int value == accepted new tracks;
    attributes carry the per-pass diagnostics _execute_query reads)."""
    def __new__(cls, value, duplicates_rejected=0, post_nms=0, n_sam_calls=0,
                n_tiles=0, n_verify_calls=0):
        o = super().__new__(cls, value)
        o.duplicates_rejected = duplicates_rejected
        o.post_nms = post_nms
        o.n_sam_calls = n_sam_calls
        o.n_tiles = n_tiles
        o.n_verify_calls = n_verify_calls
        return o


@pytest.fixture
def fake_pipeline():
    """Install a fake `pipeline` module whose execute_pass returns a _Stats with a
    dedup re-detection count, so actions._execute_query runs on CPU without torch."""
    saved = sys.modules.get("pipeline")
    mod = types.ModuleType("pipeline")

    def execute_pass(**kw):
        # 4 distinct detections; 3 matched already-known tracks (dedup), 1 new.
        nid = kw["graph"].add_candidate([0, 0, 5, 5], 0.9, kw["pass_number"])
        kw["graph"].nodes[nid].classification = "fruit"
        return _Stats(1, duplicates_rejected=3, post_nms=4, n_sam_calls=1)

    mod.execute_pass = execute_pass
    sys.modules["pipeline"] = mod
    try:
        yield mod
    finally:
        if saved is None:
            sys.modules.pop("pipeline", None)
        else:
            sys.modules["pipeline"] = saved


def test_execute_query_stashes_pass_feedback(fake_pipeline):
    ctx = ActionContext(processor=object(), image_pil=Image.new("RGB", (50, 50)),
                        graph=OrchardGraph(), cfg=Config(), cost=CostMeter())
    n = execute(QueryA(region=(0, 0, 50, 50), prompt="green fruit", conf=0.4), ctx)
    assert n == 1
    assert ctx.last_pass_stats == {"n_new": 1, "n_redetected": 3, "n_detections": 4}


def test_stop_action_clears_pass_feedback():
    ctx = ActionContext(graph=OrchardGraph(), cfg=Config(), cost=CostMeter())
    ctx.last_pass_stats = {"n_new": 9, "n_redetected": 9, "n_detections": 9}   # stale
    execute(StopA(estimate_name="N_obs"), ctx)
    assert ctx.last_pass_stats is None                     # a non-sensing action leaves no feedback


def test_run_episode_observation_carries_dedup_feedback():
    def fake_execute(action, ctx):
        if isinstance(action, StopA):
            ctx.last_pass_stats = None
            return 0
        nid = ctx.graph.add_candidate([10, 10, 20, 20], 0.9, 1)
        ctx.graph.nodes[nid].classification = "fruit"
        ctx.last_pass_stats = {"n_new": 1, "n_redetected": 3, "n_detections": 4}
        return 1

    cfg = Config()
    ctx = ActionContext(image_pil=Image.new("RGB", (50, 50)), graph=OrchardGraph(),
                        cfg=cfg, discovery=DiscoveryCurve(), partition=[(0, 0, 50, 50)],
                        cost=CostMeter())
    result = runner.run_episode(
        Image.new("RGB", (50, 50)), ctx, policy=lambda phi, part, c: StopA("N_obs"),
        max_actions=1, execute_fn=fake_execute, bootstrap_global_pass=True, auto_stop=False,
    )
    hist = result["history"]
    g = hist[2]["y"]                                       # the global_pass observation
    assert (g["n_new"], g["n_redetected"], g["n_detections"]) == (1, 3, 4)
    # the synthetic canopy_roi / leaf_map records carry zeroed feedback (no pass ran).
    assert hist[0]["y"]["n_redetected"] == 0 and hist[0]["y"]["n_detections"] == 0

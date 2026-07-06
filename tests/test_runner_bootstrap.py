"""
Tests for the run_episode bootstrap global pass + auto-stop backstop (step 7.2).

Both new behaviors are opt-in (default off), so the existing episode tests in
test_policy_heuristic.py / test_eval_sweep.py exercise the unchanged path. Here we
drive run_episode with scripted policies and a stub executor (no models, no torch).
"""

import dataclasses

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.actions import QueryA, TileQueryA, StopA, ActionContext
from agent import runner


def _ctx(partition, cfg=None):
    return ActionContext(
        cfg=cfg or Config(), graph=OrchardGraph(), discovery=DiscoveryCurve(),
        partition=partition, image_pil=None,
    )


# ----------------------- bootstrap global pass -----------------------

def test_bootstrap_runs_global_query_first():
    ctx = _ctx(partition=[(0, 0, 120, 90)])
    seen = []

    def policy(phi, partition, cfg):
        return StopA(estimate_name="N_obs")   # stop right after the bootstrap

    def stub(action, ctx_):
        seen.append(action)
        if isinstance(action, QueryA):
            ctx_.graph.add_candidate([1, 1, 6, 6], 0.9, 1)
            return 1
        return 0

    result = runner.run_episode(image=None, ctx=ctx, policy=policy, max_actions=5,
                                execute_fn=stub, bootstrap_global_pass=True)

    assert isinstance(seen[0], QueryA)                 # first executed action
    assert seen[0].region == (0, 0, 120, 90)           # over partition[0] (full frame)
    assert result["log"][0]["action"] == "QueryA"      # logged as its own step
    assert ctx.discovery.as_list()[0] == 1             # seeded the discovery curve
    assert len(ctx.graph.nodes) == 1


def test_no_bootstrap_by_default():
    ctx = _ctx(partition=[(0, 0, 100, 100)])
    seen = []

    def policy(phi, partition, cfg):
        return StopA(estimate_name="N_obs")

    def stub(action, ctx_):
        seen.append(action)
        return 0

    runner.run_episode(image=None, ctx=ctx, policy=policy, max_actions=5, execute_fn=stub)
    assert isinstance(seen[0], StopA)                  # no injected global query


# ----------------------- auto-stop backstop -----------------------

def _saturating_stub():
    """Executor whose sensing yields 5 then 0 forever -> discovery saturates."""
    counts = iter([5])

    def stub(action, ctx_):
        if isinstance(action, (QueryA, TileQueryA)):
            n = next(counts, 0)
            for j in range(n):
                ctx_.graph.add_candidate([j, 0, j + 5, 5], 0.9, 1)
            return n
        return 0

    return stub


def _never_stop_policy(phi, partition, cfg):
    return TileQueryA(prompt="green fruit", conf=0.3)


def test_auto_stop_on_saturation_without_policy_stop():
    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0)  # window_m=3
    ctx = _ctx(partition=[(0, 0, 100, 100)], cfg=cfg)

    result = runner.run_episode(
        image=None, ctx=ctx, policy=_never_stop_policy, max_actions=20,
        execute_fn=_saturating_stub(), bootstrap_global_pass=True, auto_stop=True,
    )

    actions = [e["action"] for e in result["log"]]
    assert "StopA" not in actions            # the policy never stopped
    assert len(result["log"]) < 20           # auto-stopped well before budget
    assert len(ctx.graph.nodes) == 5         # sensed on the bootstrap pass


def test_auto_stop_off_runs_to_budget():
    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0)
    ctx = _ctx(partition=[(0, 0, 100, 100)], cfg=cfg)

    result = runner.run_episode(
        image=None, ctx=ctx, policy=_never_stop_policy, max_actions=8,
        execute_fn=_saturating_stub(), bootstrap_global_pass=True,  # auto_stop off
    )

    assert "StopA" not in [e["action"] for e in result["log"]]
    assert len(result["log"]) == 8           # ran the full budget without auto-stop

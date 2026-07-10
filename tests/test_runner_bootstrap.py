"""
Tests for the run_episode bootstrap pass + auto-stop backstop (step 7.2; the
bootstrap decomposes into a 3-record canopy_roi/leaf_map/global_pass trio in
step 8.2).

Both new behaviors are opt-in (default off), so the existing episode tests in
test_policy_heuristic.py / test_eval_sweep.py exercise the unchanged path. Here we
drive run_episode with scripted policies and a stub executor (no models, no torch).
"""

import dataclasses

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.actions import QueryA, LookROIA, StopA, ActionContext
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
    # The one physical bootstrap pass is logged as three records in order.
    assert [e["action"] for e in result["log"][:3]] == ["canopy_roi", "leaf_map", "global_pass"]
    assert result["log"][2]["n_new"] == 1              # global_pass carries the observation
    assert ctx.discovery.as_list() == [1]              # billed as ONE sensing pass
    assert len(ctx.graph.nodes) == 1
    # history and log stay one-to-one and the history is JSON-serializable.
    assert len(result["history"]) == len(result["log"])
    import json
    json.dumps(result["history"])
    # canopy_roi documents the tree ROI box; global_pass carries the new node.
    assert result["history"][0]["x"]["roi"] == [0, 0, 120, 90]
    assert len(result["history"][2]["y"]["new_nodes"]) == 1


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
        if isinstance(action, (QueryA, LookROIA)):
            n = next(counts, 0)
            for j in range(n):
                ctx_.graph.add_candidate([j, 0, j + 5, 5], 0.9, 1)
            return n
        return 0

    return stub


def _never_stop_policy(phi, partition, cfg):
    return LookROIA(region=(0, 0, 100, 100))


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
    assert len(result["history"]) == len(result["log"])


def test_auto_stop_off_runs_to_budget():
    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0)
    ctx = _ctx(partition=[(0, 0, 100, 100)], cfg=cfg)

    result = runner.run_episode(
        image=None, ctx=ctx, policy=_never_stop_policy, max_actions=8,
        execute_fn=_saturating_stub(), bootstrap_global_pass=True,  # auto_stop off
    )

    assert "StopA" not in [e["action"] for e in result["log"]]
    # Budget is 8 SENSING actions: the bootstrap (1, shown as the global_pass
    # record) + 7 policy looks. The two extra bootstrap documentation records
    # (canopy_roi, leaf_map) are not billed, so the log has 8 + 2 = 10 entries.
    sensing = [e for e in result["log"] if e["action"] in ("global_pass", "LookROIA")]
    assert len(sensing) == 8
    assert len(result["log"]) == 10
    assert len(result["history"]) == len(result["log"])

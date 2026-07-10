"""
Tests for agent/history.py + the episode-history wiring in run_episode (step 8.2).

The history is the full past (x_1^t, y_1^t) the v2 policy will be shown. Here we
drive run_episode with a scripted policy + a stub executor (no models, no torch)
and check: the bootstrap emits exactly the three leading records
canopy_roi/leaf_map/global_pass; a look appends a record whose new_nodes match
the nodes the stub added; and every record is json.dumps-able.
"""

import json

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve
from agent.actions import LookROIA, StopA, ActionContext
from agent.history import EpisodeHistory, action_params, node_summary
from agent import runner


# ----------------------- EpisodeHistory unit -----------------------

def test_history_stamps_t_and_is_json_ready():
    h = EpisodeHistory()
    h.append({"action": "canopy_roi", "roi": [0, 0, 10, 10]},
             {"n_new": 0, "new_nodes": [], "totals": {"K": 0, "N_obs": 0}})
    h.append({"action": "LookROIA", "region": [0, 0, 5, 5]},
             {"n_new": 1, "new_nodes": [], "totals": {"K": 1, "N_obs": 1}})
    assert [r["t"] for r in h.as_list()] == [1, 2]
    assert len(h) == 2
    json.dumps(h.as_list())


def test_action_params_serializes_region_to_list():
    p = action_params(LookROIA(region=(1, 2, 3, 4)))
    assert p == {"action": "LookROIA", "region": [1, 2, 3, 4]}
    assert action_params(StopA(estimate_name="N_obs")) == {
        "action": "StopA", "estimate_name": "N_obs"}


def test_node_summary_shape():
    g = OrchardGraph()
    nid = g.add_candidate([1, 2, 3, 4], 0.7, 1)
    g.nodes[nid].classification = "fruit"
    s = node_summary(g.nodes[nid])
    assert s == {"id": nid, "box": [1.0, 2.0, 3.0, 4.0], "conf": 0.7, "class": "fruit"}


# ----------------------- episode wiring -----------------------

def _ctx(partition, cfg=None):
    return ActionContext(cfg=cfg or Config(), graph=OrchardGraph(),
                         discovery=DiscoveryCurve(), partition=partition, image_pil=None)


def _scripted_stub(script):
    """Executor that adds `script[i]` fruit nodes on the i-th sensing action."""
    calls = {"i": 0}

    def stub(action, ctx):
        if isinstance(action, StopA):
            return 0
        n = script[calls["i"]] if calls["i"] < len(script) else 0
        calls["i"] += 1
        for j in range(n):
            nid = ctx.graph.add_candidate([j, 0, j + 5, 5], 0.9, 1)
            ctx.graph.nodes[nid].classification = "fruit"
        return n

    return stub


def test_bootstrap_emits_exactly_three_leading_records():
    ctx = _ctx(partition=[(0, 0, 100, 80)])

    def policy(phi, partition, cfg):
        return StopA(estimate_name="N_obs")

    result = runner.run_episode(image=None, ctx=ctx, policy=policy, max_actions=5,
                                execute_fn=_scripted_stub([2]), bootstrap_global_pass=True)

    lead = result["history"][:3]
    assert [r["x"]["action"] for r in lead] == ["canopy_roi", "leaf_map", "global_pass"]
    assert lead[0]["x"]["roi"] == [0, 0, 100, 80]        # tree ROI box (partition[0])
    assert lead[1]["x"]["n_leaves"] == 0                  # stub set no leaf map
    assert lead[2]["y"]["n_new"] == 2                     # global_pass carries the obs
    # canopy_roi / leaf_map are documentation only -- no new nodes on them.
    assert lead[0]["y"]["new_nodes"] == [] and lead[1]["y"]["new_nodes"] == []
    # billed as ONE sensing pass despite three records.
    assert ctx.discovery.as_list() == [2]


def test_look_record_new_nodes_match_stub_graph():
    ctx = _ctx(partition=[(0, 0, 200, 200)])
    script = iter([LookROIA(region=(10, 10, 120, 120)), StopA(estimate_name="N_obs")])

    def policy(phi, partition, cfg):
        return next(script)

    # bootstrap adds 2 nodes; the single look adds 1.
    result = runner.run_episode(image=None, ctx=ctx, policy=policy, max_actions=5,
                                execute_fn=_scripted_stub([2, 1]), bootstrap_global_pass=True)

    look_rec = next(r for r in result["history"] if r["x"]["action"] == "LookROIA")
    assert look_rec["y"]["n_new"] == 1
    assert len(look_rec["y"]["new_nodes"]) == 1
    node = look_rec["y"]["new_nodes"][0]
    assert node["box"] == [0.0, 0.0, 5.0, 5.0]
    assert node["conf"] == 0.9 and node["class"] == "fruit"
    assert node["id"] in ctx.graph.nodes
    # the look's totals reflect all three nodes seen so far (2 bootstrap + 1 look).
    assert look_rec["y"]["totals"]["K"] == 3


def test_history_length_equals_log_length_and_is_json_ready():
    ctx = _ctx(partition=[(0, 0, 100, 100)])
    script = iter([LookROIA(region=(10, 10, 90, 90)), StopA(estimate_name="N_obs")])

    def policy(phi, partition, cfg):
        return next(script)

    result = runner.run_episode(image=None, ctx=ctx, policy=policy, max_actions=5,
                                execute_fn=_scripted_stub([1, 1]), bootstrap_global_pass=True)
    assert len(result["history"]) == len(result["log"])
    json.dumps(result["history"])                        # fully serializable
    assert len(ctx.history) == len(result["history"])    # ctx.history is the same record set


def test_no_bootstrap_means_no_documentation_records():
    ctx = _ctx(partition=[(0, 0, 100, 100)])
    script = iter([LookROIA(region=(10, 10, 90, 90)), StopA(estimate_name="N_obs")])

    def policy(phi, partition, cfg):
        return next(script)

    result = runner.run_episode(image=None, ctx=ctx, policy=policy, max_actions=5,
                                execute_fn=_scripted_stub([1]))  # bootstrap off
    actions = [r["x"]["action"] for r in result["history"]]
    assert "canopy_roi" not in actions and "leaf_map" not in actions
    assert actions[0] == "LookROIA"                       # straight to the policy action

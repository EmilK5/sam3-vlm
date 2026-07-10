"""
Tests for agent/policy_heuristic.choose (v2 look/stop fallback),
agent/runner.run_episode, and the belief.count_estimates estimators.
"""

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, count_estimates
from agent.actions import QueryA, LookROIA, StopA, ActionContext
from agent.policy_heuristic import choose, _grid_cells
from agent import runner


def _phi(**over):
    """A minimal belief summary; override any field."""
    base = dict(
        K=0, n_t=0, D=[], U=0.0, ids=[], centers=[], area=[], classification=[],
        w=[], s=[], k=[], delta=[], tiling_status=False, remaining_budget=12,
    )
    base.update(over)
    return base


# ----------------------- stopping rule -----------------------

def test_choose_stops_when_saturated_and_low_uncertainty():
    cfg = Config()  # delta_disc=1.0, delta_U=0.5, window_m=3
    # A settled node (K>0) so this is a genuine stop, not the vacuous
    # nothing-sensed-yet case guarded below.
    phi = _phi(D=[0, 0, 0], U=0.1, K=1,
               ids=["n1"], centers=[[50, 50]], area=[400.0], classification=["fruit"],
               w=[1.5], s=[0.9], k=[3], delta=[0.0])
    action = choose(phi, partition=[(0, 0, 100, 100)], cfg=cfg)
    assert isinstance(action, StopA)


def test_choose_does_not_stop_before_first_detection():
    # Vacuous-stop guard: an empty graph (K=0) with saturated discovery (D of
    # non-sensing zeros) and U=0 must NOT stop -- nothing has been sensed yet.
    cfg = Config()
    phi = _phi(D=[0, 0, 0], U=0.0, K=0)
    action = choose(phi, [(0, 0, 100, 100)], cfg)
    assert not isinstance(action, StopA)


def test_choose_does_not_stop_when_still_discovering():
    cfg = Config()
    phi = _phi(D=[5, 5, 5], U=0.1)   # recent mean 5 > delta_disc -> not saturated
    action = choose(phi, [(0, 0, 100, 100)], cfg)
    assert not isinstance(action, StopA)


def test_choose_does_not_stop_when_uncertainty_high():
    cfg = Config()
    phi = _phi(D=[0, 0, 0], U=10.0)  # saturated but U > delta_U
    action = choose(phi, [(0, 0, 100, 100)], cfg)
    assert not isinstance(action, StopA)


# ----------------------- action selection (v2: look/stop only) -----------------------

def test_empty_graph_first_action_is_a_look_inside_the_anchor():
    action = choose(_phi(D=[], K=0), [(0, 0, 100, 100)], Config())
    assert isinstance(action, LookROIA)
    assert action.region in _grid_cells((0, 0, 100, 100))


def test_look_targets_the_emptiest_grid_cell():
    # All candidates crowd the top-left cell; with no sensing passes yet (D=[]),
    # the pick is the emptiest-ranked cell, i.e. NOT the top-left one.
    phi = _phi(
        K=3, D=[], U=5.0,
        ids=["a", "b", "c"], centers=[[10, 10], [20, 20], [30, 30]],
        area=[100.0] * 3, classification=["fruit"] * 3,
        w=[1.5] * 3, s=[0.9] * 3, k=[1] * 3, delta=[0.0] * 3,
    )
    action = choose(phi, [(0, 0, 100, 100)], Config())
    assert isinstance(action, LookROIA)
    assert action.region != (0, 0, 50.0, 50.0)          # crowded TL cell not picked


def test_look_cycles_cells_with_sensing_passes():
    # Same belief, growing D: the pick must walk through all four grid cells
    # before repeating, so consecutive fallback calls never re-propose a cell.
    cells = _grid_cells((0, 0, 100, 100))
    picks = [choose(_phi(D=[1] * t, K=0, U=9.0), [(0, 0, 100, 100)], Config()).region
             for t in range(4)]
    assert sorted(picks) == sorted(cells)               # each cell exactly once


def test_choose_returns_only_v2_actions():
    action = choose(_phi(D=[1], K=0), [(0, 0, 100, 100)], Config())
    assert isinstance(action, (LookROIA, StopA))


# ----------------------- count estimators -----------------------

def test_count_estimates():
    cfg = Config()
    g = OrchardGraph()
    a = g.nodes[g.add_candidate([0, 0, 10, 10], 0.9, 1)]
    a.classification = "fruit"
    a.reinforce([0, 0, 10, 10], "s")          # k=2
    b = g.nodes[g.add_candidate([20, 0, 30, 10], 0.1, 1)]
    b.classification = "fruit"                 # k=1, low score
    g.nodes[g.add_candidate([40, 0, 50, 10], 0.9, 1)].classification = "leaf"
    g.nodes[g.add_candidate([60, 0, 70, 10], 0.9, 1)].classification = "spurious"
    g.nodes[g.add_candidate([80, 0, 90, 10], 0.9, 1)].classification = "unresolved"

    est = count_estimates(g, cfg)
    assert est["N_obs"] == 3                    # fruit(a,b) + unresolved(e); leaf/spurious excluded
    assert est["N_cons"] == 2                   # a (k>=2), e (s>=0.5); b fails both
    assert est["N_supp"] == 3                   # all three clear tau_w=0.5


# ----------------------- runner stops with a saturating executor -----------------------

def test_run_episode_stops_by_step_five():
    # δ_U relaxed so the stop hinges on discovery saturation (the executor adds
    # nodes whose instability would otherwise keep U high).
    import dataclasses
    cfg = dataclasses.replace(Config(), delta_U=1e9, delta_disc=1.0)

    ctx = ActionContext(cfg=cfg, graph=OrchardGraph(), discovery=DiscoveryCurve(),
                        partition=[(0, 0, 100, 100)])

    script = [5, 2, 0, 0]

    def stub_execute(action, c):
        if isinstance(action, StopA):
            return 0
        n = script[min(len(c.discovery.counts), len(script) - 1)]
        for j in range(n):
            nid = c.graph.add_candidate([j, 0, j + 5, 5], 0.9, len(c.discovery.counts) + 1)
            c.graph.nodes[nid].classification = "fruit"
        return n

    result = runner.run_episode(image=None, ctx=ctx, policy=choose,
                                max_actions=10, execute_fn=stub_execute)

    assert len(result["log"]) <= 5
    assert result["log"][-1]["action"] == "StopA"
    assert "N_obs" in result["counts"]


def test_run_episode_records_only_sensing_actions_in_discovery():
    # Non-sensing actions (stop) must not enter the discovery curve; otherwise
    # their zeros fake saturation.
    cfg = Config()
    ctx = ActionContext(cfg=cfg, graph=OrchardGraph(), discovery=DiscoveryCurve(),
                        partition=[(0, 0, 100, 100)])
    script = iter([
        QueryA(region=(0, 0, 100, 100), prompt="green fruit", conf=0.3),
        StopA(estimate_name="N_obs"),
    ])

    def policy(phi, partition, c):
        return next(script)

    def stub_execute(action, c):
        if isinstance(action, QueryA):
            c.graph.add_candidate([0, 0, 5, 5], 0.9, 1)
            return 1
        return 0

    runner.run_episode(image=None, ctx=ctx, policy=policy,
                       max_actions=5, execute_fn=stub_execute)
    # Only the single Query contributed; the stop did not.
    assert ctx.discovery.as_list() == [1]

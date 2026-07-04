import json

from config import Config
from graph import OrchardGraph
from agent.belief import support_score, uncertainty, DiscoveryCurve, summarize


def _graph_3(score=0.7):
    """Three fruit nodes, support 1, no jitter."""
    g = OrchardGraph()
    for i in range(3):
        nid = g.add_candidate([i * 10, 0, i * 10 + 8, 8], score, found_in_pass=1)
        g.nodes[nid].classification = "fruit"
    return g


# ----------------------- DiscoveryCurve -----------------------

def test_discovery_curve_mean_and_saturation():
    d = DiscoveryCurve()
    for x in (5, 2, 0, 0):
        d.append(x)
    assert d.mean_recent(3) == (2 + 0 + 0) / 3
    assert d.saturated(3, 1.0) is True          # recent window averages <= 1
    assert d.as_list() == [5, 2, 0, 0]

    busy = DiscoveryCurve()
    for x in (5, 4, 3):
        busy.append(x)
    assert busy.saturated(3, 1.0) is False       # window averages 4 > 1

    empty = DiscoveryCurve()
    assert empty.mean_recent(3) == 0.0
    assert empty.saturated(3, 1.0) is False       # len < m


# ----------------------- support_score -----------------------

def test_support_score_rises_with_support():
    cfg = Config()
    g = OrchardGraph()
    nid = g.add_candidate([0, 0, 10, 10], 0.5, 1)
    node = g.nodes[nid]
    w1 = support_score(node, cfg)
    node.reinforce([0, 0, 10, 10], "s2")         # k=2, no jitter
    w2 = support_score(node, cfg)
    assert w2 > w1


def test_support_score_penalized_by_jitter():
    cfg = Config()
    g = OrchardGraph()
    steady = g.nodes[g.add_candidate([0, 0, 10, 10], 0.6, 1)]
    jittery = g.nodes[g.add_candidate([0, 0, 10, 10], 0.6, 1)]
    steady.reinforce([0, 0, 10, 10], "a")        # displacement 0
    jittery.reinforce([6, 0, 16, 10], "a")       # displacement 3 -> jitter up
    assert support_score(jittery, cfg) < support_score(steady, cfg)


# ----------------------- uncertainty -----------------------

def test_adding_support_lowers_uncertainty():
    cfg = Config()
    g = _graph_3()
    d = DiscoveryCurve()
    d.append(3)
    u_before = uncertainty(g, d, cfg)
    for node in g.nodes.values():
        node.reinforce(node.box, "s2")           # higher k -> lower 1/(1+k)
    u_after = uncertainty(g, d, cfg)
    assert u_after < u_before


def test_flat_discovery_lowers_uncertainty():
    cfg = Config()
    g = _graph_3()
    busy = DiscoveryCurve()
    for x in (5, 5, 5):
        busy.append(x)
    flat = DiscoveryCurve()
    for x in (5, 0, 0):
        flat.append(x)
    assert uncertainty(g, flat, cfg) < uncertainty(g, busy, cfg)


def test_spurious_nodes_excluded_from_uncertainty():
    cfg = Config()
    g = _graph_3()
    d = DiscoveryCurve()
    d.append(0)
    u_all_fruit = uncertainty(g, d, cfg)
    # mark one node spurious -> its instability term drops out -> U decreases
    list(g.nodes.values())[0].classification = "spurious"
    assert uncertainty(g, d, cfg) < u_all_fruit


# ----------------------- summarize -----------------------

def test_summarize_complete_and_json_serializable():
    cfg = Config()
    g = _graph_3()
    list(g.nodes.values())[0].signatures.add("2:tiled:green fruit:0.35")
    d = DiscoveryCurve()
    for x in (3, 1):
        d.append(x)

    phi = summarize(g, d, budget=10, cfg=cfg)
    json.dumps(phi)  # must not raise

    assert phi["K"] == 3
    assert phi["n_t"] == 1
    assert phi["D"] == [3, 1]
    assert len(phi["w"]) == len(phi["s"]) == len(phi["k"]) == len(phi["delta"]) == 3
    assert phi["tiling_status"] is True
    assert phi["remaining_budget"] == 10


def test_summarize_tiling_status_false_without_tiled_signature():
    cfg = Config()
    g = _graph_3()
    g.nodes[list(g.nodes)[0]].signatures.add("1:global:green fruit:0.35")
    phi = summarize(g, DiscoveryCurve(), budget=5, cfg=cfg)
    assert phi["tiling_status"] is False
    assert phi["n_t"] == 0

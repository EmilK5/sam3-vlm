"""
Tests for agent/policy_vlm.choose (v2 full-history signature): stop mapping, the
empty-graph stop guard, and the fallback paths (malformed responses and the
removed tile/verify/query/subdivide actions all fall back to the heuristic). No
network. The look-mapping / overlay / history surface lives in
test_policy_vlm_v2.py; look guards live in test_policy_vlm_roi.py.
"""

import json
import types

import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import LookROIA, StopA
from agent import policy_vlm, policy_heuristic


TARGET = "green fruit"
IMG = Image.new("RGB", (256, 256))
TREE_ROI = [0, 0, 256, 256]


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _fixture():
    """A graph (with tree_roi set) holding one unresolved node and one fruit node."""
    cfg = Config()
    graph = OrchardGraph()
    graph.tree_roi = list(TREE_ROI)
    graph.add_candidate([10, 10, 40, 40], 0.9, 1)                        # stays unresolved
    fruit = graph.add_candidate([200, 200, 230, 230], 0.9, 1)
    graph.nodes[fruit].classification = "fruit"
    graph.nodes[fruit].reinforce([200, 200, 230, 230], "s")             # high support
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    return cfg, graph, phi


def _choose(content, fixture, image=IMG):
    cfg, graph, phi = fixture
    return policy_vlm.choose(phi, [], graph, cfg, client=_FakeClient(content), image=image)


def _heuristic_action(fixture):
    cfg, graph, phi = fixture
    return policy_heuristic.choose(phi, [tuple(TREE_ROI)], cfg)


# ----------------------- legal stop -----------------------

def test_legal_stop_maps_to_stopA():
    action = _choose('{"action": "stop", "args": {"estimate_name": "N_supp"}}', _fixture())
    assert isinstance(action, StopA) and action.estimate_name == "N_supp"


def test_stop_on_empty_graph_falls_back():
    # Vacuous-stop guard: a stop before any candidate is registered must fall back
    # to the heuristic (which senses) rather than terminate the episode at zero.
    cfg = Config()
    graph = OrchardGraph()
    graph.tree_roi = list(TREE_ROI)  # empty: nothing sensed yet
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    action = policy_vlm.choose(
        phi, [], graph, cfg,
        client=_FakeClient('{"action": "stop", "args": {"estimate_name": "N_obs"}}'),
        image=IMG,
    )
    assert not isinstance(action, StopA)


# ----------------------- illegal / malformed -> fallback -----------------------

@pytest.mark.parametrize("content", [
    "not json at all",
    "{}",                                                              # missing "action"
    '{"action": "query", "args": {"region_id": 0, "conf": 0.4}}',     # removed action
    '{"action": "subdivide", "args": {"region_id": 0}}',              # removed action
    '{"action": "tile", "args": {"conf": 0.3}}',                      # removed in v2
    '{"action": "verify", "args": {"node_ids": ["node_0000"]}}',      # removed in v2
    '{"action": "stop", "args": {"estimate_name": "N_bogus"}}',       # bad estimator
    '{"action": "teleport", "args": {}}',                             # unknown action
])
def test_illegal_responses_fall_back_to_heuristic(content):
    fx = _fixture()
    assert _choose(content, fx) == _heuristic_action(fx)   # identical to the heuristic's choice


def test_menu_is_look_stop_only():
    cfg, graph, phi = _fixture()
    body = policy_vlm._build_body(phi, [], graph, cfg, IMG, TREE_ROI)
    assert set(body["action_menu"]) == {"look", "stop"}
    for removed in ("tile", "verify", "query", "subdivide"):
        assert removed not in body["action_menu"]


# ----------------------- make_vlm_policy (episode adapter, v2) -----------------------

def test_make_vlm_policy_shows_history_and_chooses_look():
    from agent import runner
    from agent.actions import ActionContext

    cfg, graph, phi = _fixture()
    ctx = ActionContext(image_pil=IMG, graph=graph, cfg=cfg, partition=[tuple(TREE_ROI)])
    ctx.history = [{"t": 1, "x": {"action": "global_pass"}, "y": {"n_new": 2}}]

    vlm_client = _FakeClient('{"action": "look", "args": {"region": [20, 20, 120, 120]}}')
    policy = runner.make_vlm_policy(ctx, vlm_client=vlm_client)
    action = policy(phi, [tuple(TREE_ROI)], cfg)
    assert isinstance(action, LookROIA) and action.region == (20, 20, 120, 120)

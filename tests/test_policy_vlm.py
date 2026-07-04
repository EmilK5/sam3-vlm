"""
Tests for agent/policy_vlm.choose: legal responses map to the right dataclass;
malformed / illegal responses all fall back to the heuristic policy. No network.
"""

import json
import types

import pytest

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import QueryA, TileQueryA, SubdivideA, VerifyA, StopA
from agent import policy_vlm, policy_heuristic


TARGET = "green fruit"  # matches getattr(cfg, "target_prompt", "green fruit")


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _fixture():
    """A graph with one unresolved node (verifiable) and one fruit node; phi + partition."""
    cfg = Config()
    graph = OrchardGraph()
    unresolved = graph.add_candidate([10, 10, 40, 40], 0.9, 1)          # stays unresolved
    fruit = graph.add_candidate([200, 200, 230, 230], 0.9, 1)
    graph.nodes[fruit].classification = "fruit"
    graph.nodes[fruit].reinforce([200, 200, 230, 230], "s")             # high support -> not low-w
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    partition = [(0, 0, 128, 128), (128, 0, 256, 128)]
    z = {"target_present": True, "density": "dense", "object_scale": "small",
         "occlusion": "medium", "recommend": "tile", "notes": ""}
    return cfg, graph, phi, partition, z, unresolved, fruit


def _choose(content, fixture):
    cfg, graph, phi, partition, z, _, _ = fixture
    return policy_vlm.choose(phi, z, partition, graph, cfg, client=_FakeClient(content))


# ----------------------- legal responses -> correct dataclass -----------------------

def test_legal_query_maps_to_queryA():
    fx = _fixture()
    action = _choose('{"action": "query", "args": {"region_id": 1, "conf": 0.4}}', fx)
    assert isinstance(action, QueryA)
    assert action.region == fx[3][1]          # partition[1]
    assert action.prompt == TARGET and action.conf == 0.4


def test_legal_tile_maps_to_tileA():
    action = _choose('{"action": "tile", "args": {"conf": 0.3}}', _fixture())
    assert isinstance(action, TileQueryA) and action.conf == 0.3


def test_legal_subdivide_maps_to_subdivideA():
    fx = _fixture()
    action = _choose('{"action": "subdivide", "args": {"region_id": 0}}', fx)
    assert isinstance(action, SubdivideA) and action.region == fx[3][0]


def test_legal_verify_maps_to_verifyA():
    fx = _fixture()
    unresolved_id = fx[5]
    action = _choose(json.dumps({"action": "verify", "args": {"node_ids": [unresolved_id]}}), fx)
    assert isinstance(action, VerifyA) and action.node_ids == [unresolved_id]


def test_legal_stop_maps_to_stopA():
    action = _choose('{"action": "stop", "args": {"estimate_name": "N_supp"}}', _fixture())
    assert isinstance(action, StopA) and action.estimate_name == "N_supp"


# ----------------------- illegal / malformed -> fallback -----------------------

def _heuristic_action(fx):
    cfg, graph, phi, partition, z, _, _ = fx
    return policy_heuristic.choose(phi, partition, cfg)


@pytest.mark.parametrize("content", [
    "not json at all",
    "{}",                                                              # missing "action"
    '{"action": "query", "args": {"region_id": 5, "conf": 0.4}}',     # region_id out of range
    '{"action": "query", "args": {"region_id": 0, "conf": 1.5}}',     # conf out of range
    '{"action": "query", "args": {"region_id": 0, "conf": 0.4, "prompt": "apples"}}',  # wrong prompt
    '{"action": "subdivide", "args": {"region_id": -1}}',             # bad region
    '{"action": "verify", "args": {"node_ids": ["node_nope"]}}',      # id not in graph
    '{"action": "stop", "args": {"estimate_name": "N_bogus"}}',       # bad estimator
    '{"action": "teleport", "args": {}}',                             # unknown action
])
def test_illegal_responses_fall_back_to_heuristic(content):
    fx = _fixture()
    action = _choose(content, fx)
    assert action == _heuristic_action(fx)   # identical to the heuristic's choice


def test_verify_of_confident_fruit_node_falls_back():
    # The fruit node is neither unresolved nor low-w, so verifying it is illegal.
    fx = _fixture()
    fruit_id = fx[6]
    action = _choose(json.dumps({"action": "verify", "args": {"node_ids": [fruit_id]}}), fx)
    assert action == _heuristic_action(fx)


def test_no_action_accepts_raw_coordinates():
    # Even a "query" carrying a bbox must be treated by region_id only; a coords-only
    # payload (no valid region_id) cannot inject a box -> fallback.
    fx = _fixture()
    action = _choose('{"action": "query", "args": {"bbox": [0, 0, 10, 10], "conf": 0.4}}', fx)
    assert action == _heuristic_action(fx)

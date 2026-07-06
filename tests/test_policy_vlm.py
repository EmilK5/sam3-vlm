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


def _fixture(verifier_mode="vip"):
    """A graph with one unresolved node (verifiable) and one fruit node; phi + partition.

    verifier_mode defaults to "vip" so the verify action is on the menu; pass
    "ioc" to exercise the verify-unavailable path.
    """
    import dataclasses
    cfg = dataclasses.replace(Config(), verifier_mode=verifier_mode)
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
# NOTE: query and subdivide were removed from the VLM menu in step 7.3 (replaced by
# image-grounded "look" ROIs; see tests/test_policy_vlm_roi.py). A query/subdivide
# response is now an unknown action and falls back (covered in the illegal cases).

def test_legal_tile_maps_to_tileA():
    action = _choose('{"action": "tile", "args": {"conf": 0.3}}', _fixture())
    assert isinstance(action, TileQueryA) and action.conf == 0.3


def test_legal_verify_maps_to_verifyA():
    fx = _fixture()
    unresolved_id = fx[5]
    action = _choose(json.dumps({"action": "verify", "args": {"node_ids": [unresolved_id]}}), fx)
    assert isinstance(action, VerifyA) and action.node_ids == [unresolved_id]


def test_legal_stop_maps_to_stopA():
    action = _choose('{"action": "stop", "args": {"estimate_name": "N_supp"}}', _fixture())
    assert isinstance(action, StopA) and action.estimate_name == "N_supp"


def test_stop_on_empty_graph_falls_back():
    # Vacuous-stop guard: a stop before any candidate is registered must fall back
    # to the heuristic (which senses) rather than terminate the episode at zero.
    import dataclasses
    cfg = dataclasses.replace(Config(), verifier_mode="ioc")
    graph = OrchardGraph()  # empty: nothing sensed yet
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    partition = [(0, 0, 128, 128)]
    z = {"target_present": True, "density": "dense", "object_scale": "small",
         "occlusion": "medium", "recommend": "tile", "notes": ""}
    action = policy_vlm.choose(
        phi, z, partition, graph, cfg,
        client=_FakeClient('{"action": "stop", "args": {"estimate_name": "N_obs"}}'),
    )
    assert not isinstance(action, StopA)


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


def test_verify_without_vip_verifier_falls_back():
    # verify needs the FM+V-IP oracle; under verifier_mode="ioc" it is not on the
    # menu and a VLM response choosing it must fall back (episode wiring has no
    # oracle/query_set to execute it with).
    fx = _fixture(verifier_mode="ioc")
    unresolved_id = fx[5]
    action = _choose(json.dumps({"action": "verify", "args": {"node_ids": [unresolved_id]}}), fx)
    assert not isinstance(action, VerifyA)
    assert action == _heuristic_action(fx)


def test_verify_not_in_menu_without_vip_verifier():
    fx = _fixture(verifier_mode="ioc")
    cfg, graph, phi, partition, z, _, _ = fx
    messages = policy_vlm._build_messages(phi, z, graph, cfg)   # no image -> text content
    body = json.loads(messages[-1]["content"].split("\n")[1])
    assert "verify" not in body["action_menu"]
    assert body["verifiable_node_ids"] == []


def test_no_action_accepts_raw_coordinates():
    # Even a "query" carrying a bbox must be treated by region_id only; a coords-only
    # payload (no valid region_id) cannot inject a box -> fallback.
    fx = _fixture()
    action = _choose('{"action": "query", "args": {"bbox": [0, 0, 10, 10], "conf": 0.4}}', fx)
    assert action == _heuristic_action(fx)


# ----------------------- make_vlm_policy (episode adapter) -----------------------

def test_make_vlm_policy_inspects_then_chooses():
    from agent import runner
    from agent.actions import ActionContext
    from PIL import Image

    cfg, graph, phi, partition, z, _, _ = _fixture()
    ctx = ActionContext(image_pil=Image.new("RGB", (256, 128)), graph=graph, cfg=cfg,
                        partition=partition)

    inspect_client = _FakeClient('{"target_present": true, "density": "dense", '
                                 '"object_scale": "small", "occlusion": "low", '
                                 '"recommend": "tile", "notes": ""}')
    vlm_client = _FakeClient('{"action": "tile", "args": {"conf": 0.3}}')

    policy = runner.make_vlm_policy(ctx, inspect_client=inspect_client, vlm_client=vlm_client)
    action = policy(phi, partition, cfg)   # t=1 -> inspect fires, then VLM chooses tile
    assert isinstance(action, TileQueryA) and action.conf == 0.3

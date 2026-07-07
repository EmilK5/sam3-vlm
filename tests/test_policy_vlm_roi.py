"""
Tests for the guided-ROI VLM policy (step 7.3): the VLM sees the image and picks
from a look/tile/verify/stop menu; a legal "look" maps to a LookROIA sensing target,
malformed/out-of-bounds looks fall back to the heuristic, and query/subdivide are
gone. No network (a fake client returns a canned response).
"""

import dataclasses
import json
import types

from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import LookROIA, TileQueryA, VerifyA, StopA
from agent import policy_vlm


TARGET = "green fruit"
IMG = Image.new("RGB", (200, 200))
PARTITION = [(0, 0, 200, 200)]


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _fixture(verifier_mode="vip"):
    cfg = dataclasses.replace(Config(), verifier_mode=verifier_mode)
    graph = OrchardGraph()
    unresolved = graph.add_candidate([10, 10, 40, 40], 0.9, 1)      # stays unresolved
    fruit = graph.add_candidate([120, 120, 150, 150], 0.9, 1)
    graph.nodes[fruit].classification = "fruit"
    graph.nodes[fruit].reinforce([120, 120, 150, 150], "s")         # high support -> not low-w
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    z = {"target_present": True, "density": "dense", "object_scale": "small",
         "occlusion": "medium", "recommend": "look", "notes": ""}
    return cfg, graph, phi, z, unresolved, fruit


def _choose(content, fx, image=IMG):
    cfg, graph, phi, z, _, _ = fx
    return policy_vlm.choose(phi, z, PARTITION, graph, cfg, client=_FakeClient(content), image=image)


# ----------------------- look -> LookROIA -----------------------

def test_legal_look_maps_to_lookroia():
    action = _choose('{"action": "look", "args": {"region": [40, 40, 120, 120]}}', _fixture())
    assert isinstance(action, LookROIA)
    assert action.region == (40, 40, 120, 120)


def test_look_out_of_bounds_falls_back():
    action = _choose('{"action": "look", "args": {"region": [40, 40, 900, 900]}}', _fixture())
    assert not isinstance(action, LookROIA)   # 900 > image 200 -> invalid -> heuristic


def test_look_wrong_arity_falls_back():
    action = _choose('{"action": "look", "args": {"region": [40, 40, 120]}}', _fixture())
    assert not isinstance(action, LookROIA)


def test_look_degenerate_box_falls_back():
    action = _choose('{"action": "look", "args": {"region": [120, 120, 40, 40]}}', _fixture())
    assert not isinstance(action, LookROIA)   # x2 < x1


def test_look_region_as_json_string_still_maps_to_lookroia():
    # Some local VLMs (observed with Qwen3-VL via Ollama) stringify nested JSON
    # values instead of emitting a real array; we should still parse it.
    action = _choose('{"action": "look", "args": {"region": "[40, 40, 120, 120]"}}', _fixture())
    assert isinstance(action, LookROIA)
    assert action.region == (40, 40, 120, 120)


def test_look_region_as_malformed_string_falls_back():
    action = _choose('{"action": "look", "args": {"region": "not json"}}', _fixture())
    assert not isinstance(action, LookROIA)


def test_look_without_image_falls_back():
    action = _choose('{"action": "look", "args": {"region": [40, 40, 120, 120]}}',
                     _fixture(), image=None)
    assert not isinstance(action, LookROIA)   # no frame to place the box


# ----------------------- menu / prompt shape -----------------------

def test_menu_has_look_not_query_or_subdivide():
    cfg, graph, phi, z, _, _ = _fixture()
    body = policy_vlm._build_body(phi, z, graph, cfg, IMG)
    menu = body["action_menu"]
    assert set(("look", "tile", "verify", "stop")) <= set(menu)
    assert "query" not in menu and "subdivide" not in menu
    assert body["image_size"] == [200, 200]


def test_image_is_sent_in_the_request():
    cfg, graph, phi, z, _, _ = _fixture()
    messages = policy_vlm._build_messages(phi, z, graph, cfg, IMG)
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_look_absent_from_menu_without_image():
    cfg, graph, phi, z, _, _ = _fixture()
    body = policy_vlm._build_body(phi, z, graph, cfg, None)
    assert "look" not in body["action_menu"]
    assert body["image_size"] is None


# ----------------------- other actions still validate -----------------------

def test_tile_still_maps_to_tileA():
    action = _choose('{"action": "tile", "args": {"conf": 0.3}}', _fixture())
    assert isinstance(action, TileQueryA) and action.conf == 0.3


def test_tile_conf_as_json_string_still_maps_to_tileA():
    action = _choose('{"action": "tile", "args": {"conf": "0.3"}}', _fixture())
    assert isinstance(action, TileQueryA) and action.conf == 0.3


def test_verify_still_maps_to_verifyA():
    fx = _fixture()
    unresolved_id = fx[4]
    action = _choose(json.dumps({"action": "verify", "args": {"node_ids": [unresolved_id]}}), fx)
    assert isinstance(action, VerifyA) and action.node_ids == [unresolved_id]


def test_verify_node_ids_as_json_string_still_maps_to_verifyA():
    fx = _fixture()
    unresolved_id = fx[4]
    content = json.dumps({"action": "verify", "args": {"node_ids": json.dumps([unresolved_id])}})
    action = _choose(content, fx)
    assert isinstance(action, VerifyA) and action.node_ids == [unresolved_id]


def test_stop_on_nonempty_graph_maps_to_stopA():
    action = _choose('{"action": "stop", "args": {"estimate_name": "N_supp"}}', _fixture())
    assert isinstance(action, StopA) and action.estimate_name == "N_supp"

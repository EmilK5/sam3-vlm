"""
Tests for the v2 "look" validation surface of policy_vlm.choose: a legal look
maps to a LookROIA sensing target; malformed / stringified / out-of-tree-ROI /
imageless looks fall back to the heuristic. No network (a fake client returns a
canned response).

Since v2 the heuristic fallback itself emits a LookROIA (a grid cell of the tree
ROI), so "fell back" is asserted as equality with the heuristic's action, never
by action type.
"""

import types

from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import LookROIA
from agent import policy_vlm, policy_heuristic


IMG = Image.new("RGB", (200, 200))
TREE_ROI = [20, 20, 180, 180]   # strictly inside the frame, so "inside frame but
                                # outside tree ROI" is a testable region


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _fixture():
    cfg = Config()
    graph = OrchardGraph()
    graph.tree_roi = list(TREE_ROI)
    graph.add_candidate([30, 30, 60, 60], 0.9, 1)
    fruit = graph.add_candidate([120, 120, 150, 150], 0.9, 1)
    graph.nodes[fruit].classification = "fruit"
    graph.nodes[fruit].reinforce([120, 120, 150, 150], "s")
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    return cfg, graph, phi


def _choose(content, fx, image=IMG):
    cfg, graph, phi = fx
    return policy_vlm.choose(phi, [], graph, cfg, client=_FakeClient(content), image=image)


def _heuristic_action(fx, image=IMG):
    cfg, graph, phi = fx
    tree_roi = policy_vlm._resolve_tree_roi(graph, image)
    partition = [tuple(tree_roi)] if tree_roi is not None else [(0.0, 0.0, 0.0, 0.0)]
    return policy_heuristic.choose(phi, partition, cfg)


# ----------------------- look -> LookROIA -----------------------

def test_legal_look_inside_tree_roi_maps_to_lookroia():
    action = _choose('{"action": "look", "args": {"region": [30, 30, 120, 120]}}', _fixture())
    assert isinstance(action, LookROIA)
    assert action.region == (30, 30, 120, 120)


def test_look_region_as_json_string_still_maps_to_lookroia():
    # Some local VLMs (Qwen3-VL via Ollama) stringify nested JSON values.
    action = _choose('{"action": "look", "args": {"region": "[30, 30, 120, 120]"}}', _fixture())
    assert isinstance(action, LookROIA)
    assert action.region == (30, 30, 120, 120)


# ----------------------- look guards -> fallback -----------------------

def test_look_outside_tree_roi_falls_back():
    # In-bounds of the frame (200x200) but x1=5 < tree ROI x1=20 -> illegal.
    fx = _fixture()
    action = _choose('{"action": "look", "args": {"region": [5, 30, 120, 120]}}', fx)
    assert action == _heuristic_action(fx)
    assert action.region != (5, 30, 120, 120)   # the VLM box was NOT adopted


def test_look_out_of_frame_falls_back():
    fx = _fixture()
    action = _choose('{"action": "look", "args": {"region": [30, 30, 900, 900]}}', fx)
    assert action == _heuristic_action(fx)


def test_look_wrong_arity_falls_back():
    fx = _fixture()
    action = _choose('{"action": "look", "args": {"region": [30, 30, 120]}}', fx)
    assert action == _heuristic_action(fx)


def test_look_degenerate_box_falls_back():
    fx = _fixture()
    action = _choose('{"action": "look", "args": {"region": [120, 120, 30, 30]}}', fx)  # x2<x1
    assert action == _heuristic_action(fx)


def test_look_region_as_malformed_string_falls_back():
    fx = _fixture()
    action = _choose('{"action": "look", "args": {"region": "not json"}}', fx)
    assert action == _heuristic_action(fx)


def test_look_without_image_falls_back():
    fx = _fixture()
    action = _choose('{"action": "look", "args": {"region": [30, 30, 120, 120]}}', fx, image=None)
    # No frame -> look isn't offered; falls back with graph.tree_roi as the anchor.
    assert action == _heuristic_action(fx, image=None)
    assert getattr(action, "region", None) != (30, 30, 120, 120)


# ----------------------- menu / prompt shape -----------------------

def test_look_absent_from_menu_without_image():
    cfg, graph, phi = _fixture()
    body = policy_vlm._build_body(phi, [], graph, cfg, None, None)
    assert "look" not in body["action_menu"]
    assert body["image_size"] is None

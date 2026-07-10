"""
Step 8.3 spec tests: the full-history VLM policy + the overlay image.

Covers: the request carries exactly two image parts (raw + overlay); the full
episode history is embedded verbatim in the prompt body; the menu is look/stop;
a legal look inside the tree ROI maps to LookROIA while a look inside the frame
but outside the tree ROI falls back; and render_overlay draws a same-size PIL
frame with no model call. No network.
"""

import json
import types

from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import LookROIA, ActionContext
from agent.history import EpisodeHistory
from agent import policy_vlm, policy_heuristic, runner
from agent.overlay import render_overlay


FRAME = Image.new("RGB", (300, 300))
TREE_ROI = [50, 50, 250, 250]   # strictly inside the 300x300 frame


class _CaptureClient:
    """Records the create() kwargs (so we can inspect the built messages) and
    returns a canned response."""

    def __init__(self, content):
        self._content = content
        self.captured = None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _fixture():
    cfg = Config()
    graph = OrchardGraph()
    graph.tree_roi = list(TREE_ROI)
    graph.add_candidate([70, 70, 100, 100], 0.9, 1)                      # unresolved
    fruit = graph.add_candidate([150, 150, 180, 180], 0.9, 1)
    graph.nodes[fruit].classification = "fruit"
    graph.nodes[fruit].reinforce([150, 150, 180, 180], "s")
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    return cfg, graph, phi


def _history():
    h = EpisodeHistory()
    h.append({"action": "canopy_roi", "roi": TREE_ROI},
             {"n_new": 0, "new_nodes": [], "totals": {"K": 0, "N_obs": 0}})
    h.append({"action": "leaf_map", "n_leaves": 7},
             {"n_new": 0, "new_nodes": [], "totals": {"K": 0, "N_obs": 0}})
    h.append({"action": "global_pass", "region": TREE_ROI, "prompt": "green fruit", "conf": 0.3},
             {"n_new": 2, "new_nodes": [{"id": "node_x", "box": [70, 70, 100, 100],
                                         "conf": 0.9, "class": "unresolved"}],
              "totals": {"K": 2, "N_obs": 2}})
    return h


def _user_parts(client):
    return client.captured["messages"][-1]["content"]


def _body_from_capture(client):
    text = next(p["text"] for p in _user_parts(client) if p["type"] == "text")
    # text = "<line0>\n<json>\n<line2>"; recover the middle JSON.
    return json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])


# ----------------------- prompt shape: two images + verbatim history -----------------------

def test_request_carries_exactly_two_image_parts():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "stop", "args": {"estimate_name": "N_obs"}}')
    policy_vlm.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    parts = _user_parts(client)
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 2                                  # raw frame + overlay
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in images)


def test_prompt_body_embeds_full_history_verbatim():
    cfg, graph, phi = _fixture()
    history = _history()
    client = _CaptureClient('{"action": "stop", "args": {"estimate_name": "N_obs"}}')
    policy_vlm.choose(phi, history, graph, cfg, client=client, image=FRAME)
    body = _body_from_capture(client)
    assert body["history"] == history.as_list()             # complete x_1^t, y_1^t
    assert [r["x"]["action"] for r in body["history"]] == ["canopy_roi", "leaf_map", "global_pass"]
    assert body["tree_roi"] == [float(v) for v in TREE_ROI]
    assert set(body["action_menu"]) == {"look", "stop"}


# ----------------------- look validation against the tree ROI -----------------------

def test_legal_look_inside_tree_roi_maps_to_lookroia():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "look", "args": {"region": [60, 60, 240, 240]}}')
    action = policy_vlm.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert isinstance(action, LookROIA) and action.region == (60, 60, 240, 240)


def test_look_inside_frame_but_outside_tree_roi_falls_back():
    cfg, graph, phi = _fixture()
    # [10,60,240,240] is in-frame (300x300) but x1=10 < tree ROI x1=50 -> illegal.
    client = _CaptureClient('{"action": "look", "args": {"region": [10, 60, 240, 240]}}')
    action = policy_vlm.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    expected = policy_heuristic.choose(phi, [tuple(TREE_ROI)], cfg)
    assert action == expected                                # fell back to the heuristic
    assert action.region != (10, 60, 240, 240)               # VLM box not adopted


# ----------------------- overlay render (PIL only) -----------------------

def test_render_overlay_returns_same_size_rgb_frame():
    cfg, graph, phi = _fixture()
    sensed = [(60, 60, 120, 120)]
    out = render_overlay(FRAME, graph, sensed_rois=sensed, tree_roi=TREE_ROI)
    assert isinstance(out, Image.Image)
    assert out.size == FRAME.size and out.mode == "RGB"
    assert out is not FRAME                                   # a copy, original untouched


def test_render_overlay_handles_empty_graph_and_no_rois():
    out = render_overlay(FRAME, OrchardGraph(), sensed_rois=None, tree_roi=None)
    assert out.size == FRAME.size


# ----------------------- make_vlm_policy wiring (history + sensed_rois, no inspect) -----------------------

def test_make_vlm_policy_passes_history_and_sensed_rois():
    cfg, graph, phi = _fixture()
    ctx = ActionContext(image_pil=FRAME, graph=graph, cfg=cfg, partition=[tuple(TREE_ROI)])
    ctx.history = _history()
    ctx.sensed_rois = [(60, 60, 120, 120)]

    client = _CaptureClient('{"action": "look", "args": {"region": [60, 60, 240, 240]}}')
    policy = runner.make_vlm_policy(ctx, vlm_client=client)
    action = policy(phi, [tuple(TREE_ROI)], cfg)

    assert isinstance(action, LookROIA)
    body = _body_from_capture(client)
    assert body["history"] == ctx.history.as_list()          # the ctx history reached the prompt
    images = [p for p in _user_parts(client) if p.get("type") == "image_url"]
    assert len(images) == 2                                   # raw + overlay (with sensed ROI drawn)


def test_make_vlm_policy_signature_has_no_inspect_client():
    import inspect as _inspect
    params = list(_inspect.signature(runner.make_vlm_policy).parameters)
    assert "inspect_client" not in params                    # z/inspect retired in v2
    assert params == ["ctx", "vlm_client"]

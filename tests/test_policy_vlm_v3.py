"""
Step 9.1 spec tests: the prompt-refinement (v3) policy.

The region is FIXED to the tree ROI; the VLM chooses the SAM3 text prompt
(1-2 adjectives + noun) + threshold. Covers: the request carries raw+overlay and
the full history + prompts_tried; the menu is {refine, stop}; a legal refine maps
to a GLOBAL QueryA over the tree ROI with a normalized prompt and a clamped
threshold; a missing threshold uses the default; an over-long / empty prompt and a
stop on an empty graph fall back to the heuristic; and with no image, refine is not
offered and any refine response falls back. No network.
"""

import json
import types

from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import QueryA, StopA
from agent.history import EpisodeHistory
from agent import policy_vlm_v3, policy_heuristic


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
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    return cfg, graph, phi


def _history():
    """A confident seed pass (@0.65) that found two candidates -- the record the
    v3 policy distills into prompts_tried."""
    h = EpisodeHistory()
    h.append({"action": "global_pass", "region": TREE_ROI, "prompt": "green fruit", "conf": 0.65},
             {"n_new": 2, "new_nodes": [], "totals": {"K": 2, "N_obs": 2}})
    return h


def _user_parts(client):
    return client.captured["messages"][-1]["content"]


def _body_from_capture(client):
    text = next(p["text"] for p in _user_parts(client) if p["type"] == "text")
    return json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])


# ----------------------- prompt shape: two images + history + trajectory -----------------------

def test_request_carries_two_images_history_and_menu():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "stop", "args": {"estimate_name": "N_obs"}}')
    policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    parts = _user_parts(client)
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 2                                  # raw frame + overlay
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in images)

    body = _body_from_capture(client)
    assert set(body["action_menu"]) == {"refine", "stop"}
    assert body["tree_roi"] == [float(v) for v in TREE_ROI]
    assert body["history"] == _history().as_list()          # complete x_1^t, y_1^t
    assert body["prompts_tried"] == [{"prompt": "green fruit", "conf": 0.65, "n_new": 2}]


# ----------------------- refine validation -----------------------

def test_legal_refine_maps_to_global_queryA_over_tree_roi():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "refine", "args": {"prompt": "round green fruit", "threshold": 0.4}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert isinstance(action, QueryA)
    assert action.region == tuple(float(v) for v in TREE_ROI)    # global pass over the tree ROI
    assert action.prompt == "round green fruit"
    assert action.conf == 0.4


def test_threshold_clamped_to_range():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "refine", "args": {"prompt": "spherical fruit", "threshold": 0.95}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert isinstance(action, QueryA) and action.conf == cfg.refine_conf_max


def test_missing_threshold_uses_default():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "refine", "args": {"prompt": "waxy fruit"}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert isinstance(action, QueryA) and action.conf == cfg.refine_conf_default


def test_overlong_prompt_falls_back():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "refine", "args": {"prompt": "small round waxy green citrus fruit"}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert action == policy_heuristic.choose(phi, [tuple(TREE_ROI)], cfg)   # heuristic net
    assert not (isinstance(action, QueryA) and action.prompt.startswith("small round waxy"))


def test_empty_prompt_falls_back():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "refine", "args": {"prompt": "   "}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert action == policy_heuristic.choose(phi, [tuple(TREE_ROI)], cfg)


# ----------------------- stop -----------------------

def test_stop_on_nonempty_graph():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "stop", "args": {"estimate_name": "N_cons"}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert isinstance(action, StopA) and action.estimate_name == "N_cons"


def test_stop_on_empty_graph_falls_back():
    cfg = Config()
    graph = OrchardGraph()
    graph.tree_roi = list(TREE_ROI)
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    client = _CaptureClient('{"action": "stop", "args": {"estimate_name": "N_obs"}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=FRAME)
    assert not isinstance(action, StopA)                    # empty-graph stop rejected -> fallback


# ----------------------- no image: refine not offered -----------------------

def test_no_image_menu_excludes_refine():
    cfg, graph, phi = _fixture()
    body = policy_vlm_v3._build_body(phi, _history(), graph, cfg, image=None,
                                     tree_roi=[float(v) for v in TREE_ROI])
    assert set(body["action_menu"]) == {"stop"}


def test_no_image_refine_response_falls_back():
    cfg, graph, phi = _fixture()
    client = _CaptureClient('{"action": "refine", "args": {"prompt": "round fruit", "threshold": 0.4}}')
    action = policy_vlm_v3.choose(phi, _history(), graph, cfg, client=client, image=None)
    assert not (isinstance(action, QueryA) and action.prompt == "round fruit")   # VLM text not adopted

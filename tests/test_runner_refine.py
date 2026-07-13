"""
Step 9.2 wiring test: runner.make_refine_policy adapts the v3 prompt-refinement
policy into an episode policy. Mirrors the make_vlm_policy wiring test: the ctx
history reaches the built prompt, and a legal refine response maps to a GLOBAL
QueryA over the tree ROI (text only -- never a box, never a sub-region). Offline,
no SAM3, no network.
"""

import inspect as _inspect
import json
import types

from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import QueryA, ActionContext
from agent.history import EpisodeHistory
from agent import runner


FRAME = Image.new("RGB", (300, 300))
TREE_ROI = [50, 50, 250, 250]


class _CaptureClient:
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
    fruit = graph.add_candidate([150, 150, 180, 180], 0.9, 1)
    graph.nodes[fruit].classification = "fruit"
    phi = summarize(graph, DiscoveryCurve(), budget=12, cfg=cfg)
    return cfg, graph, phi


def _history():
    h = EpisodeHistory()
    h.append({"action": "global_pass", "region": TREE_ROI, "prompt": "green fruit", "conf": 0.65},
             {"n_new": 1, "new_nodes": [], "totals": {"K": 1, "N_obs": 1}})
    return h


def _user_parts(client):
    return client.captured["messages"][-1]["content"]


def _body_from_capture(client):
    text = next(p["text"] for p in _user_parts(client) if p["type"] == "text")
    return json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])


def test_make_refine_policy_wires_history_and_maps_refine_to_global_query():
    cfg, graph, phi = _fixture()
    ctx = ActionContext(image_pil=FRAME, graph=graph, cfg=cfg, partition=[tuple(TREE_ROI)])
    ctx.history = _history()

    client = _CaptureClient('{"action": "refine", "args": {"prompt": "round fruit", "threshold": 0.6}}')
    policy = runner.make_refine_policy(ctx, vlm_client=client)
    action = policy(phi, [tuple(TREE_ROI)], cfg)

    assert isinstance(action, QueryA)
    assert action.region == tuple(float(v) for v in TREE_ROI)      # global pass over the tree ROI
    assert action.prompt == "round fruit" and action.conf == 0.6   # in [floor 0.50, 0.85]

    body = _body_from_capture(client)
    assert body["history"] == ctx.history.as_list()               # ctx history reached the prompt
    images = [p for p in _user_parts(client) if p.get("type") == "image_url"]
    assert len(images) == 2                                        # raw + overlay


def test_make_refine_policy_signature():
    params = list(_inspect.signature(runner.make_refine_policy).parameters)
    assert params == ["ctx", "vlm_client"]

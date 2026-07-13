"""
Step 9.5 spec tests: the refine threshold floor rises with each new prompt, so later
passes (which scan an already-well-covered scene) are progressively stricter and do
not re-admit clutter at a low threshold. The VLM may choose a HIGHER value but never
a lower one; a below-floor or missing value is raised to the floor; the ceiling is
refine_conf_max. No network.
"""

import types

from PIL import Image

from config import Config
from graph import OrchardGraph
from agent.belief import DiscoveryCurve, summarize
from agent.actions import QueryA
from agent.history import EpisodeHistory
from agent import policy_vlm_v3


FRAME = Image.new("RGB", (200, 200))
TREE_ROI = [20, 20, 180, 180]


class _Client:
    def __init__(self, content):
        self._c = content
        self.chat = self
        self.completions = self

    def create(self, **kw):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._c))])


def _fixture():
    cfg = Config()
    g = OrchardGraph()
    g.tree_roi = list(TREE_ROI)
    nid = g.add_candidate([40, 40, 70, 70], 0.9, 1)
    g.nodes[nid].classification = "fruit"
    return cfg, g, summarize(g, DiscoveryCurve(), budget=12, cfg=cfg)


def _history(n_prior_refines=0):
    """A seed pass plus `n_prior_refines` prior refine (QueryA) passes, each with a
    distinct wording so they count toward the rising floor without tripping the
    repeat guard."""
    h = EpisodeHistory()
    h.append({"action": "global_pass", "region": TREE_ROI, "prompt": "green fruit", "conf": 0.65},
             {"n_new": 1, "n_redetected": 0, "n_detections": 1, "new_nodes": [],
              "totals": {"K": 1, "N_obs": 1}})
    for i in range(n_prior_refines):
        h.append({"action": "QueryA", "region": TREE_ROI, "prompt": f"variant{i} fruit", "conf": 0.5},
                 {"n_new": 0, "n_redetected": 1, "n_detections": 1, "new_nodes": [],
                  "totals": {"K": 1, "N_obs": 1}})
    return h


def _refine_conf(cfg, graph, phi, history, threshold):
    client = _Client('{"action": "refine", "args": {"prompt": "fresh wording", "threshold": %s}}' % threshold)
    action = policy_vlm_v3.choose(phi, history, graph, cfg, client=client, image=FRAME)
    assert isinstance(action, QueryA)
    return action.conf


def test_first_refine_below_floor_is_raised_to_default():
    cfg, g, phi = _fixture()
    # 0.30 is below the first-refine floor (refine_conf_default = 0.50) -> raised.
    assert _refine_conf(cfg, g, phi, _history(0), 0.30) == cfg.refine_conf_default


def test_floor_rises_by_step_per_prior_refine():
    cfg, g, phi = _fixture()
    step = cfg.refine_conf_step
    # after 2 prior refines the floor is default + 2*step; a 0.30 request is raised to it.
    expected = cfg.refine_conf_default + 2 * step
    assert _refine_conf(cfg, g, phi, _history(2), 0.30) == expected


def test_vlm_may_choose_stricter_than_floor():
    cfg, g, phi = _fixture()
    # 0.75 is above the first-refine floor and below the ceiling -> kept as chosen.
    assert _refine_conf(cfg, g, phi, _history(0), 0.75) == 0.75


def test_threshold_capped_at_max():
    cfg, g, phi = _fixture()
    assert _refine_conf(cfg, g, phi, _history(0), 0.99) == cfg.refine_conf_max


def test_missing_threshold_uses_current_floor():
    cfg, g, phi = _fixture()
    client = _Client('{"action": "refine", "args": {"prompt": "fresh wording"}}')
    action = policy_vlm_v3.choose(phi, _history(1), g, cfg, client=client, image=FRAME)
    assert action.conf == cfg.refine_conf_default + cfg.refine_conf_step   # floor after 1 prior refine

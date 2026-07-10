"""
Regression tests for the functional-review fixes (2026-07-04).

(Item 1, VerifyA availability, was removed in v2 step 8.1 along with the
VerifyA action itself: verification now happens only inside execute_pass.)

  2. The cached leaf map is keyed by the ROI it was generated under and is
     regenerated when a region-restricted pass changes the frame.
  3. Cross-pass dedup under verifier "vip"/"off" also matches "unresolved"
     tracks, so repeated passes reinforce instead of re-registering (IoC dedup
     unchanged: fruit only).
  4. PassStats carries the pass's actual call counts (n_sam_calls/n_verify_calls).
  5. cfg.vip_epsilon=None defers to the query set; a float overrides it.
  6. QwenOracle survives request (network) exceptions via the same
     retry-then-fallback path as parse failures. (The companion inspect_scene
     check was removed in v2 step 8.3, which deleted agent/inspect.py.)
  7. sam3_phrase survives query-set loading, so Sam3Oracle can answer.
"""

import dataclasses
import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from config import Config
from graph import OrchardGraph
from verifier.queries import load_query_set
from verifier.oracle import MockOracle, QwenOracle, Sam3Oracle
from verifier.verify import verify_candidate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


# ----------------------- stubbed torch/inference -> real pipeline -----------------------

@pytest.fixture
def pipeline_module():
    """Import the real pipeline against a functioning stub `inference` module:
    run_raw_inference returns one fixed candidate box (per prompt) and
    apply_nms_dualgate is an identity pass-through. Records every SAM3 call."""
    saved = {name: sys.modules.get(name) for name in ("torch", "inference", "pipeline")}
    sys.modules["torch"] = types.ModuleType("torch")

    inf = types.ModuleType("inference")
    inf.calls = []

    def run_raw_inference(processor, image_np, confidence, prompt, pos_boxes=None,
                          neg_boxes=None, disable_size_filter=False, return_masks=False):
        inf.calls.append(prompt)
        if prompt == "green leaf":
            boxes = np.array([[60.0, 60.0, 90.0, 90.0]])
        else:  # target / canopy prompts: one candidate box
            boxes = np.array([[10.0, 10.0, 30.0, 30.0]])
        scores = np.array([0.9])
        if return_masks:
            h, w = image_np.shape[:2]
            m = np.zeros((h, w), dtype=bool)
            b = boxes[0].astype(int)
            m[b[1]:b[3], b[0]:b[2]] = True
            return boxes, scores, [m]
        return boxes, scores

    def apply_nms_dualgate(boxes, scores, confidence, use_concentric=False,
                           masks=None, return_indices=False, **kw):
        idx = np.arange(len(boxes))
        if return_indices:
            return boxes, scores, idx
        return boxes, scores

    inf.run_raw_inference = run_raw_inference
    inf.apply_nms_dualgate = apply_nms_dualgate
    inf.apply_nms = apply_nms_dualgate
    inf.apply_clahe = lambda img: img
    sys.modules["inference"] = inf
    sys.modules.pop("pipeline", None)

    import pipeline
    pipeline._stub_inference = inf

    try:
        yield pipeline
    finally:
        sys.modules.pop("pipeline", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


# ----------------------- 2. leaf cache keyed by ROI -----------------------

def _leaf_calls(pipeline):
    return sum(1 for p in pipeline._stub_inference.calls if p == "green leaf")


def test_leaf_map_regenerates_when_roi_changes(pipeline_module):
    graph = OrchardGraph()
    img = Image.new("RGB", (100, 100))

    pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False, tiling=False,
                                 pass_number=1, prompt="green fruit",
                                 roi_override=(0, 0, 100, 100))
    assert _leaf_calls(pipeline_module) == 1
    assert graph.cached_leaf_roi == [0, 0, 100, 100]

    # different ROI -> the cached (ROI-relative) leaf boxes are invalid: regenerate
    pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False, tiling=False,
                                 pass_number=2, prompt="green fruit",
                                 roi_override=(0, 0, 50, 50))
    assert _leaf_calls(pipeline_module) == 2
    assert graph.cached_leaf_roi == [0, 0, 50, 50]

    # same ROI again -> reuse the cache
    pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False, tiling=False,
                                 pass_number=3, prompt="green fruit",
                                 roi_override=(0, 0, 50, 50))
    assert _leaf_calls(pipeline_module) == 2


# ----------------------- 3. dedup under vip / off -----------------------

def _register_twice(pipeline_module, cfg, oracle=None, query_set=None, image_np=None):
    graph = OrchardGraph()
    box = np.array([[10.0, 10.0, 18.0, 18.0]])  # <12px a side: vip leaves unresolved
    score = np.array([0.9])
    for p in (1, 2):
        added, dup = pipeline_module.register_and_verify_candidates(
            box, score, leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=p,
            cfg=cfg, oracle=oracle, query_set=query_set, image_np=image_np,
        )
    return graph, added, dup


def test_off_mode_dedups_unresolved_across_passes(pipeline_module):
    cfg = dataclasses.replace(Config(), verifier_mode="off")
    graph, added, dup = _register_twice(pipeline_module, cfg)
    assert len(graph.nodes) == 1          # reinforced, not re-registered
    assert (added, dup) == (0, 1)         # second pass: pure duplicate
    assert list(graph.nodes.values())[0].support == 2


def test_vip_mode_dedups_unresolved_across_passes(pipeline_module):
    cfg = dataclasses.replace(Config(), verifier_mode="vip")
    qs = load_query_set(GREEN_CITRUS)
    graph, added, dup = _register_twice(
        pipeline_module, cfg, oracle=MockOracle("target"), query_set=qs,
        image_np=np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(graph.nodes) == 1
    assert (added, dup) == (0, 1)


def test_ioc_dedup_still_matches_fruit_only(pipeline_module):
    # Golden-path guarantee: an unresolved node never absorbs an IoC candidate.
    graph = OrchardGraph()
    nid = graph.add_candidate([10, 10, 30, 30], 0.9, 1)
    graph.nodes[nid].classification = "unresolved"
    added, dup = pipeline_module.register_and_verify_candidates(
        np.array([[10.0, 10.0, 30.0, 30.0]]), np.array([0.9]),
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=2, cfg=None,
    )
    assert (added, dup) == (1, 0)         # registered as a new node, as before
    assert len(graph.nodes) == 2


# ----------------------- 4. PassStats call counts -----------------------

def test_pass_stats_counts_actual_sam_calls(pipeline_module):
    graph = OrchardGraph()
    img = Image.new("RGB", (100, 100))
    stats = pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False,
                                         tiling=False, pass_number=1, prompt="green fruit",
                                         roi_override=(0, 0, 100, 100))
    assert stats.n_sam_calls == 2         # leaf map + global proposal (no canopy: override)
    stats2 = pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False,
                                          tiling=False, pass_number=2, prompt="green fruit",
                                          roi_override=(0, 0, 100, 100))
    assert stats2.n_sam_calls == 1        # cached leaf map -> proposal only


def test_pass_stats_counts_vip_oracle_calls(pipeline_module):
    graph = OrchardGraph()
    img = Image.new("RGB", (100, 100))
    cfg = dataclasses.replace(Config(), verifier_mode="vip")
    qs = load_query_set(GREEN_CITRUS)
    stats = pipeline_module.execute_pass(None, img, graph, conf=0.35, clahe=False,
                                         tiling=False, pass_number=1, prompt="green fruit",
                                         cfg=cfg, oracle=MockOracle("target"), query_set=qs,
                                         roi_override=(0, 0, 100, 100))
    assert stats.n_verify_calls == 1      # one candidate verified, batched = 1 call
    assert int(stats) == 1


# ----------------------- 5. vip_epsilon override -----------------------

def test_vip_epsilon_none_defers_to_query_set_and_float_overrides():
    qs = load_query_set(GREEN_CITRUS)  # epsilon 0.15
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    box = [10, 10, 50, 50]
    oracle = MockOracle("target")

    r_default = verify_candidate(image, box, oracle, qs, Config())  # vip_epsilon=None
    r_noisy = verify_candidate(image, box, oracle, qs,
                               dataclasses.replace(Config(), vip_epsilon=0.45))

    assert r_default["p_target"] >= 0.9                 # clean answers, low noise
    assert r_noisy["p_target"] < r_default["p_target"]  # more assumed noise -> less confident
    assert qs.epsilon == 0.15                           # the loaded set is not mutated


# ----------------------- 6. request-exception fallbacks -----------------------

class _ExplodingClient:
    def __init__(self):
        def _create(**kw):
            raise RuntimeError("connection refused")
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def test_qwen_oracle_survives_request_exceptions():
    qs = load_query_set(GREEN_CITRUS)
    oracle = QwenOracle(Config(), client=_ExplodingClient())
    answers = oracle.answer_batch(Image.new("RGB", (32, 32)), qs)
    assert answers.tolist() == [0] * len(qs.queries)    # all-zeros fallback, no raise


# ----------------------- 7. sam3_phrase round-trip -----------------------

def _write_query_set(tmp_path, queries):
    data = {
        "concept": "green citrus",
        "classes": ["target", "distractor", "spurious"],
        "class_names": {"target": "fruit", "distractor": "leaf", "spurious": "clutter"},
        "epsilon": 0.15,
        "queries": queries,
    }
    path = tmp_path / "qs.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_sam3_phrase_survives_loading_and_reaches_the_oracle(tmp_path):
    path = _write_query_set(tmp_path, [
        {"id": "q01", "text": "Is there a stem?", "sam3_phrase": "stem",
         "templates": {"target": 1, "distractor": -1, "spurious": 0}},
        {"id": "q02", "text": "Is it round?",
         "templates": {"target": 1, "distractor": -1, "spurious": 0}},
    ])
    qs = load_query_set(path)
    assert qs.queries[0].sam3_phrase == "stem"
    assert qs.queries[1].sam3_phrase is None

    oracle = Sam3Oracle(processor=None, tau=0.5)
    oracle._presence_score = lambda crop, phrase: 0.9   # stub the SAM3 call
    answers = oracle.answer_batch(Image.new("RGB", (32, 32)), qs)
    assert answers.tolist() == [1, 0]                   # phrase query answered, other skipped


def test_invalid_sam3_phrase_rejected(tmp_path):
    path = _write_query_set(tmp_path, [
        {"id": "q01", "text": "t?", "sam3_phrase": "",
         "templates": {"target": 1, "distractor": -1, "spurious": 0}},
    ])
    with pytest.raises(ValueError, match="sam3_phrase"):
        load_query_set(path)

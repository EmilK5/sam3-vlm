"""
Tests for V-IP answer routing (step 8.4): the per-query `route` field, the
classical-CV answer channel (cv_answers), and RouterOracle's channel dispatch.

CPU-only, no network, no model weights: the VLM channel is a fake that records
which queries it received, and the SAM3 presence score is stubbed.
"""

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from config import Config
from verifier.queries import QuerySet, Query, load_query_set
from verifier import cv_answers
from verifier.oracle import RouterOracle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")

CLASSES = ["target", "distractor", "spurious"]
_TMPL = {"target": 1, "distractor": -1, "spurious": 0}


def _qs(queries):
    return QuerySet(concept="t", classes=CLASSES, epsilon=0.15, queries=queries,
                    class_names={"target": "fruit", "distractor": "leaf", "spurious": "clutter"})


def _green_disc(size=256):
    """A dark frame with a large filled green circle -> high green-hue fraction +
    high circularity, so cv answers it yes-green / yes-round with no oracle."""
    img = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.ellipse([size * 0.1, size * 0.1, size * 0.9, size * 0.9], fill=(30, 180, 40))
    return img


class _FakeVLM:
    """Records the query ids it was asked and returns +1 for each."""
    def __init__(self):
        self.calls = 0
        self.seen_ids = None

    def answer_batch(self, crop_pil, query_set):
        self.calls += 1
        self.seen_ids = [q.id for q in query_set.queries]
        return np.ones(len(query_set.queries), dtype=np.int8)


# ----------------------- cv_answers unit -----------------------

def test_cv_answers_green_disc_is_green_and_round():
    disc = _green_disc()
    green = {"feature": "hue_fraction", "hue_lo": 35, "hue_hi": 85, "yes_above": 0.3, "no_below": 0.1}
    round_ = {"feature": "circularity", "yes_above": 0.7, "no_below": 0.4}
    assert cv_answers.answer(disc, green) == 1
    assert cv_answers.answer(disc, round_) == 1


def test_cv_answers_direction_low_flips_the_sense():
    disc = _green_disc()  # high circularity
    low = {"feature": "circularity", "direction": "low", "yes_above": 0.7, "no_below": 0.4}
    # a round disc is NOT leaf-like -> direction=low yields -1 (no).
    assert cv_answers.answer(disc, low) == -1


def test_validate_cv_check_rejects_bad_specs():
    with pytest.raises(ValueError):
        cv_answers.validate_cv_check({"feature": "nope", "yes_above": 0.5, "no_below": 0.1})
    with pytest.raises(ValueError):
        cv_answers.validate_cv_check({"feature": "circularity", "yes_above": 0.1, "no_below": 0.5})  # no_below>yes_above
    with pytest.raises(ValueError):
        cv_answers.validate_cv_check({"feature": "hue_fraction", "hue_lo": 200, "hue_hi": 10,
                                      "yes_above": 0.5, "no_below": 0.1})  # hue out of [0,179]


# ----------------------- query-set routing validation -----------------------

def test_route_default_is_vlm():
    qs = _qs([Query(id="q1", text="t", templates=_TMPL)])
    assert qs.queries[0].route == "vlm"


def test_checked_in_green_citrus_partitions_routes():
    qs = load_query_set(GREEN_CITRUS)
    routes = {q.id: q.route for q in qs.queries}
    assert routes["q01"] == "cv" and qs.queries[0].cv_check["feature"] == "circularity"
    assert routes["q06"] == "cv" and qs.queries[5].cv_check["feature"] == "edge_density"
    assert routes["q02"] == "vlm" and routes["q05"] == "vlm"
    # every cv-routed query carries a validated cv_check; vlm ones carry none.
    for q in qs.queries:
        assert (q.cv_check is not None) == (q.route == "cv")


def test_cv_route_requires_cv_check(tmp_path):
    _assert_bad_query(tmp_path, {"id": "q1", "text": "t", "templates": _TMPL, "route": "cv"})


def test_sam3_route_requires_phrase(tmp_path):
    _assert_bad_query(tmp_path, {"id": "q1", "text": "t", "templates": _TMPL, "route": "sam3"})


def test_bad_route_value_rejected(tmp_path):
    _assert_bad_query(tmp_path, {"id": "q1", "text": "t", "templates": _TMPL, "route": "psychic"})


def _assert_bad_query(tmp_path, query):
    import json
    data = {"concept": "t", "classes": CLASSES,
            "class_names": {"target": "fruit", "distractor": "leaf", "spurious": "clutter"},
            "epsilon": 0.15, "queries": [query]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_query_set(str(path))


# ----------------------- RouterOracle dispatch -----------------------

def _mixed_set():
    """One query per channel: cv (green), sam3 (stem), vlm (semantic)."""
    return _qs([
        Query(id="qcv", text="green?", templates=_TMPL, route="cv",
              cv_check={"feature": "hue_fraction", "hue_lo": 35, "hue_hi": 85,
                        "yes_above": 0.3, "no_below": 0.1}),
        Query(id="qsam", text="stem?", templates=_TMPL, route="sam3", sam3_phrase="stem"),
        Query(id="qvlm", text="is it a fruit?", templates=_TMPL, route="vlm"),
    ])


def test_router_sends_only_vlm_queries_to_the_vlm_oracle():
    fake = _FakeVLM()
    router = RouterOracle(Config(), processor=None, vlm_oracle=fake)
    out = router.answer_batch(_green_disc(), _mixed_set())

    assert fake.seen_ids == ["qvlm"]            # ONLY the vlm-routed query reached Qwen
    assert fake.calls == 1 and router.n_vlm_calls == 1
    assert out[0] == 1                          # cv answered the green disc (yes)
    assert out[1] == 0                          # sam3 has no processor -> unsure
    assert out[2] == 1                          # vlm fake returned +1


def test_router_cv_only_makes_no_model_calls():
    qs = _qs([
        Query(id="qg", text="green?", templates=_TMPL, route="cv",
              cv_check={"feature": "hue_fraction", "hue_lo": 35, "hue_hi": 85,
                        "yes_above": 0.3, "no_below": 0.1}),
        Query(id="qr", text="round?", templates=_TMPL, route="cv",
              cv_check={"feature": "circularity", "yes_above": 0.7, "no_below": 0.4}),
    ])
    router = RouterOracle(Config(), processor=None, vlm_oracle=None)
    out = router.answer_batch(_green_disc(), qs)
    assert out.tolist() == [1, 1]               # yes-green, yes-round -- no oracle
    assert router.n_vlm_calls == 0 and router.n_sam_calls == 0


def test_router_missing_vlm_oracle_yields_zeros_not_crash():
    router = RouterOracle(Config(), processor=None, vlm_oracle=None)
    out = router.answer_batch(_green_disc(), _mixed_set())
    assert out[2] == 0                          # vlm query unanswered -> unsure, no raise
    assert router.n_vlm_calls == 0


def test_router_counts_sam3_calls_with_a_stubbed_processor():
    router = RouterOracle(Config(), processor=object(), vlm_oracle=None)
    router._sam3._presence_score = lambda crop, phrase: 0.9   # stub: above tau -> yes
    out = router.answer_batch(_green_disc(), _mixed_set())
    assert out[1] == 1                          # sam3 answered the stem query
    assert router.n_sam_calls == 1

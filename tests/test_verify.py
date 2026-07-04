import dataclasses
import os

import numpy as np
from PIL import Image

from config import Config
from verifier.queries import load_query_set
from verifier.oracle import MockOracle
from verifier.verify import extract_crop, verify_candidate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


def _image(h=64, w=64):
    # deterministic non-uniform RGB content
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ----------------------- extract_crop -----------------------

def test_extract_crop_returns_pil_of_requested_size():
    img = _image()
    crop = extract_crop(img, [10, 10, 30, 30], scale=1.4, size=256)
    assert isinstance(crop, Image.Image)
    assert crop.size == (256, 256)  # PIL size is (width, height)


def test_extract_crop_clamps_box_outside_image():
    img = _image(64, 64)
    # box straddling the top-left corner; must not raise and must be sized
    crop = extract_crop(img, [-20, -20, 10, 10], scale=1.4, size=128)
    assert crop.size == (128, 128)


def test_extract_crop_handles_degenerate_box():
    img = _image(64, 64)
    crop = extract_crop(img, [30, 30, 30, 30], scale=1.0, size=64)  # zero area
    assert crop.size == (64, 64)


# ----------------------- verify_candidate -----------------------

def test_verify_candidate_batched_recovers_class():
    img = _image()
    qs = load_query_set(GREEN_CITRUS)
    cfg = Config()  # answer_mode="batched", vip_stop=0.10, vip_max_queries=10
    for cls in qs.classes:
        result = verify_candidate(img, [10, 10, 40, 40], MockOracle(cls), qs, cfg)
        assert result["verdict"] == cls
        assert result["n_oracle_calls"] == 1
        assert len(result["posterior"]) == len(qs.classes)
        assert 0.0 <= result["p_target"] <= 1.0
        # chain entries are interpretable
        for step in result["chain"]:
            assert set(step.keys()) == {"q", "a"}
            assert step["a"] in ("yes", "no", "unsure")


def test_verify_candidate_target_has_high_p_target():
    img = _image()
    qs = load_query_set(GREEN_CITRUS)
    result = verify_candidate(img, [5, 5, 35, 35], MockOracle("target"), qs, Config())
    assert result["verdict"] == "target"
    assert result["p_target"] > 0.9


def test_sequential_mode_calls_oracle_per_selected_query():
    img = _image()
    qs = load_query_set(GREEN_CITRUS)
    cfg = dataclasses.replace(Config(), answer_mode="sequential")
    result = verify_candidate(img, [10, 10, 40, 40], MockOracle("distractor"), qs, cfg)
    assert result["verdict"] == "distractor"
    # one oracle call per query in the chain (not a single batched call)
    assert result["n_oracle_calls"] == len(result["chain"])
    assert result["n_oracle_calls"] >= 1


def test_batched_and_sequential_agree_on_verdict():
    img = _image()
    qs = load_query_set(GREEN_CITRUS)
    batched = verify_candidate(img, [0, 0, 20, 20], MockOracle("spurious"), qs, Config())
    seq = verify_candidate(img, [0, 0, 20, 20], MockOracle("spurious"), qs,
                           dataclasses.replace(Config(), answer_mode="sequential"))
    assert batched["verdict"] == seq["verdict"] == "spurious"
    assert np.allclose(batched["posterior"], seq["posterior"])

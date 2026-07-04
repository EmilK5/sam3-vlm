import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

from config import Config
from verifier.queries import QuerySet, Query, load_query_set
from verifier.oracle import MockOracle, QwenOracle, Sam3Oracle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


def _crop():
    return Image.new("RGB", (8, 8), (0, 128, 0))


def _toy_query_set():
    return QuerySet(
        concept="toy",
        classes=["target", "distractor", "spurious"],
        epsilon=0.15,
        queries=[
            Query("q01", "round?", {"target": 1, "distractor": -1, "spurious": 0}),
            Query("q02", "veins?", {"target": -1, "distractor": 1, "spurious": 0}),
            Query("q03", "sky?", {"target": -1, "distractor": -1, "spurious": 1}),
        ],
    )


# ----------------------- MockOracle -----------------------

def test_mock_oracle_noiseless_returns_templates():
    qs = load_query_set(GREEN_CITRUS)
    oracle = MockOracle("target", noise=0.0)
    ans = oracle.answer_batch(_crop(), qs)
    expected = np.array([q.templates["target"] for q in qs.queries], dtype=np.int8)
    assert ans.dtype == np.int8
    assert np.array_equal(ans, expected)


def test_mock_oracle_is_deterministic_for_fixed_seed():
    qs = load_query_set(GREEN_CITRUS)
    a = MockOracle("distractor", noise=0.4, seed=7).answer_batch(_crop(), qs)
    b = MockOracle("distractor", noise=0.4, seed=7).answer_batch(_crop(), qs)
    assert np.array_equal(a, b)
    # values always stay in the valid answer alphabet
    assert set(np.unique(a)).issubset({-1, 0, 1})


def test_mock_oracle_noise_changes_some_answers():
    qs = load_query_set(GREEN_CITRUS)
    clean = MockOracle("target", noise=0.0).answer_batch(_crop(), qs)
    noisy = MockOracle("target", noise=1.0, seed=1).answer_batch(_crop(), qs)
    assert not np.array_equal(clean, noisy)  # noise=1.0 flips every query


# ----------------------- QwenOracle -----------------------

class _FakeResponse:
    def __init__(self, content):
        message = types.SimpleNamespace(content=content)
        self.choices = [types.SimpleNamespace(message=message)]


class _FakeClient:
    """Mimics the openai client surface: client.chat.completions.create(...)."""
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        content = self._contents[min(self.calls, len(self._contents) - 1)]
        self.calls += 1
        return _FakeResponse(content)


def test_qwen_oracle_parses_canned_json():
    qs = _toy_query_set()
    content = '{"answers": {"q01": "yes", "q02": "no", "q03": "unsure"}}'
    oracle = QwenOracle(Config(), client=_FakeClient([content]))
    ans = oracle.answer_batch(_crop(), qs)
    assert ans.dtype == np.int8
    assert np.array_equal(ans, np.array([1, -1, 0], dtype=np.int8))


def test_qwen_oracle_missing_ids_map_to_zero():
    qs = _toy_query_set()
    content = '{"answers": {"q01": "yes"}}'  # q02, q03 omitted
    oracle = QwenOracle(Config(), client=_FakeClient([content]))
    ans = oracle.answer_batch(_crop(), qs)
    assert np.array_equal(ans, np.array([1, 0, 0], dtype=np.int8))


def test_qwen_oracle_retries_then_falls_back_to_zeros():
    qs = _toy_query_set()
    cfg = Config()  # oracle_max_retries defaults to 2 -> 3 attempts total
    client = _FakeClient(["not json at all"])
    oracle = QwenOracle(cfg, client=client)
    ans = oracle.answer_batch(_crop(), qs)
    assert np.array_equal(ans, np.zeros(3, dtype=np.int8))
    assert client.calls == cfg.oracle_max_retries + 1


def test_qwen_oracle_parse_helper_rejects_malformed():
    qs = _toy_query_set()
    oracle = QwenOracle(Config())
    assert oracle._parse_answers("garbage", qs) is None
    assert oracle._parse_answers('{"answers": [1, 2, 3]}', qs) is None  # not a dict


# ----------------------- Sam3Oracle -----------------------

def test_sam3_oracle_returns_zeros_and_calls_no_model_without_phrases():
    qs = load_query_set(GREEN_CITRUS)  # no query carries a sam3_phrase

    class ExplodingProcessor:
        def __call__(self, *a, **k):
            raise AssertionError("processor must not be called when no sam3_phrase exists")

    ans = Sam3Oracle(ExplodingProcessor(), tau=0.5).answer_batch(_crop(), qs)
    assert np.array_equal(ans, np.zeros(len(qs.queries), dtype=np.int8))


@pytest.fixture
def stub_torch():
    saved = sys.modules.get("torch")
    torch_stub = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch_stub.no_grad = _NoGrad
    sys.modules["torch"] = torch_stub
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


class _FakeInputs(dict):
    """A mapping (so model(**inputs) works) that also supports .to(device)."""
    def to(self, device):
        return self


class _FakeModel:
    device = "cpu"

    def __call__(self, **kwargs):
        return types.SimpleNamespace()


class _FakeProcessor:
    """Minimal stand-in mirroring the SAM3 presence call surface used by Sam3Oracle."""
    def __init__(self, score):
        self._score = score
        self.model = _FakeModel()

    def __call__(self, images=None, text=None, return_tensors=None):
        original_sizes = types.SimpleNamespace(tolist=lambda: [[8, 8]])
        return _FakeInputs(original_sizes=original_sizes)

    def post_process_instance_segmentation(self, outputs, threshold, target_sizes):
        return [{"scores": [self._score]}]


def _phrase_query_set():
    q = Query("q01", "is a stem visible?", {"target": 1, "distractor": 0, "spurious": -1})
    q.sam3_phrase = "stem"  # Query is a mutable dataclass; attach an optional field
    return QuerySet(concept="c", classes=["target", "distractor", "spurious"],
                    epsilon=0.15, queries=[q])


def test_sam3_oracle_thresholds_presence_above_tau(stub_torch):
    ans = Sam3Oracle(_FakeProcessor(score=0.9), tau=0.5).answer_batch(_crop(), _phrase_query_set())
    assert ans[0] == 1


def test_sam3_oracle_thresholds_presence_below_tau(stub_torch):
    ans = Sam3Oracle(_FakeProcessor(score=0.1), tau=0.5).answer_batch(_crop(), _phrase_query_set())
    assert ans[0] == -1

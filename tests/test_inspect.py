"""
Tests for agent/inspect.py: should_inspect protocol logic and inspect_scene JSON
parse/fallback against a canned fake client (no network).
"""

import types

import pytest
from PIL import Image

from config import Config
from agent.inspect import inspect_scene, should_inspect, _parse_z


def _z(**over):
    base = {"target_present": True, "density": "medium", "object_scale": "medium",
            "occlusion": "medium", "recommend": "query", "notes": "ok"}
    base.update(over)
    return base


# ----------------------- should_inspect -----------------------

def test_inspect_at_first_step():
    assert should_inspect(t=1, phi={"D": []}, last_z=None) is True


def test_inspect_when_last_recommend_is_tile_or_subdivide():
    assert should_inspect(2, {"D": [3]}, _z(recommend="tile")) is True
    assert should_inspect(2, {"D": [3]}, _z(recommend="subdivide")) is True


def test_no_inspect_when_querying_and_not_saturated():
    assert should_inspect(2, {"D": [3, 2, 1]}, _z(recommend="query")) is False


def test_inspect_when_saturated_and_dense():
    phi = {"D": [4, 0, 0, 0]}  # last 3 passes found nothing -> saturated
    assert should_inspect(3, phi, _z(density="dense", recommend="query")) is True


def test_no_inspect_when_saturated_but_sparse():
    phi = {"D": [4, 0, 0, 0]}
    assert should_inspect(3, phi, _z(density="sparse", recommend="query")) is False


def test_no_inspect_when_no_prior_z_and_later_step():
    assert should_inspect(2, {"D": [0, 0, 0]}, None) is False


# ----------------------- _parse_z validation -----------------------

def test_parse_z_accepts_valid_and_truncates_notes():
    content = ('{"target_present": true, "density": "dense", "object_scale": "small", '
               '"occlusion": "high", "recommend": "tile", "notes": "' + "x" * 300 + '"}')
    z = _parse_z(content)
    assert z["density"] == "dense" and z["recommend"] == "tile"
    assert len(z["notes"]) == 200


@pytest.mark.parametrize("content", [
    "not json",
    "[1, 2, 3]",
    '{"density": "dense"}',                                  # missing keys
    '{"target_present": "yes", "density": "dense", "object_scale": "small", "occlusion": "high", "recommend": "tile"}',  # target_present not bool
    '{"target_present": true, "density": "huge", "object_scale": "small", "occlusion": "high", "recommend": "tile"}',    # bad density enum
    '{"target_present": true, "density": "dense", "object_scale": "small", "occlusion": "high", "recommend": "fly"}',    # bad recommend enum
])
def test_parse_z_rejects_invalid(content):
    assert _parse_z(content) is None


# ----------------------- inspect_scene (canned fake client) -----------------------

class _FakeResponse:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class _FakeClient:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = self._contents[min(self.calls, len(self._contents) - 1)]
        self.calls += 1
        return _FakeResponse(content)


def _img():
    return Image.new("RGB", (8, 8))


def test_inspect_scene_parses_valid_response():
    content = ('{"target_present": true, "density": "dense", "object_scale": "small", '
               '"occlusion": "medium", "recommend": "tile", "notes": "crowded canopy"}')
    z = inspect_scene(_img(), {"K": 0}, Config(), client=_FakeClient([content]))
    assert z["density"] == "dense" and z["recommend"] == "tile"


def test_inspect_scene_falls_back_to_neutral_z():
    cfg = Config()  # oracle_max_retries=2 -> 3 attempts
    client = _FakeClient(["garbage not json"])
    z = inspect_scene(_img(), {"K": 0}, cfg, client=client)
    assert z["recommend"] == "query"
    assert z["notes"] == "fallback"
    assert client.calls == cfg.oracle_max_retries + 1

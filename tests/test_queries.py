import copy
import json
import os

import pytest

from verifier.queries import load_query_set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


def test_load_checked_in_green_citrus():
    qs = load_query_set(GREEN_CITRUS)
    assert qs.concept == "green citrus"
    assert qs.classes == ["target", "distractor", "spurious"]
    assert qs.class_names["target"] == "fruit"
    assert 0.0 < qs.epsilon < 0.5
    assert len(qs.queries) >= 5   # trimmed to a small, general set for green citrus
    # every query covers all classes with values in {-1, 0, 1}
    for q in qs.queries:
        assert set(q.templates) == set(qs.classes)
        assert all(v in (-1, 0, 1) for v in q.templates.values())
    # ids are unique
    ids = [q.id for q in qs.queries]
    assert len(ids) == len(set(ids))


def _load_raw():
    with open(GREEN_CITRUS, "r") as f:
        return json.load(f)


def _write(tmp_path, data):
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_bad_template_value_fails(tmp_path):
    data = _load_raw()
    data["queries"][0]["templates"]["target"] = 2  # illegal (not in {-1,0,1})
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_missing_class_in_templates_fails(tmp_path):
    data = _load_raw()
    del data["queries"][0]["templates"]["spurious"]  # not all classes present
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_duplicate_id_fails(tmp_path):
    data = _load_raw()
    data["queries"][1]["id"] = data["queries"][0]["id"]  # duplicate id
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_epsilon_out_of_range_fails(tmp_path):
    data = _load_raw()
    data["epsilon"] = 0.8  # outside (0, 0.5)
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_missing_top_level_key_fails(tmp_path):
    data = _load_raw()
    del data["classes"]
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_extra_unexpected_class_in_templates_fails(tmp_path):
    data = _load_raw()
    data["queries"][0]["templates"]["bogus"] = 1  # class not in `classes`
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_empty_queries_list_fails(tmp_path):
    data = _load_raw()
    data["queries"] = []
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_non_dict_templates_fails(tmp_path):
    data = _load_raw()
    data["queries"][0]["templates"] = [1, -1, 0]  # list, not dict
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_class_names_missing_entry_fails(tmp_path):
    data = _load_raw()
    del data["class_names"]["spurious"]  # class present but no human-readable name
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


@pytest.mark.parametrize("bad_epsilon", [0.0, 0.5, -0.1, 1.0])
def test_epsilon_boundaries_fail(tmp_path, bad_epsilon):
    data = _load_raw()
    data["epsilon"] = bad_epsilon
    with pytest.raises(ValueError):
        load_query_set(_write(tmp_path, data))


def test_all_queries_have_at_least_one_informative_class():
    # Not required by the schema, but a sanity check on the checked-in file:
    # an all-zero-template query would be dead weight (never selected by V-IP).
    qs = load_query_set(GREEN_CITRUS)
    for q in qs.queries:
        assert set(q.templates.values()) != {0}, f"{q.id} is uninformative for every class"

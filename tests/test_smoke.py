import dataclasses

import pytest

from config import Config


def test_config_instantiates_with_defaults():
    cfg = Config()
    assert cfg.conf == 0.35
    assert cfg.nms_mode == "dualgate"
    assert cfg.verifier_mode == "ioc"
    assert isinstance(cfg.lambdas, dict)


def test_config_load_overrides(tmp_path):
    override_path = tmp_path / "override.json"
    override_path.write_text('{"conf": 0.5, "verifier_mode": "vip"}')

    cfg = Config.load(str(override_path))

    assert cfg.conf == 0.5
    assert cfg.verifier_mode == "vip"
    assert cfg.nms_mode == "dualgate"


def test_cost_and_agent_fields_present():
    cfg = Config()
    # costs are normalized relative to one global SAM3 call
    assert cfg.c_sam == 1.0
    for field in ("c_tile", "c_verify", "c_inspect", "c_orch"):
        assert 0.0 < getattr(cfg, field) < 1.0
    # agent knobs
    assert cfg.window_m == 3
    assert cfg.budget_max_actions == 12
    assert "lambda_D" in cfg.lambdas and "lambda_S" in cfg.lambdas


def test_verifier_defaults_keep_ioc_as_the_safe_default():
    # Hard constraint: the old IoC path must stay the default; vip is opt-in.
    cfg = Config()
    assert cfg.verifier_mode == "ioc"
    assert cfg.answer_mode == "batched"
    # vip_epsilon: None defers to the query set's epsilon (a float overrides it).
    assert cfg.vip_epsilon is None
    # box overlap stays the default; mask IoU/IoM is opt-in.
    assert cfg.overlap_mode == "box"


def test_policy_tunables_live_in_config():
    # These were getattr defaults scattered across belief/policies; they must
    # live in config (CLAUDE.md: no magic numbers outside config.py).
    cfg = Config()
    assert cfg.target_prompt == "green fruit"
    # (c0 / small_area were removed in v2 step 8.5 -- only the deleted tile-menu
    # heuristic used them.)
    assert cfg.tau_w == 0.5
    assert cfg.k_min == 2
    assert cfg.tau_high == 0.5
    # inert defaults: the lambda_A support term stays a no-op until narrowed
    assert cfg.area_min == 0.0 and cfg.area_max == float("inf")


def test_oracle_config_reads_env(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "http://example:8000/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen3-vl-test")
    cfg = Config()
    assert cfg.oracle_base_url == "http://example:8000/v1"
    assert cfg.oracle_model_name == "qwen3-vl-test"


def test_api_key_is_never_a_config_field():
    # QWEN_API_KEY must never be stored on Config (never logged/printed).
    field_names = {f.name for f in dataclasses.fields(Config)}
    assert not any("api_key" in name.lower() or "apikey" in name.lower() for name in field_names)


def test_load_rejects_unknown_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not_a_field": 1}')
    with pytest.raises(TypeError):
        Config.load(str(bad))

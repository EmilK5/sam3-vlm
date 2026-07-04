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

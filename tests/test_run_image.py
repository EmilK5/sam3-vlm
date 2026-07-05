"""
Tests for the pure helpers in scripts/run_image.py (step 0.2).

The module imports torch/inference/pipeline at import time, none of which are
installed in the CPU-only test environment. We stub those heavy modules in
sys.modules so the pure argparse / path logic can be exercised without any
model machinery. No network, no weights.
"""

import os
import subprocess
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def run_image_module():
    saved = {name: sys.modules.get(name) for name in ("torch", "inference", "pipeline")}
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda spec: spec
    sys.modules["torch"] = torch_stub
    sys.modules["inference"] = types.ModuleType("inference")
    sys.modules["pipeline"] = types.ModuleType("pipeline")
    sys.modules.pop("scripts.run_image", None)

    import scripts.run_image as ri

    try:
        yield ri
    finally:
        sys.modules.pop("scripts.run_image", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_build_output_paths(run_image_module):
    overlay, graph_json = run_image_module.build_output_paths("/data/img_042.png", "out")
    assert overlay == "out/img_042_overlay.jpg"
    assert graph_json == "out/img_042_graph.json"


def test_build_output_paths_with_mode_suffix(run_image_module):
    overlay, graph_json = run_image_module.build_output_paths("/data/img_042.png", "out", suffix="vip")
    assert overlay == "out/img_042_vip_overlay.jpg"
    assert graph_json == "out/img_042_vip_graph.json"


def test_parse_args_defaults_and_flags(run_image_module):
    ns = run_image_module.parse_args(
        ["--image", "a.jpg", "--prompt", "green fruit", "--passes", "3", "--tiling"]
    )
    assert ns.image == "a.jpg"
    assert ns.prompt == "green fruit"
    assert ns.passes == 3
    assert ns.tiling is True
    assert ns.clahe is False
    assert ns.conf == 0.35  # default pulled from Config().conf
    assert ns.verifier == "ioc"  # default keeps the old behavior
    assert ns.draw_gt is False


def test_parse_args_draw_gt_flag(run_image_module):
    ns = run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x", "--draw-gt"])
    assert ns.draw_gt is True


def test_find_label_file_images_split_layout(run_image_module, tmp_path):
    # dataset/images/train/img.png -> dataset/labels/train/img.txt
    img = tmp_path / "images" / "train"
    lbl = tmp_path / "labels" / "train"
    img.mkdir(parents=True)
    lbl.mkdir(parents=True)
    (img / "img.png").write_bytes(b"")
    label = lbl / "img.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")

    found = run_image_module.find_label_file(str(img / "img.png"))
    assert found is not None and os.path.abspath(found) == os.path.abspath(str(label))


def test_find_label_file_missing_returns_none(run_image_module, tmp_path):
    img = tmp_path / "images" / "train"
    img.mkdir(parents=True)
    (img / "img.png").write_bytes(b"")
    assert run_image_module.find_label_file(str(img / "img.png")) is None


def test_parse_args_verifier_choices(run_image_module):
    for mode in ("ioc", "vip", "off"):
        ns = run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x", "--verifier", mode])
        assert ns.verifier == mode
    with pytest.raises(SystemExit):
        run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x", "--verifier", "bogus"])


def test_parse_args_overlap_mode_defaults_to_box(run_image_module):
    ns = run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x"])
    assert ns.overlap_mode == "box"
    ns = run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x", "--overlap-mode", "mask"])
    assert ns.overlap_mode == "mask"
    with pytest.raises(SystemExit):
        run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x", "--overlap-mode", "bogus"])


def test_build_verifier_threads_overlap_mode(run_image_module):
    ns = run_image_module.parse_args(
        ["--image", "a.jpg", "--prompt", "x", "--verifier", "off", "--overlap-mode", "mask"])
    cfg, _, _ = run_image_module.build_verifier(ns)
    assert cfg.overlap_mode == "mask"


def test_build_verifier_off_and_ioc_have_no_oracle(run_image_module):
    for mode in ("ioc", "off"):
        ns = run_image_module.parse_args(["--image", "a.jpg", "--prompt", "x", "--verifier", mode])
        cfg, oracle, query_set = run_image_module.build_verifier(ns)
        assert cfg.verifier_mode == mode
        assert oracle is None and query_set is None


def test_build_verifier_vip_mock_builds_oracle_and_queries(run_image_module):
    query_file = os.path.join(REPO_ROOT, "queries", "green_citrus.json")
    ns = run_image_module.parse_args(
        ["--image", "a.jpg", "--prompt", "x", "--verifier", "vip",
         "--oracle", "mock", "--query-file", query_file]
    )
    cfg, oracle, query_set = run_image_module.build_verifier(ns)
    assert cfg.verifier_mode == "vip"
    assert oracle is not None
    assert query_set is not None and len(query_set.queries) > 0


def test_parse_args_requires_image_and_prompt(run_image_module):
    with pytest.raises(SystemExit):
        run_image_module.parse_args(["--prompt", "x"])


def test_script_invocation_can_import_config():
    """Regression: `python scripts/run_image.py` must add the repo root to
    sys.path BEFORE importing config. Run from a foreign cwd so only the
    script's own directory is auto-added. The run may still fail later on the
    missing `torch`, but it must NOT fail with 'No module named config'."""
    script = os.path.join(REPO_ROOT, "scripts", "run_image.py")
    proc = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True, text=True, cwd=os.path.dirname(REPO_ROOT),
    )
    combined = proc.stdout + proc.stderr
    assert "No module named 'config'" not in combined

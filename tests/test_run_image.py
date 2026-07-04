"""
Tests for the pure helpers in scripts/run_image.py (step 0.2).

The module imports torch/inference/pipeline at import time, none of which are
installed in the CPU-only test environment. We stub those heavy modules in
sys.modules so the pure argparse / path logic can be exercised without any
model machinery. No network, no weights.
"""

import sys
import types

import pytest


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


def test_parse_args_requires_image_and_prompt(run_image_module):
    with pytest.raises(SystemExit):
        run_image_module.parse_args(["--prompt", "x"])

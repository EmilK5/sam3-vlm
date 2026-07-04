"""
Tests for the verifier="vip" integration in pipeline.register_and_verify_candidates.

pipeline.py imports `inference` (which imports torch) at module top, so we stub
torch/inference in sys.modules before importing pipeline. register_and_verify_
candidates does not use `inference`, so the stub is safe; the VIP path only
touches verify/vip/oracle (all torch-free).
"""

import dataclasses
import os
import sys
import types

import numpy as np
import pytest

from graph import OrchardGraph
from config import Config
from verifier.queries import load_query_set
from verifier.oracle import MockOracle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN_CITRUS = os.path.join(REPO_ROOT, "queries", "green_citrus.json")


@pytest.fixture
def pipeline_module():
    saved = {name: sys.modules.get(name) for name in ("torch", "inference", "pipeline")}
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["inference"] = types.ModuleType("inference")
    sys.modules.pop("pipeline", None)

    import pipeline

    try:
        yield pipeline
    finally:
        sys.modules.pop("pipeline", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _vip_cfg():
    return dataclasses.replace(Config(), verifier_mode="vip")


def _image():
    return np.zeros((256, 256, 3), dtype=np.uint8)


# ----------------------- VIP path -----------------------

def test_vip_classifies_target_as_fruit(pipeline_module):
    graph = OrchardGraph()
    qs = load_query_set(GREEN_CITRUS)
    boxes = np.array([[10, 10, 60, 60], [80, 80, 140, 140]], dtype=float)
    scores = np.array([0.9, 0.8])

    added, dup = pipeline_module.register_and_verify_candidates(
        boxes, scores, leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=1,
        cfg=_vip_cfg(), oracle=MockOracle("target"), query_set=qs, image_np=_image(),
    )

    assert (added, dup) == (2, 0)
    nodes = list(graph.nodes.values())
    assert all(n.classification == "fruit" for n in nodes)
    for n in nodes:
        assert n.vip_chain is not None
        assert n.vip_posterior is not None
        assert n.scores["fruit_verification"] > 0.9
    # to_dict surfaces the vip trace when present
    d = graph.to_dict()["nodes"][0]
    assert "vip_chain" in d and "vip_posterior" in d


def test_vip_distractor_maps_to_leaf(pipeline_module):
    graph = OrchardGraph()
    qs = load_query_set(GREEN_CITRUS)
    added, _ = pipeline_module.register_and_verify_candidates(
        np.array([[10, 10, 60, 60]], dtype=float), np.array([0.7]),
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=1,
        cfg=_vip_cfg(), oracle=MockOracle("distractor"), query_set=qs, image_np=_image(),
    )
    assert added == 1
    assert list(graph.nodes.values())[0].classification == "leaf"


def test_vip_spurious_maps_to_spurious(pipeline_module):
    graph = OrchardGraph()
    qs = load_query_set(GREEN_CITRUS)
    pipeline_module.register_and_verify_candidates(
        np.array([[10, 10, 60, 60]], dtype=float), np.array([0.7]),
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=1,
        cfg=_vip_cfg(), oracle=MockOracle("spurious"), query_set=qs, image_np=_image(),
    )
    assert list(graph.nodes.values())[0].classification == "spurious"


def test_vip_skips_tiny_boxes(pipeline_module):
    graph = OrchardGraph()
    qs = load_query_set(GREEN_CITRUS)
    added, _ = pipeline_module.register_and_verify_candidates(
        np.array([[10, 10, 18, 18]], dtype=float), np.array([0.9]),  # 8px a side
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=1,
        cfg=_vip_cfg(), oracle=MockOracle("target"), query_set=qs, image_np=_image(),
    )
    assert added == 1
    node = list(graph.nodes.values())[0]
    assert node.classification == "unresolved"
    assert node.vip_chain is None  # never verified


def test_vip_requires_oracle_and_query_set(pipeline_module):
    graph = OrchardGraph()
    with pytest.raises(ValueError):
        pipeline_module.register_and_verify_candidates(
            np.array([[10, 10, 60, 60]], dtype=float), np.array([0.9]),
            leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=1,
            cfg=_vip_cfg(), oracle=None, query_set=None, image_np=None,
        )


# ----------------------- verifier off -----------------------

def test_verifier_off_registers_without_classifying(pipeline_module):
    graph = OrchardGraph()
    cfg = dataclasses.replace(Config(), verifier_mode="off")
    boxes = np.array([[10, 10, 60, 60], [80, 80, 140, 140]], dtype=float)
    scores = np.array([0.9, 0.2])

    # leaf_boxes present + no oracle: proves neither the IoC gate nor VIP runs.
    added, dup = pipeline_module.register_and_verify_candidates(
        boxes, scores, leaf_boxes=np.array([[10, 10, 60, 60]], dtype=float),
        graph=graph, pass_number=3, cfg=cfg,
    )

    assert (added, dup) == (2, 0)
    for n in graph.nodes.values():
        assert n.classification == "unresolved"  # untouched by any verifier
        assert n.vip_chain is None
        # no verification scores written
        assert n.scores["fruit_verification"] == 0.0
        assert n.scores["leaf_verification"] == 0.0


# ----------------------- IoC path (golden, must stay byte-for-byte) -----------------------

def _run_ioc(pipeline_module, cfg):
    """Register three boxes through the IoC gate and return their tags."""
    graph = OrchardGraph()
    boxes = np.array([[0, 0, 20, 20], [100, 100, 120, 120], [400, 400, 420, 420]], dtype=float)
    scores = np.array([0.9, 0.5, 0.10])
    leaf_boxes = np.array([[100, 100, 120, 120]], dtype=float)
    added, dup = pipeline_module.register_and_verify_candidates(
        boxes, scores, leaf_boxes, graph=graph, pass_number=1, cfg=cfg,
    )
    return graph, added, dup


def test_ioc_golden_classifications(pipeline_module):
    # cfg=None -> default IoC gate.
    #   box0 score 0.9, no leaf overlap        -> fruit
    #   box1 fully inside a leaf (IoC=1.0)      -> leaf
    #   box2 score 0.10, pass 1, no leaf        -> spurious
    graph, added, dup = _run_ioc(pipeline_module, cfg=None)
    tags = [n.classification for n in graph.nodes.values()]
    assert tags == ["fruit", "leaf", "spurious"]
    assert (added, dup) == (3, 0)
    # no vip trace should be written on the IoC path
    assert "vip_chain" not in graph.to_dict()["nodes"][0]


def test_ioc_explicit_mode_equals_default(pipeline_module):
    graph_none, _, _ = _run_ioc(pipeline_module, cfg=None)
    graph_ioc, _, _ = _run_ioc(pipeline_module, cfg=Config())  # verifier_mode="ioc"
    tags_none = [n.classification for n in graph_none.nodes.values()]
    tags_ioc = [n.classification for n in graph_ioc.nodes.values()]
    assert tags_none == tags_ioc == ["fruit", "leaf", "spurious"]

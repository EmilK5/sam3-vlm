"""
Tests for OrchardNode support/jitter/signatures/area (step 2.1) and the pipeline
dedup branch that reinforces a matched track.

The pipeline integration test stubs torch/inference so pipeline imports on a
CPU-only box (register_and_verify_candidates doesn't touch inference).
"""

import sys
import types

import numpy as np
import pytest

from graph import OrchardNode, OrchardGraph


# ----------------------- OrchardNode.reinforce (unit) -----------------------

def test_new_node_defaults():
    n = OrchardNode([0, 0, 10, 20], score=0.9, found_in_pass=1)
    assert n.support == 1
    assert n.signatures == set()
    assert n.jitter == 0.0
    assert n.area == 200.0  # 10 * 20


def test_reinforce_same_box_keeps_jitter_zero():
    n = OrchardNode([0, 0, 10, 10], 0.9, 1)
    n.reinforce([0, 0, 10, 10], "1:global:green fruit:0.35")   # same center -> d=0
    assert n.support == 2
    assert "1:global:green fruit:0.35" in n.signatures
    assert n.jitter == pytest.approx(0.0)


def test_reinforce_shifted_box_grows_jitter_as_running_mean():
    n = OrchardNode([0, 0, 10, 10], 0.9, 1)     # center (5,5)
    n.reinforce([0, 0, 10, 10], "s1")           # d=0
    n.reinforce([4, 0, 14, 10], "s2")           # center (9,5) -> d=4
    assert n.support == 3
    assert n.jitter == pytest.approx((0 + 4) / 2)  # running mean over 2 re-detections
    assert n.signatures == {"s1", "s2"}


def test_to_dict_exposes_support_fields():
    g = OrchardGraph()
    nid = g.add_candidate([0, 0, 10, 10], 0.9, found_in_pass=1)
    g.nodes[nid].reinforce([0, 0, 10, 10], "sig")
    d = g.to_dict()["nodes"][0]
    assert d["support"] == 2
    assert d["signatures"] == ["sig"]
    assert d["area"] == 100.0
    assert "jitter" in d


# ----------------------- pipeline dedup -> reinforce (integration) -----------------------

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


def _register(pipeline_module, graph, box, pass_number, signature):
    return pipeline_module.register_and_verify_candidates(
        np.array([box], dtype=float), np.array([0.9]),
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=pass_number,
        signature=signature,   # cfg=None -> IoC path, high score -> classified "fruit"
    )


def test_duplicate_detection_reinforces_matched_track(pipeline_module):
    graph = OrchardGraph()
    # pass 1: register (IoC gate, score 0.9, no leaf overlap -> fruit, support 1)
    _register(pipeline_module, graph, [0, 0, 20, 20], 1, "1:global:green fruit:0.35")
    node = list(graph.nodes.values())[0]
    assert node.classification == "fruit"
    assert node.support == 1 and node.signatures == {"1:global:green fruit:0.35"}

    # pass 2: identical box -> duplicate -> reinforce (not a new node)
    added, dup = _register(pipeline_module, graph, [0, 0, 20, 20], 2, "2:global:green fruit:0.35")
    assert (added, dup) == (0, 1)
    assert len(graph.nodes) == 1
    assert node.support == 2
    assert len(node.signatures) == 2
    assert node.jitter == pytest.approx(0.0)   # identical box

    # pass 3: shifted-but-overlapping box -> reinforce, jitter grows
    added, dup = _register(pipeline_module, graph, [2, 0, 22, 20], 3, "3:global:green fruit:0.35")
    assert (added, dup) == (0, 1)
    assert node.support == 3
    assert node.jitter > 0.0    # center moved by 2px on pass 3


# ----------------------- cross-pass dedup_metric: iou vs iom -----------------------
#
# A tight detection (small box) and a loose detection (bigger box) of the SAME
# real object have low IoU (small overlap relative to the big union) but high
# IoM (the smaller box sits almost entirely inside the bigger one) -- the
# validated CARPK fix for IoU under-merging tight-vs-loose detections of one
# dense-scene object.

def test_dedup_metric_iom_merges_tight_and_loose_detection_of_same_object(pipeline_module):
    graph = OrchardGraph()
    _register(pipeline_module, graph, [0, 0, 40, 40], 1, "s1")  # loose, area 1600

    # tight box mostly inside the loose one: IoU=100/1600=0.0625, IoM=100/100=1.0
    added, dup = pipeline_module.register_and_verify_candidates(
        np.array([[5, 5, 15, 15]], dtype=float), np.array([0.9]),
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=2,
        signature="s2", iou_threshold=0.40, dedup_metric="iom",
    )
    assert (added, dup) == (0, 1)
    assert len(graph.nodes) == 1  # merged, not a new node


def test_dedup_metric_iou_default_does_not_merge_same_scenario(pipeline_module):
    graph = OrchardGraph()
    _register(pipeline_module, graph, [0, 0, 40, 40], 1, "s1")

    added, dup = pipeline_module.register_and_verify_candidates(
        np.array([[5, 5, 15, 15]], dtype=float), np.array([0.9]),
        leaf_boxes=np.empty((0, 4)), graph=graph, pass_number=2,
        signature="s2", iou_threshold=0.40,  # dedup_metric defaults to "iou"
    )
    assert (added, dup) == (1, 0)
    assert len(graph.nodes) == 2  # NOT merged -- demonstrates IoU under-merging

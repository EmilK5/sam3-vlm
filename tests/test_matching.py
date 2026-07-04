import numpy as np

from graph import OrchardGraph
from eval.matching import match, pool_recall, per_pass_pool_recall


# ----------------------- 4 toy match() cases -----------------------

def test_perfect_match():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    pred = [[0, 0, 10, 10], [20, 20, 30, 30]]
    r = match(pred, gt)
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0
    assert r["pred_matched"].all() and r["gt_matched"].all()


def test_nested_box_prefers_exact_and_drops_nested():
    # One GT; two candidates: an exact box (IoU 1.0) and a nested box (IoU 0.25).
    # Hungarian must take the exact match; the nested box is a false positive.
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [0, 0, 5, 5]]
    r = match(pred, gt, iou_thr=0.5)
    assert r["matches"] == [(0, 0)]
    assert r["recall"] == 1.0
    assert r["precision"] == 0.5          # 1 of 2 predictions matched
    assert list(r["pred_matched"]) == [True, False]


def test_below_threshold_no_match():
    # Offset box with IoU < 0.5 -> unmatched.
    gt = [[0, 0, 10, 10]]
    pred = [[6, 6, 16, 16]]               # IoU = 16 / 184 ~= 0.087
    r = match(pred, gt, iou_thr=0.5)
    assert r["matches"] == []
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_empty_predictions():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    r = match([], gt)
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["pred_matched"].shape == (0,)
    assert not r["gt_matched"].any()


# ----------------------- pool_recall -----------------------

def test_pool_recall_counts_coverage_not_one_to_one():
    # Two candidates both cover the single GT: pool recall is 1.0 even though
    # one-to-one precision would penalize the duplicate.
    gt = [[0, 0, 10, 10]]
    pool = [[0, 0, 10, 10], [1, 0, 11, 10]]  # both high IoU with the GT
    assert pool_recall(pool, gt) == 1.0


def test_pool_recall_partial_and_empty():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    pool = [[0, 0, 10, 10]]                 # only the first GT covered
    assert pool_recall(pool, gt) == 0.5
    assert pool_recall([], gt) == 0.0
    assert pool_recall(pool, []) == 0.0     # no GT -> 0.0


# ----------------------- per_pass_pool_recall -----------------------

def test_per_pass_pool_recall_is_cumulative():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    graph = OrchardGraph()
    graph.add_candidate([0, 0, 10, 10], score=0.9, found_in_pass=1)      # covers GT0
    graph.add_candidate([50, 50, 60, 60], score=0.8, found_in_pass=2)    # covers GT1

    by_pass = per_pass_pool_recall(graph, gt)
    assert by_pass == {1: 0.5, 2: 1.0}     # pass 2 pool includes pass-1 candidates


def test_per_pass_pool_recall_empty_graph():
    assert per_pass_pool_recall(OrchardGraph(), [[0, 0, 10, 10]]) == {}

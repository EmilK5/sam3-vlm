"""
eval/matching.py

Detection-level evaluation: Hungarian one-to-one matching of predicted boxes to
ground-truth boxes at an IoU threshold, and pool-recall diagnostics.

Boxes are global-frame xyxy pixels. Definitions follow the proposal:
    precision   = #matched candidates / #candidate tracks
    recall      = #matched GT / #GT
    F1          = harmonic mean of precision and recall
    pool recall = fraction of GT covered by ANY candidate in the pool
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(pred_boxes, gt_boxes) -> np.ndarray:
    """(P,G) IoU matrix between predicted and ground-truth boxes (xyxy)."""
    pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=float)

    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    union = pred_area[:, None] + gt_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred_boxes, gt_boxes, iou_thr: float = 0.5) -> dict:
    """One-to-one Hungarian matching at an IoU threshold.

    Returns a dict:
        precision, recall, f1 : floats
        pred_matched          : bool array (P,) — which predictions matched a GT
        gt_matched            : bool array (G,) — which GT objects were matched
        matches               : list of (pred_idx, gt_idx) accepted pairs
    """
    pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    P, G = len(pred), len(gt)

    iou = iou_matrix(pred, gt)
    pred_matched = np.zeros(P, dtype=bool)
    gt_matched = np.zeros(G, dtype=bool)
    matches = []

    if P > 0 and G > 0:
        # maximize total IoU, then drop pairs that fall below the threshold
        rows, cols = linear_sum_assignment(iou, maximize=True)
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_thr:
                pred_matched[r] = True
                gt_matched[c] = True
                matches.append((int(r), int(c)))

    n = len(matches)
    precision = n / P if P > 0 else 0.0
    recall = n / G if G > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_matched": pred_matched,
        "gt_matched": gt_matched,
        "matches": matches,
    }


def pool_recall(all_candidate_boxes, gt_boxes, iou_thr: float = 0.5) -> float:
    """Fraction of GT objects covered by AT LEAST ONE candidate (not one-to-one).

    Measures whether the candidate pool discovered each true object, independent
    of how many candidates overlap it. Returns 0.0 when there is no GT.
    """
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if len(gt) == 0:
        return 0.0
    cand = np.asarray(all_candidate_boxes, dtype=float).reshape(-1, 4)
    if len(cand) == 0:
        return 0.0
    iou = iou_matrix(cand, gt)              # (C, G)
    covered = iou.max(axis=0) >= iou_thr    # per-GT best candidate clears the bar
    return float(covered.sum()) / len(gt)


def per_pass_pool_recall(graph, gt_boxes, iou_thr: float = 0.5) -> dict:
    """Cumulative pool recall after each pass, using node.found_in_pass.

    Returns {pass_number: pool_recall} for passes 1..max_pass, where the pool at
    pass p is every candidate discovered on pass <= p.
    """
    nodes = list(graph.nodes.values())
    if not nodes:
        return {}
    max_pass = max(n.found_in_pass for n in nodes)
    result = {}
    for p in range(1, max_pass + 1):
        pool = [n.box for n in nodes if n.found_in_pass <= p]
        result[p] = pool_recall(pool, gt_boxes, iou_thr)
    return result

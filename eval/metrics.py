"""
eval/metrics.py

Count-level metrics and the normalized sensing cost.

mae/rmse/exact take paired predicted vs ground-truth counts. normalized_cost
implements the proposal's compute model from a dict of call counts and the cost
weights in Config (weights normalized relative to one global SAM3 call). When
CostMeter arrives (step 3.2) it can expose the same counts dict to this function.
"""

import numpy as np


def _arrays(preds, gts):
    p = np.asarray(preds, dtype=float)
    g = np.asarray(gts, dtype=float)
    if p.shape != g.shape:
        raise ValueError(f"preds and gts must be the same length; got {p.shape} vs {g.shape}.")
    return p, g


def mae(preds, gts) -> float:
    """Mean absolute error of paired counts."""
    p, g = _arrays(preds, gts)
    return float(np.mean(np.abs(p - g))) if p.size else 0.0


def rmse(preds, gts) -> float:
    """Root mean squared error of paired counts."""
    p, g = _arrays(preds, gts)
    return float(np.sqrt(np.mean((p - g) ** 2))) if p.size else 0.0


def exact(preds, gts) -> float:
    """Fraction of images whose predicted count exactly equals the ground truth."""
    p, g = _arrays(preds, gts)
    return float(np.mean(np.round(p) == np.round(g))) if p.size else 0.0


def normalized_cost(counts: dict, cfg) -> float:
    """Sensing cost normalized to one global SAM3 call.

    counts keys (any missing -> 0): n_global, n_tile, n_verify, n_inspect, n_orch.
    Weights come from cfg.costs ratios: lambda_x = c_x / c_sam.
    """
    def n(key):
        return counts.get(key, 0)

    c_sam = cfg.c_sam
    return (
        n("n_global")
        + (cfg.c_tile / c_sam) * n("n_tile")
        + (cfg.c_verify / c_sam) * n("n_verify")
        + (cfg.c_inspect / c_sam) * n("n_inspect")
        + (cfg.c_orch / c_sam) * n("n_orch")
    )

"""Entropy, expected information gain, and realized information utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from agent.asht.belief import bayes_update, normalize_distribution
from agent.asht.kernels import SurrogateKernel


def entropy(probabilities: Sequence[float]) -> float:
    p = normalize_distribution(probabilities)
    positive = p[p > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def predictive_distribution(
    posterior: Sequence[float],
    kernel: SurrogateKernel | np.ndarray,
) -> np.ndarray:
    p = normalize_distribution(posterior)
    matrix = kernel.matrix if isinstance(kernel, SurrogateKernel) else np.asarray(kernel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(p):
        raise ValueError("kernel rows must match posterior length")
    predictive = p @ matrix
    return normalize_distribution(predictive)


def hypothetical_posterior(
    posterior: Sequence[float],
    kernel: SurrogateKernel | np.ndarray,
    observation_index: int,
) -> np.ndarray:
    matrix = kernel.matrix if isinstance(kernel, SurrogateKernel) else np.asarray(kernel, dtype=float)
    if not 0 <= int(observation_index) < matrix.shape[1]:
        raise ValueError("observation index out of range")
    updated, _ = bayes_update(posterior, matrix[:, int(observation_index)])
    return updated


def expected_information_gain(
    posterior: Sequence[float],
    kernel: SurrogateKernel | np.ndarray,
) -> float:
    matrix = kernel.matrix if isinstance(kernel, SurrogateKernel) else np.asarray(kernel, dtype=float)
    predictive = predictive_distribution(posterior, matrix)
    current_entropy = entropy(posterior)
    expected_after = 0.0
    for observation_index, probability in enumerate(predictive):
        if probability <= 0.0:
            continue
        expected_after += float(probability) * entropy(
            hypothetical_posterior(posterior, matrix, observation_index)
        )
    value = current_entropy - expected_after
    return max(0.0, float(value))


def kl_divergence(posterior_after: Sequence[float], posterior_before: Sequence[float]) -> float:
    q = normalize_distribution(posterior_after)
    p = normalize_distribution(posterior_before)
    if q.shape != p.shape:
        raise ValueError("KL distributions must have the same shape")
    if np.any((q > 0.0) & (p <= 0.0)):
        return math.inf
    mask = q > 0.0
    return float(np.sum(q[mask] * np.log(q[mask] / p[mask])))


def realized_entropy_reduction(
    posterior_before: Sequence[float],
    posterior_after: Sequence[float],
) -> float:
    return entropy(posterior_before) - entropy(posterior_after)


@dataclass(frozen=True)
class RankedAction:
    action_id: str
    expected_information_gain: float
    expected_cost: float | None
    score: float
    rank: int


def rank_actions(
    posterior: Sequence[float],
    kernels: Mapping[str, SurrogateKernel],
    *,
    expected_costs: Mapping[str, float] | None = None,
) -> tuple[RankedAction, ...]:
    rows: list[tuple[str, float, float | None, float]] = []
    for action_id, kernel in kernels.items():
        eig = expected_information_gain(posterior, kernel)
        cost = None if expected_costs is None else float(expected_costs[action_id])
        if cost is not None:
            if not math.isfinite(cost) or cost <= 0.0:
                raise ValueError("expected costs must be finite and positive")
            score = eig / cost
        else:
            score = eig
        rows.append((action_id, eig, cost, score))
    rows.sort(key=lambda row: (-row[3], -row[1], row[0]))
    return tuple(
        RankedAction(
            action_id=action_id,
            expected_information_gain=eig,
            expected_cost=cost,
            score=score,
            rank=index + 1,
        )
        for index, (action_id, eig, cost, score) in enumerate(rows)
    )


__all__ = [
    "RankedAction",
    "entropy",
    "expected_information_gain",
    "hypothetical_posterior",
    "kl_divergence",
    "predictive_distribution",
    "rank_actions",
    "realized_entropy_reduction",
]

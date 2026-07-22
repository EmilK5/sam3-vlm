"""Posterior-confidence stopping and declaration rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from agent.asht.belief import normalize_distribution


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    declaration: str | None
    confidence: float
    threshold: float
    reason: str


def posterior_stop(
    class_names: Sequence[str],
    posterior: Sequence[float],
    *,
    threshold: float,
    budget_exhausted: bool = False,
) -> StopDecision:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    p = normalize_distribution(posterior)
    if len(class_names) != len(p):
        raise ValueError("class_names must match posterior length")
    index = int(np.argmax(p))
    confidence = float(p[index])
    if confidence >= threshold:
        return StopDecision(True, str(class_names[index]), confidence, threshold, "confidence")
    if budget_exhausted:
        return StopDecision(True, str(class_names[index]), confidence, threshold, "budget")
    return StopDecision(False, None, confidence, threshold, "not_stopped")


__all__ = ["StopDecision", "posterior_stop"]

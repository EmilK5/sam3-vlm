"""Hard and soft counts derived from patch-level posterior beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent.asht.belief import PatchBelief


@dataclass(frozen=True)
class CountEstimate:
    target_class: str
    hard_count: int
    soft_count: float
    variance: float
    unresolved_count: int


def estimate_count(
    beliefs: Iterable[PatchBelief],
    target_class: str,
) -> CountEstimate:
    hard = 0
    soft = 0.0
    variance = 0.0
    unresolved = 0
    seen = 0
    for belief in beliefs:
        seen += 1
        try:
            index = belief.class_names.index(target_class)
        except ValueError as exc:
            raise ValueError(
                f"target class {target_class!r} absent from belief {belief.node_id!r}"
            ) from exc
        probability = float(belief.posterior[index])
        soft += probability
        variance += probability * (1.0 - probability)
        if belief.map_class == target_class:
            hard += 1
        if belief.status == "unresolved":
            unresolved += 1
    return CountEstimate(
        target_class=target_class,
        hard_count=hard,
        soft_count=soft,
        variance=variance,
        unresolved_count=unresolved,
    )


__all__ = ["CountEstimate", "estimate_count"]

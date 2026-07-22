"""Patch-level Bayesian belief state for active sequential testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class BeliefError(ValueError):
    """Raised when a belief or likelihood vector is invalid."""


_VALID_STATUSES = {"unresolved", "stopped", "budget_exhausted", "rejected"}


def normalize_distribution(values: Sequence[float], *, epsilon: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape((-1,))
    if array.size == 0:
        raise BeliefError("a probability vector must not be empty")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise BeliefError("probabilities must be finite and non-negative")
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise BeliefError("epsilon must be finite and non-negative")
    if epsilon:
        array = np.maximum(array, epsilon)
    total = float(array.sum())
    if total <= 0.0:
        raise BeliefError("a probability vector must have positive mass")
    if math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return array.copy()
    return array / total


def bayes_update(
    posterior_before: Sequence[float],
    likelihood_by_class: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Apply one Bayesian update and return posterior plus predictive mass."""

    prior = normalize_distribution(posterior_before)
    likelihood = np.asarray(likelihood_by_class, dtype=float).reshape((-1,))
    if likelihood.shape != prior.shape:
        raise BeliefError("likelihood and posterior must have the same shape")
    if np.any(~np.isfinite(likelihood)) or np.any(likelihood < 0.0):
        raise BeliefError("likelihoods must be finite and non-negative")
    likelihood = np.maximum(likelihood, epsilon)
    unnormalized = prior * likelihood
    predictive = float(unnormalized.sum())
    if not math.isfinite(predictive) or predictive <= 0.0:
        raise BeliefError("the observation has zero or invalid predictive probability")
    return unnormalized / predictive, predictive


@dataclass
class PatchBelief:
    """Mutable runtime state for one candidate patch.

    The class is intentionally serializable without numpy-specific JSON support.
    ``posterior_history`` stores every posterior, including the initial one.
    """

    node_id: str
    class_names: tuple[str, ...]
    prior: np.ndarray
    posterior: np.ndarray | None = None
    query_count: int = 0
    status: str = "unresolved"
    decision: str | None = None
    stop_reason: str | None = None
    posterior_history: list[np.ndarray] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise BeliefError("node_id must be non-empty")
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise BeliefError("class_names must be non-empty and unique")
        self.prior = normalize_distribution(self.prior)
        if len(self.prior) != len(self.class_names):
            raise BeliefError("prior length must match class_names")
        self.posterior = normalize_distribution(
            self.prior if self.posterior is None else self.posterior
        )
        if len(self.posterior) != len(self.class_names):
            raise BeliefError("posterior length must match class_names")
        if self.query_count < 0:
            raise BeliefError("query_count must be non-negative")
        if self.status not in _VALID_STATUSES:
            raise BeliefError(f"invalid belief status: {self.status!r}")
        if self.decision is not None and self.decision not in self.class_names:
            raise BeliefError("decision must be one of class_names")
        normalized_history: list[np.ndarray] = []
        for index, item in enumerate(self.posterior_history):
            normalized = normalize_distribution(item)
            if len(normalized) != len(self.class_names):
                raise BeliefError(
                    f"posterior_history[{index}] length must match class_names"
                )
            normalized_history.append(normalized)
        self.posterior_history = normalized_history or [self.posterior.copy()]

    @classmethod
    def from_mapping(
        cls,
        node_id: str,
        probabilities: Mapping[str, float],
    ) -> "PatchBelief":
        classes = tuple(probabilities.keys())
        return cls(
            node_id=node_id,
            class_names=classes,
            prior=np.asarray([probabilities[name] for name in classes], dtype=float),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchBelief":
        required = {"node_id", "class_names", "prior", "posterior"}
        missing = required - set(payload)
        if missing:
            raise BeliefError(f"belief payload missing fields: {sorted(missing)}")
        return cls(
            node_id=str(payload["node_id"]),
            class_names=tuple(str(value) for value in payload["class_names"]),
            prior=np.asarray(payload["prior"], dtype=float),
            posterior=np.asarray(payload["posterior"], dtype=float),
            query_count=int(payload.get("query_count", 0)),
            status=str(payload.get("status", "unresolved")),
            decision=(
                None if payload.get("decision") is None else str(payload["decision"])
            ),
            stop_reason=(
                None
                if payload.get("stop_reason") is None
                else str(payload["stop_reason"])
            ),
            posterior_history=[
                np.asarray(item, dtype=float)
                for item in payload.get("posterior_history", [])
            ],
            evidence_ids=[str(value) for value in payload.get("evidence_ids", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "class_names": list(self.class_names),
            "prior": [float(value) for value in self.prior],
            "posterior": [float(value) for value in self.posterior],
            "query_count": int(self.query_count),
            "status": self.status,
            "decision": self.decision,
            "stop_reason": self.stop_reason,
            "posterior_history": [
                [float(value) for value in posterior]
                for posterior in self.posterior_history
            ],
            "evidence_ids": list(self.evidence_ids),
        }

    def as_mapping(self) -> dict[str, float]:
        return {
            name: float(self.posterior[index])
            for index, name in enumerate(self.class_names)
        }

    def prior_mapping(self) -> dict[str, float]:
        return {
            name: float(self.prior[index])
            for index, name in enumerate(self.class_names)
        }

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.posterior))

    @property
    def map_class(self) -> str:
        return self.class_names[self.map_index]

    @property
    def confidence(self) -> float:
        return float(self.posterior[self.map_index])

    @property
    def resolved(self) -> bool:
        return self.status in {"stopped", "budget_exhausted", "rejected"}

    def apply_likelihood(
        self,
        likelihood_by_class: Sequence[float],
        *,
        evidence_id: str | None = None,
        epsilon: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        before = self.posterior.copy()
        after, predictive = bayes_update(before, likelihood_by_class, epsilon=epsilon)
        self.posterior = after
        self.query_count += 1
        self.posterior_history.append(after.copy())
        if evidence_id is not None:
            self.evidence_ids.append(evidence_id)
        return before, after.copy(), predictive

    def mark_stopped(self, *, reason: str, declaration: str | None = None) -> None:
        if reason not in {"confidence", "budget", "no_valid_action", "policy", "error"}:
            raise BeliefError(f"unsupported stop reason: {reason!r}")
        self.status = "budget_exhausted" if reason == "budget" else "stopped"
        self.decision = declaration or self.map_class
        self.stop_reason = reason


__all__ = [
    "BeliefError",
    "PatchBelief",
    "bayes_update",
    "normalize_distribution",
]

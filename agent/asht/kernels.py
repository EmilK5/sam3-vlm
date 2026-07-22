"""Surrogate observation kernels for zero-shot active testing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from agent.asht.belief import BeliefError, normalize_distribution


class KernelError(ValueError):
    """Raised when a sensor profile or surrogate kernel is invalid."""


@dataclass(frozen=True)
class SensorProfile:
    observation_labels: tuple[str, ...]
    present: tuple[float, ...]
    absent: tuple[float, ...]
    source: str = "configured"

    def __post_init__(self) -> None:
        if not self.observation_labels:
            raise KernelError("observation_labels must not be empty")
        if len(set(self.observation_labels)) != len(self.observation_labels):
            raise KernelError("observation_labels must be unique")
        if len(self.present) != len(self.observation_labels):
            raise KernelError("present profile length must match observation labels")
        if len(self.absent) != len(self.observation_labels):
            raise KernelError("absent profile length must match observation labels")
        _validate_profile(self.present, "present")
        _validate_profile(self.absent, "absent")
        if not self.source.strip():
            raise KernelError("profile source must be non-empty")


@dataclass(frozen=True)
class SurrogateKernel:
    class_names: tuple[str, ...]
    observation_labels: tuple[str, ...]
    beta_by_class: Mapping[str, float]
    matrix: np.ndarray
    present_profile: np.ndarray
    absent_profile: np.ndarray
    epsilon: float
    profile_source: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "present_profile", np.asarray(self.present_profile, dtype=float))
        object.__setattr__(self, "absent_profile", np.asarray(self.absent_profile, dtype=float))
        if matrix.shape != (len(self.class_names), len(self.observation_labels)):
            raise KernelError("kernel matrix shape does not match classes and observations")
        if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise KernelError("kernel probabilities must be finite and non-negative")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
            raise KernelError("each kernel row must sum to one")

    def likelihood(self, observation: int | str) -> np.ndarray:
        if isinstance(observation, str):
            try:
                index = self.observation_labels.index(observation)
            except ValueError as exc:
                raise KernelError(f"unknown observation label: {observation!r}") from exc
        else:
            index = int(observation)
        if not 0 <= index < len(self.observation_labels):
            raise KernelError("observation index out of range")
        return self.matrix[:, index].copy()

    def row_mapping(self) -> dict[str, tuple[float, ...]]:
        return {
            name: tuple(float(v) for v in self.matrix[index])
            for index, name in enumerate(self.class_names)
        }


def build_surrogate_kernel(
    class_names: Sequence[str],
    beta_by_class: Mapping[str, float],
    profile: SensorProfile,
    *,
    epsilon: float = 1e-6,
) -> SurrogateKernel:
    """Build q_m^a(y)=beta_m g(y|1)+(1-beta_m)g(y|0)."""

    classes = tuple(str(name) for name in class_names)
    if not classes or len(set(classes)) != len(classes):
        raise KernelError("class_names must be non-empty and unique")
    if set(classes) != set(beta_by_class):
        raise KernelError("beta_by_class keys must exactly match class_names")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise KernelError("epsilon must lie in (0, 0.5)")

    present = normalize_distribution(profile.present)
    absent = normalize_distribution(profile.absent)
    rows = []
    for name in classes:
        beta = float(beta_by_class[name])
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise KernelError(f"beta_by_class[{name!r}] must lie in [0, 1]")
        row = beta * present + (1.0 - beta) * absent
        row = np.clip(row, epsilon, None)
        row = row / row.sum()
        rows.append(row)
    matrix = np.vstack(rows)
    return SurrogateKernel(
        class_names=classes,
        observation_labels=profile.observation_labels,
        beta_by_class={name: float(beta_by_class[name]) for name in classes},
        matrix=matrix,
        present_profile=present,
        absent_profile=absent,
        epsilon=epsilon,
        profile_source=profile.source,
    )


def _validate_profile(values: Sequence[float], name: str) -> None:
    try:
        normalized = normalize_distribution(values)
    except BeliefError as exc:
        raise KernelError(f"invalid {name} profile: {exc}") from exc
    if not np.allclose(normalized, np.asarray(values, dtype=float), atol=1e-9):
        raise KernelError(f"{name} profile must already sum to one")


__all__ = [
    "KernelError",
    "SensorProfile",
    "SurrogateKernel",
    "build_surrogate_kernel",
]

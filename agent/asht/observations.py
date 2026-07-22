"""Finite observation encoder for targeted SAM3 verification queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from provenance.schema import ObservationLevel


@dataclass(frozen=True)
class ObservationEncoderConfig:
    min_iou: float = 0.10
    weak_score: float = 0.25
    strong_score: float = 0.60
    strong_iou: float = 0.40
    version: str = "target-overlap-v1"

    def __post_init__(self) -> None:
        for name in ("min_iou", "weak_score", "strong_score", "strong_iou"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.strong_score < self.weak_score:
            raise ValueError("strong_score must be at least weak_score")
        if not self.version.strip():
            raise ValueError("encoder version must be non-empty")


@dataclass(frozen=True)
class EncodedObservation:
    level: ObservationLevel
    label: str
    matched_indices: tuple[int, ...]
    best_index: int | None
    overlap_metrics: Mapping[str, float]
    score_statistics: Mapping[str, float]
    bin_thresholds: Mapping[str, float]
    raw_features: Mapping[str, object] = field(default_factory=dict)

    @property
    def index(self) -> int:
        order = (
            ObservationLevel.NOT_FOUND,
            ObservationLevel.WEAK_MATCH,
            ObservationLevel.STRONG_MATCH,
        )
        return order.index(self.level)


def encode_target_observation(
    target_box: Sequence[float],
    detection_boxes: Sequence[Sequence[float]],
    detection_scores: Sequence[float],
    *,
    config: ObservationEncoderConfig = ObservationEncoderConfig(),
) -> EncodedObservation:
    boxes = np.asarray(detection_boxes, dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    boxes = boxes.reshape((-1, 4))
    scores = np.asarray(detection_scores, dtype=float).reshape((-1,))
    if len(boxes) != len(scores):
        raise ValueError("detection boxes and scores must have the same length")
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("detection scores must be finite and in [0, 1]")

    ious = np.asarray([_iou(target_box, box) for box in boxes], dtype=float)
    matched = np.where(ious >= config.min_iou)[0]
    best_index = None
    best_iou = 0.0
    best_score = 0.0
    if matched.size:
        ordering = sorted(
            (int(index) for index in matched),
            key=lambda index: (float(ious[index]), float(scores[index])),
            reverse=True,
        )
        best_index = ordering[0]
        best_iou = float(ious[best_index])
        best_score = float(scores[best_index])

    if best_index is None or best_score < config.weak_score:
        level = ObservationLevel.NOT_FOUND
        label = "not_found"
    elif best_score >= config.strong_score and best_iou >= config.strong_iou:
        level = ObservationLevel.STRONG_MATCH
        label = "strong_match"
    else:
        level = ObservationLevel.WEAK_MATCH
        label = "weak_match"

    return EncodedObservation(
        level=level,
        label=label,
        matched_indices=tuple(int(v) for v in matched),
        best_index=best_index,
        overlap_metrics={
            "best_iou": best_iou,
            "mean_matched_iou": float(ious[matched].mean()) if matched.size else 0.0,
        },
        score_statistics={
            "best_score": best_score,
            "mean_matched_score": float(scores[matched].mean()) if matched.size else 0.0,
            "num_detections": float(len(scores)),
            "num_matched": float(len(matched)),
        },
        bin_thresholds={
            "min_iou": config.min_iou,
            "weak_score": config.weak_score,
            "strong_score": config.strong_score,
            "strong_iou": config.strong_iou,
        },
        raw_features={
            "all_ious": [float(value) for value in ious],
            "all_scores": [float(value) for value in scores],
            "encoder_version": config.version,
        },
    )


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


__all__ = [
    "EncodedObservation",
    "ObservationEncoderConfig",
    "encode_target_observation",
]

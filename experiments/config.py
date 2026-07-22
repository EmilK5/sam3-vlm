"""Immutable configuration for the unified experiment runner."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from agent.asht.runner_static import StaticAshtConfig
from agent.asht.tiling import AdaptiveTilingConfig
from provenance.run_store import ReportingLevel
from provenance.schema import TilingMode


class ExperimentMode(str, Enum):
    SAM3_SINGLE_PASS = "sam3_single_pass"
    SAM3_FIXED_MULTIPASS = "sam3_fixed_multipass"
    SAM3_ADAPTIVE_TILING = "sam3_adaptive_tiling"
    LEGACY_VLM = "legacy_vlm"
    ASHT_STATIC = "asht_static"
    ASHT_QWEN = "asht_qwen"
    ASHT_ADAPTIVE_TILING = "asht_adaptive_tiling"
    ASHT_AGENT_TILING = "asht_agent_tiling"


@dataclass(frozen=True)
class DiscoveryPassSpec:
    prompt: str
    threshold: float
    tiling_mode: TilingMode = TilingMode.OFF
    tile_scale: float | None = None
    region: tuple[float, float, float, float] | None = None
    signature: str | None = None
    return_masks: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("discovery prompt must be non-empty")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("discovery threshold must lie in [0, 1]")
        if self.tile_scale is not None and (
            not math.isfinite(self.tile_scale) or self.tile_scale <= 0.0
        ):
            raise ValueError("tile_scale must be finite and positive")
        if self.region is not None:
            x1, y1, x2, y2 = self.region
            if not all(math.isfinite(float(value)) for value in self.region):
                raise ValueError("region must contain finite coordinates")
            if x2 < x1 or y2 < y1:
                raise ValueError("region must satisfy x2 >= x1 and y2 >= y1")


@dataclass(frozen=True)
class UnifiedExperimentConfig:
    mode: ExperimentMode
    output_root: Path
    reporting_level: ReportingLevel = ReportingLevel.FULL
    random_seed: int = 0
    overwrite: bool = False
    fsync: bool = True
    checkpoint_every_events: int = 0
    raise_on_error: bool = True
    class_names: tuple[str, ...] = ("target", "non_target", "background")
    target_class: str = "target"
    discovery_passes: tuple[DiscoveryPassSpec, ...] = ()
    asht_config: StaticAshtConfig | None = None
    adaptive_tiling: AdaptiveTilingConfig = field(default_factory=AdaptiveTilingConfig)
    suppression_confidence: float = 0.0
    suppression_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cross_pass_dedup_metric: str = "iou"
    cross_pass_dedup_threshold: float = 0.40
    model_id: str = "sam3"
    run_name: str | None = None
    resolved_overrides: Mapping[str, Any] = field(default_factory=dict)
    legacy_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be non-empty and unique")
        if self.target_class not in self.class_names:
            raise ValueError("target_class must belong to class_names")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.checkpoint_every_events < 0:
            raise ValueError("checkpoint_every_events must be non-negative")
        if not 0.0 <= self.suppression_confidence <= 1.0:
            raise ValueError("suppression_confidence must lie in [0, 1]")
        if self.cross_pass_dedup_metric not in {"iou", "iom"}:
            raise ValueError("cross_pass_dedup_metric must be 'iou' or 'iom'")
        if not 0.0 <= self.cross_pass_dedup_threshold <= 1.0:
            raise ValueError("cross_pass_dedup_threshold must lie in [0, 1]")
        discovery_modes = {
            ExperimentMode.SAM3_SINGLE_PASS,
            ExperimentMode.SAM3_FIXED_MULTIPASS,
            ExperimentMode.SAM3_ADAPTIVE_TILING,
        }
        if self.mode in discovery_modes and not self.discovery_passes:
            raise ValueError(f"{self.mode.value} requires discovery_passes")
        if self.mode is ExperimentMode.SAM3_SINGLE_PASS and len(self.discovery_passes) != 1:
            raise ValueError("sam3_single_pass requires exactly one discovery pass")
        if self.mode is ExperimentMode.SAM3_ADAPTIVE_TILING:
            if not any(spec.tiling_mode is not TilingMode.OFF for spec in self.discovery_passes):
                raise ValueError("sam3_adaptive_tiling requires a tiled pass")
        if self.mode in {
            ExperimentMode.ASHT_STATIC,
            ExperimentMode.ASHT_QWEN,
            ExperimentMode.ASHT_ADAPTIVE_TILING,
            ExperimentMode.ASHT_AGENT_TILING,
        } and self.asht_config is None:
            raise ValueError(f"{self.mode.value} requires asht_config")

    def resolved_mapping(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode.value,
            "output_root": str(self.output_root),
            "reporting_level": self.reporting_level.value,
            "random_seed": self.random_seed,
            "class_names": list(self.class_names),
            "target_class": self.target_class,
            "discovery_passes": [_dataclass_json(spec) for spec in self.discovery_passes],
            "adaptive_tiling": _dataclass_json(self.adaptive_tiling),
            "suppression_confidence": self.suppression_confidence,
            "suppression_kwargs": dict(self.suppression_kwargs),
            "cross_pass_dedup_metric": self.cross_pass_dedup_metric,
            "cross_pass_dedup_threshold": self.cross_pass_dedup_threshold,
            "model_id": self.model_id,
            "run_name": self.run_name,
            "legacy_options": dict(self.legacy_options),
        }
        if self.asht_config is not None:
            payload["asht"] = _describe_asht(self.asht_config)
        payload.update(dict(self.resolved_overrides))
        return payload


def _describe_asht(config: StaticAshtConfig) -> Mapping[str, Any]:
    return {
        "class_names": list(config.class_names),
        "target_class": config.target_class,
        "prior": dict(config.prior),
        "stopping_threshold": config.stopping_threshold,
        "max_total_queries": config.max_total_queries,
        "max_queries_per_node": config.max_queries_per_node,
        "use_cost_adjusted_eig": config.use_cost_adjusted_eig,
        "suppression_confidence": config.suppression_confidence,
        "suppression_kwargs": dict(config.suppression_kwargs),
        "bootstrap_query": (
            None
            if config.bootstrap_query is None
            else {
                "prompt": config.bootstrap_query.prompt,
                "threshold": config.bootstrap_query.threshold,
                "region": list(config.bootstrap_query.region),
                "return_masks": config.bootstrap_query.return_masks,
            }
        ),
        "bootstrap_dedup_metric": config.bootstrap_dedup_metric,
        "bootstrap_dedup_threshold": config.bootstrap_dedup_threshold,
        "bootstrap_tiling_mode": config.bootstrap_tiling_mode.value,
        "action_bank": {
            "allow_semantic_reuse": config.action_bank.allow_semantic_reuse,
            "positive_class": config.action_bank.positive_class,
            "negative_class": config.action_bank.negative_class,
            "positive_exemplar_threshold": config.action_bank.positive_exemplar_threshold,
            "negative_exemplar_threshold": config.action_bank.negative_exemplar_threshold,
            "templates": [_dataclass_json(item) for item in config.action_bank.templates],
            "sensor_profile": _dataclass_json(config.action_bank.profile),
        },
    }


def _dataclass_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _dataclass_json(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _dataclass_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_dataclass_json(item) for item in value]
    return value


__all__ = ["DiscoveryPassSpec", "ExperimentMode", "UnifiedExperimentConfig"]

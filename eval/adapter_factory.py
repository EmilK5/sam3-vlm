"""Configuration-driven construction of canonical dataset adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.dataset_adapters import (
    CarpkAdapter,
    DatasetAdapter,
    DatasetAdapterError,
    DatasetRegistry,
    FourDSemanticMappingAdapter,
    Fscd147Adapter,
    GenericFolderAdapter,
    GreenCitrusAdapter,
    OmniCountAdapter,
    YoloDatasetAdapter,
)


@dataclass(frozen=True)
class DatasetAdapterSpec:
    kind: str
    root: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.kind.strip():
            raise DatasetAdapterError("adapter kind must be non-empty")


def build_adapter(spec: DatasetAdapterSpec) -> DatasetAdapter:
    kind = spec.kind.strip().lower().replace("-", "_")
    options = dict(spec.options)
    if kind in {"green_citrus", "citrus"}:
        return GreenCitrusAdapter(spec.root, **options)
    if kind == "carpk":
        return CarpkAdapter(spec.root, **options)
    if kind in {"fscd147", "fsc147"}:
        return Fscd147Adapter(spec.root, **options)
    if kind in {"omnicount", "omni_count"}:
        try:
            annotation_file = options.pop("annotation_file")
        except KeyError as exc:
            raise DatasetAdapterError("OmniCount requires annotation_file") from exc
        return OmniCountAdapter(spec.root, annotation_file, **options)
    if kind in {"4d", "4d_semantic_mapping", "four_d"}:
        return FourDSemanticMappingAdapter(spec.root, **options)
    if kind in {"generic", "folder"}:
        return GenericFolderAdapter(spec.root, **options)
    if kind == "yolo":
        required = ("name", "target_concept", "target_class")
        missing = [name for name in required if name not in options]
        if missing:
            raise DatasetAdapterError(f"YOLO adapter missing options: {missing}")
        return YoloDatasetAdapter(spec.root, **options)
    raise DatasetAdapterError(f"unknown adapter kind: {spec.kind!r}")


def build_registry(specifications: Sequence[DatasetAdapterSpec]) -> DatasetRegistry:
    return DatasetRegistry(build_adapter(specification) for specification in specifications)


def export_dataset_summary(
    adapter: DatasetAdapter,
    split: str,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adapter.summary(split), indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "DatasetAdapterSpec",
    "build_adapter",
    "build_registry",
    "export_dataset_summary",
]

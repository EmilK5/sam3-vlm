"""Evaluation and dataset-normalization utilities."""

from eval.dataset_adapters import (
    CanonicalSample,
    CarpkAdapter,
    DatasetAdapter,
    DatasetAdapterError,
    DatasetCapabilities,
    DatasetRegistry,
    FourDSemanticMappingAdapter,
    Fscd147Adapter,
    GenericFolderAdapter,
    GreenCitrusAdapter,
    OmniCountAdapter,
    YoloDatasetAdapter,
    validate_adapter,
)
from eval.adapter_factory import (
    DatasetAdapterSpec,
    build_adapter,
    build_registry,
    export_dataset_summary,
)

__all__ = [
    "CanonicalSample",
    "DatasetAdapterSpec",
    "CarpkAdapter",
    "DatasetAdapter",
    "DatasetAdapterError",
    "DatasetCapabilities",
    "DatasetRegistry",
    "FourDSemanticMappingAdapter",
    "Fscd147Adapter",
    "GenericFolderAdapter",
    "GreenCitrusAdapter",
    "OmniCountAdapter",
    "YoloDatasetAdapter",
    "build_adapter",
    "build_registry",
    "export_dataset_summary",
    "validate_adapter",
]

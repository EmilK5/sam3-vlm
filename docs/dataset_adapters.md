# Canonical Dataset Adapters

Phase 10 normalizes all experiment datasets into `CanonicalSample` objects.
Every box uses global-frame `xyxy` pixels. Labels remain evaluation-only and
are never passed to SAM3, Qwen, or the ASHT controller.

## Adapters

- `GreenCitrusAdapter`: YOLO citrus annotations.
- `CarpkAdapter`: `Images/Annotations/ImageSets` CARPK layout.
- `Fscd147Adapter`: standard FSC/FSCD-147 annotations, point counts, and exemplar boxes.
- `OmniCountAdapter`: COCO-format image/category samples.
- `FourDSemanticMappingAdapter`: manifest-driven or recursive RGB frame loading.
- `GenericFolderAdapter`: arbitrary image folders with optional JSON sidecars.
- `YoloDatasetAdapter`: reusable YOLO loader for additional datasets.

## Canonical fields

Each sample contains dataset identity, split, sample key, image path, target
concept, target class, confounders, count, boxes, masks, exemplar boxes, point
annotations, sequence identity, metadata, and explicit annotation
capabilities.

## Registry and validation

`DatasetRegistry` is the only lookup interface used by the experiment runner.
`DatasetAdapterSpec` and `build_registry` construct adapters from configuration.
`validate_adapter` checks every sample, duplicate keys, files, and box bounds.
`export_dataset_summary` writes deterministic split summaries for experiment
setup and debugging.

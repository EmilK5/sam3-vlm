# Phases 9 and 10 Completion

## Phase 9

- Added an immutable unified experiment configuration.
- Added discovery pass specifications and eight pipeline modes.
- Added one experiment runner and one run directory per sample.
- Added provenance-complete single-pass, fixed-multipass, and adaptive-tiling execution.
- Integrated existing static and Qwen ASHT runners.
- Added a provenance-aware legacy policy adapter boundary.
- Added input-image materialization for immutable run manifests.
- Added deterministic configuration hashing, environment capture, failure finalization, and final graph artifacts.

## Phase 10

- Added the canonical sample and dataset capability contracts.
- Added adapters for green citrus, CARPK, FSCD-147, OmniCount, 4D semantic mapping, generic folders, and generic YOLO data.
- Added configuration-driven adapter construction and a dataset registry.
- Added dataset integrity validation and summary export.
- Added synthetic-layout tests for every adapter family.

All Phase 1-10 tests pass together.

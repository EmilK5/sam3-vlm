# Phase 3: Explicit Pipeline Stages

The new staged path is additive and does not change the legacy `execute_pass`
behavior. `pipeline.execute_staged_pass` is the entry point for the future ASHT
runner.

## Stage boundaries

1. `Sam3QuerySpec` freezes every sensing parameter before the model call.
2. `execute_sam3_request` performs exactly one SAM3 call and emits the call and
   raw detections when a provenance context is supplied.
3. `normalize_sam3_output` validates index alignment across boxes, scores, and
   masks.
4. Coordinate translation is explicit in each `DetectionBatch`; both local and
   image-global boxes are retained.
5. `suppress_detection_batch` executes intra-pass NMS, preserves source indices,
   and records suppressed detections and their best surviving match.
6. `register_detection_batch` performs cross-pass comparisons and is the only
   stage in this path allowed to mutate the graph.
7. Semantic verification is intentionally absent from the staged discovery
   path. It is handled by the ASHT belief update in Phase 4.

## Compatibility

The original `execute_pass`, `run_inference_block`, global engine, tiled engine,
and IoC/VIP verification paths remain unchanged. Existing callers continue to
receive `PassStats`.

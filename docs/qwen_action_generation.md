# Qwen action-generation contract

Qwen returns one JSON object containing `reasoning_summary` and `actions`.
Every action may contain:

- `family`;
- `prompt`;
- `semantic_key`;
- `beta_by_class`;
- `threshold`;
- `region` in global XYXY pixels;
- positive and negative exemplar node IDs;
- `tiling_mode`;
- `tile_scale`;
- `expected_cost`;
- `rationale`;
- `expected_visual_distinction`.

The generator clips regions to the image and rejects verification regions that
do not intersect the target node. Exemplar IDs must refer to graph nodes that
already satisfy the configured posterior-confidence threshold. Duplicate or
previously used semantic keys are rejected.

A generation failure never bypasses provenance. The exact response, parse
failure, rejected payload, retry count, and fallback selection are all stored.

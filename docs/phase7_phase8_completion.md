# Phases 7 and 8 completion

## Phase 7: Qwen candidate-action generation

The ASHT controller can now use Qwen as a finite candidate-set generator while
retaining numerical action selection in Python.

Implemented behavior:

- exact system/user prompt capture;
- image, node, graph, posterior, query-history, and class context in the request;
- strict JSON parsing, including fenced JSON responses;
- target, negative, and exploration action families;
- action-level prompts, semantic keys, class descriptor probabilities, ROIs,
  exemplar references, thresholds, tiling preferences, expected costs, concise
  rationales, and expected visual distinctions;
- validation of class keys, probabilities, regions, target intersection,
  exemplar trust, semantic duplication, thresholds, and tiling values;
- retry accounting;
- exact raw and parsed response storage;
- token and latency storage;
- rejected-payload and validation-error storage;
- static-bank fallback;
- unchanged EIG ranking, SAM3 execution, Bayesian update, and stopping logic.

`QwenAshtRunner` uses the same mathematical and sensing path as the static
runner. Qwen proposes actions only; it never selects the winner and never adds a
detection to the graph.

## Phase 8: density-aware ROI tiling

A standalone adaptive tiling subsystem has been integrated into bootstrap
candidate discovery and targeted verification.

The implementation follows the public SAM3Count image pipeline's documented
structure and released inference logic:

1. perform a normal SAM3 pass;
2. suppress and clean the initial detections;
3. estimate density from object coverage, count, and average object size;
4. select `LARGE`, `MEDIUM`, or `SMALL` tiling parameters;
5. build a padded union ROI from stage-1 detections;
6. generate overlapping tiles and retain ROI-intersecting tiles;
7. execute SAM3 independently on every selected tile;
8. translate every tile detection to image-global coordinates;
9. suppress duplicates across tiles;
10. continue through the existing staged suppression, registration, or
    verification path.

Supported modes:

- `off`;
- `always`;
- `density_adaptive`;
- `agent_controlled`.

Every decision records its density inputs, thresholds, rule, reason, selected
ROI, tile size, overlap, stride, base tile count, selected tile count, budget
truncation, and every tile-level SAM3 call and detection.

The implementation is training-free and does not depend on a SAM3Count model
checkpoint.

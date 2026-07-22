# Phases 3 and 4 Completion

## Phase 3

- Added typed request and detection-batch boundaries.
- Added provenance-aware SAM3 call normalization.
- Added local/global coordinate preservation.
- Added lineage-preserving intra-pass suppression.
- Added auditable cross-pass graph association and registration.
- Added `pipeline.execute_staged_pass` while leaving legacy `execute_pass`
  unchanged.

## Phase 4

- Added patch priors and posterior state.
- Added surrogate observation-kernel construction.
- Added finite observation encoding.
- Added Bayes updates, entropy, EIG, realized KL, and entropy change.
- Added action ranking and optional cost adjustment.
- Added confidence stopping and declaration.
- Added hard and soft count estimates with variance.
- Added canonical records for every mathematical calculation.

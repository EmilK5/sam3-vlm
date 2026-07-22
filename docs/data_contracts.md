# Canonical Provenance Data Contracts

Schema version: `1.0.0`

This package defines the immutable records that later phases use for complete
run provenance.  Phase 1 is additive: no existing runner, graph, policy, SAM3
call, Qwen call, or evaluator imports these contracts yet.

## Design rules

1. Every record is a frozen dataclass derived from `CanonicalRecord`.
2. Every serialized object contains `schema_version` and `record_type`.
3. Serialization is strict JSON: NaN and Infinity are rejected.
4. Nested records preserve their own type tags and round-trip through JSON.
5. Every pipeline entity receives a namespaced run-scoped ID.
6. Large binary data is referenced through `ArtifactRef`; masks may also store
   JSON-compatible RLE or polygon data in their producing records.
7. Records contain only data.  They do not call models or mutate the graph.

## Identifier model

A run ID is globally unique and sortable:

- `run_<UTC timestamp>_<random token>`

All child IDs are scoped to that run and monotonically numbered:

- `pass_<run token>_000001`
- `action_<run token>_000001`
- `qwen_<run token>_000001`
- `sam3_<run token>_000001`
- `tile_<run token>_000001`
- `det_<run token>_000001`
- `node_<run token>_000001`
- `dedup_<run token>_000001`
- `registration_<run token>_000001`
- `observation_<run token>_000001`
- `kernel_<run token>_000001`
- `ig_<run token>_000001`
- `belief_<run token>_000001`

`IdFactory.snapshot()` records counters so an interrupted run can resume
without reusing an ID.

## Record map

### Run context

- `RepositoryStateRecord`
- `EnvironmentRecord`
- `ModelIdentityRecord`
- `ArtifactRef`
- `DatasetSampleRecord`
- `RunRecord`

### Actions and model calls

- `SensingActionRecord`
- `CandidateActionSetRecord`
- `QwenCallRecord`
- `Sam3CallRecord`
- `TileRecord`

### Detection and graph lineage

- `RawDetectionRecord`
- `DedupComparisonRecord`
- `RegistrationDecisionRecord`
- `GraphNodeSnapshotRecord`

### Mathematical state

- `EncodedObservationRecord`
- `SurrogateKernelRecord`
- `InformationGainRecord`
- `BeliefUpdateRecord`
- `StoppingDecisionRecord`

### Reporting and evaluation

- `CostSnapshotRecord`
- `EvaluationResultRecord`
- `ErrorEventRecord`
- `PassRecord`
- `EventEnvelope`

## Mathematical correspondence

| Formulation object | Contract |
|---|---|
| Candidate patch `p` | `GraphNodeSnapshotRecord` |
| Posterior `rho_p(t)` | `GraphNodeSnapshotRecord.posterior` |
| Action `a=(b,s,e,gamma)` | `SensingActionRecord` |
| Candidate set `C_t` | `CandidateActionSetRecord` |
| SAM3 result | `Sam3CallRecord` + `RawDetectionRecord` |
| Finite observation `Y_t` | `EncodedObservationRecord` |
| Surrogate kernel `q_hat_m^a` | `SurrogateKernelRecord` |
| Expected information gain | `InformationGainRecord` |
| Bayesian update | `BeliefUpdateRecord` |
| Stopping and declaration | `StoppingDecisionRecord` |
| Complete iteration | `PassRecord` |
| Complete experiment | `RunRecord` |

## Phase 2 integration boundary

Phase 2 should add builders and writers around these records:

1. Create the run directory and initial `RunRecord` context.
2. Append `EventEnvelope` objects to `events.jsonl`.
3. Write large artifacts and produce `ArtifactRef` records.
4. Build a `PassRecord` from all events in one pass.
5. Consolidate the event stream into the final `RunRecord`.
6. Save atomically and preserve partial runs on failure.

The contract package must remain free of model and pipeline imports so it can be
used by the sandbox, evaluators, replay tools, and offline analysis scripts.


## Phase 8 additions

- `TilingDecisionRecord`: density inputs, trigger mode, selected ROI, grid rule, tile budget, and thresholds.
- `TileRecord`: per-tile coordinates, transforms, SAM3 calls, and detections.
- `PassRecord.tiling_decision`: optional pass-level tiling decision reference.

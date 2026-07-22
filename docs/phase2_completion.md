# Phase 2 completion: provenance storage and recovery

Phase 2 implements the append-only provenance and run-directory layer planned
after the Phase 1 canonical data contracts.

## Added

- strict JSON and atomic filesystem utilities;
- append-only event writer and validated event reader;
- truncated-tail repair with hard failure on earlier corruption;
- event-kind inference for all Phase 1 record types;
- deterministic event consolidation into `RunRecord`;
- `RunStore` creation, reopening, checkpointing, success/failure/interruption
  finalization, summaries, and artifact management;
- reporting levels (`minimal`, `standard`, `full`);
- complete event embedding in final `run.json`;
- ID-counter recovery from event payloads written after the latest checkpoint;
- storage and recovery documentation;
- eleven new storage tests in addition to the Phase 1 tests.

## Intentionally not changed

- SAM3 execution;
- Qwen calls or policies;
- graph behavior;
- deduplication;
- pipeline orchestration;
- application UI;
- dataset loading;
- evaluation behavior.

Those integrations begin in later phases. Phase 2 is additive and provides the
storage surface they will call.

## Validation

- 23 tests pass.
- All provenance and test modules compile successfully.

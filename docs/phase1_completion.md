# Phase 1 Completion Checklist

- [x] Versioned schema constant.
- [x] Strict canonical JSON serialization.
- [x] Recursive deserialization and record registry.
- [x] Stable IDs for run, pass, action, model calls, detections, graph nodes,
      deduplication, registration, observations, kernels, beliefs, artifacts,
      evaluations, errors, and events.
- [x] Run-level records.
- [x] Action and Qwen/SAM3 call records.
- [x] Detection, tiling, deduplication, assignment, and graph records.
- [x] Observation-kernel, information-gain, posterior-update, and stopping
      records.
- [x] Pass and full-run aggregates.
- [x] Cost, evaluation, error, and artifact records.
- [x] Validation for boxes, probabilities, normalized posteriors, kernel
      dimensions, costs, and ID namespaces.
- [x] Unit tests for ID generation, counter resume, strict JSON, nested
      round-trip, polymorphic events, and invalid values.
- [x] Existing pipeline behavior left untouched.

Phase 2 may now implement the run-directory manager and append-only event
writer against these contracts.

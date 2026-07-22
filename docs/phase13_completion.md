# Phase 13: validation, reproducibility, and regression completion

Phase 13 closes the implementation plan with a testable release boundary.

## Added validation tools

`validation.run_validator` audits a self-contained run directory for:

- required files and strict canonical decoding;
- configuration, manifest, summary, and image-hash consistency;
- exact `events.jsonl` / `run.json` parity;
- contiguous event counts and sequences;
- artifact existence, size, and SHA-256 integrity;
- pass, action, SAM3 call, tile, detection, deduplication, registration,
  observation, kernel, belief, stopping, and evaluation lineage;
- reconstruction of the final graph artifact and comparison with the final
  canonical graph snapshots.

`validate_experiments.py` exposes three commands:

```text
python validate_experiments.py validate PATH_TO_RUN
python validate_experiments.py scan PATH_TO_OUTPUT_ROOT
python validate_experiments.py compare RUN_A RUN_B
```

The commands return nonzero exit status for invalid runs or semantically
nonmatching comparisons.

## Reproducibility fingerprints

`validation.reproducibility` creates a semantic run trace that retains the
resolved experiment, actions, detections, graph transitions, kernels,
posteriors, and results while normalizing only run IDs, timestamps, latencies,
hostnames, and storage paths. This permits deterministic mocked runs to be
compared even when executed at different times with different run IDs.

## Configuration completeness

The resolved configuration now records:

- all operational unified-runner settings;
- the exact per-sample dataset identity;
- backend, processor, and legacy-adapter types;
- explicit runtime configuration supplied by the project;
- Qwen generator settings, sensor profile, client type, and fallback action
  bank;
- complete ASHT observation, bootstrap, action-bank, and adaptive-tiling
  settings.

Resume matching and final run creation use the same
`resolved_config_for_sample` method, preventing a run from being skipped under
one hash and written under another.

## Schema migration

`provenance.migrations` provides recursive migration from the development
`0.9.0` canonical payload format to stable schema `1.0.0`. Unknown schema
versions remain hard errors.

## Regression and integration coverage

The suite exercises:

- all eight unified experiment modes;
- static and Qwen ASHT controllers with mocked sensors;
- density and agent-controlled tiling;
- provenance recovery and interrupted-run finalization;
- complete citrus and CARPK adapter runs on synthetic official-style layouts;
- saved graph replay and artifact tamper detection;
- deterministic semantic fingerprints;
- resume-safe sweeps;
- legacy `execute_pass` compatibility without loading heavyweight models.

`pytest.ini` makes the repository importable under plain `pytest`, and the CI
workflow runs the dependency-light core suite on Python 3.11 and 3.12.

Real SAM3/Qwen checkpoint smoke tests remain machine-specific and should be
run in the project environment after this patch is applied.

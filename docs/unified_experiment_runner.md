# Unified Experiment Runner

Phase 9 introduces one run boundary for every supported pipeline. The runner
resolves a dataset sample, copies the input image into a self-contained run
directory, creates one `RunStore`, dispatches the selected pipeline, and
finalizes the same canonical `run.json` contract.

## Supported modes

- `sam3_single_pass`
- `sam3_fixed_multipass`
- `sam3_adaptive_tiling`
- `legacy_vlm`
- `asht_static`
- `asht_qwen`
- `asht_adaptive_tiling`
- `asht_agent_tiling`

Discovery-only modes use `DiscoveryExperimentExecutor`. Static and Qwen ASHT
modes use the existing Phase 6-8 controllers. The legacy VLM mode uses an
injected provenance-aware adapter so the old policy is not hard-coded into the
new experiment core.

## Run lifecycle

1. Resolve one `CanonicalSample` through `DatasetRegistry`.
2. Freeze the resolved configuration and hash it.
3. Declare the input image in the immutable run manifest.
4. Create `RunStore` and materialize the declared image artifact.
5. Dispatch the selected pipeline mode.
6. Save the final graph as an artifact.
7. Finalize `run.json`, `summary.json`, and the complete event stream.
8. On an exception, finalize a failed run before optionally re-raising.

The unified runner contains no dataset-specific branching. Dataset format
knowledge is isolated in Phase 10 adapters.

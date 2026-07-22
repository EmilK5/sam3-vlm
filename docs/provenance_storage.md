# Provenance storage and recovery

Phase 2 adds a crash-safe storage layer around the canonical records defined in
Phase 1. It does not alter any existing SAM3, Qwen, graph, policy, or evaluation
runtime behavior.

## Run directory

Each experiment owns one directory named by its stable `run_id`:

```text
<output_root>/<run_id>/
  manifest.json
  events.jsonl
  config.json
  environment.json
  run.partial.json
  run.json
  summary.json
  checkpoints/
    latest.json
  artifacts/
```

- `manifest.json` is the immutable starting `RunRecord`.
- `events.jsonl` is the append-only source of truth while a run is active.
- `run.partial.json` and `checkpoints/latest.json` are atomic running snapshots.
- `run.json` is written only when the run succeeds, fails, or is explicitly
  marked interrupted. It embeds the complete event stream.
- `summary.json` is a lightweight status and metrics index.
- `artifacts/` contains masks, crops, overlays, logs, and other large files.

## Durability model

Every event is serialized as one strict JSON object and appended with `O_APPEND`.
The writer optionally calls `fsync` after each append. Snapshot files are written
to a temporary file in the destination directory and installed with
`os.replace`.

If a process stops halfway through the final JSONL append, recovery removes only
that malformed final line. A malformed earlier line is treated as corruption and
raises an error rather than silently discarding data.

## Event and ID recovery

Events have contiguous sequence numbers and unique event IDs. Reopening a run
scans the valid event stream, restores the next event sequence, and observes all
run-scoped entity IDs inside event payloads. This prevents action, node,
detection, tile, and other IDs written after the latest checkpoint from being
reused after a crash.

## Consolidation

`consolidate_run` deterministically folds the manifest and event stream into a
`RunRecord`. It collects completed passes, latest graph-node snapshots,
evaluations, errors, and every referenced artifact. Running checkpoints omit the
embedded event tuple because `events.jsonl` is authoritative. Final `run.json`
includes the full event tuple so all structured information is available from a
single JSON file.

## Reporting levels

The store supports `minimal`, `standard`, and `full` reporting levels. Each event
append may declare its minimum required level. Research experiments should use
`full`; the lower levels exist for lightweight debugging and production-style
runs.

## Artifact handling

Files and byte payloads can be copied into the run directory through the store.
Each resulting `ArtifactRef` records the relative path, media type, SHA-256 hash,
byte size, optional dimensions, and metadata. Artifact references are also
written to the event stream.

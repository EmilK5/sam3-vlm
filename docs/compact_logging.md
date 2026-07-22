# Compact research logging

The default provenance format is organized around three useful questions:

1. What happened in each SAM3 pass?
2. What was sent to and returned by Qwen?
3. When was each graph object created or updated?

`run.json` contains compact pass records, Qwen calls, final object snapshots, and
metrics.  It never embeds the append-only event stream.  `events.jsonl` remains
available for crash recovery, but normally contains one event per completed pass
plus artifact, evaluation, and error events.

The default log does not contain every failed pairwise deduplication comparison,
full graph snapshots before and after each operation, mask pixels, image arrays,
or logits.  Deduplication records contain only the comparison that caused a
suppression or assignment.  Masks are stored as compressed `.npz` artifacts and
referenced from the final graph JSON.

Reporting levels:

- `minimal`: final results, errors, and essential artifacts.
- `standard`: compact research log and the default.
- `full`: the same compact structure with all useful model and mathematical fields.
- `debug`: reserved for deliberately verbose diagnostics.

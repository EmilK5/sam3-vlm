# Phases 11 and 12: dashboard and evaluation

## Focused experiment dashboard

`experiment_dashboard.py` replaces the old low-level orchestration sandbox for
new experiments. It delegates all execution to `UnifiedExperimentRunner` and
all dataset access to the canonical adapter registry.

The main panel exposes only research-level choices:

- dataset, split, and sample;
- pipeline mode;
- target-concept override;
- reporting level;
- pass, SAM3-call, and Qwen-call budgets;
- posterior stopping confidence;
- tiling mode;
- run name.

Deduplication and suppression internals are not normal controls. A factory may
accept explicit expert overrides through the closed advanced JSON panel, and
those values are captured in the resolved run configuration.

The dashboard contains:

- complete-pipeline execution;
- input/final overlay;
- pass timeline;
- object lineage and posterior table;
- Qwen request/result summaries;
- EIG ranking and realized-information views;
- tiling decisions;
- replay of a saved run without model calls;
- pass-by-pass replay;
- comparison of saved runs;
- searchable canonical `run.json`.

### Launch contract

Model construction and ASHT presets are machine-specific, so the entry point
requires a project factory:

```text
python experiment_dashboard.py --factory my_dashboard_config:create_dependencies
```

The factory returns `dashboard.DashboardDependencies`, containing the dataset
registry, unified runtime, configuration factory, and output root. Gradio is an
optional dependency and is imported only when the dashboard is launched.

## Per-run evaluation

The unified runner now evaluates each successful run before finalization and
stores an `EvaluationResultRecord` in the canonical event stream. Depending on
available annotations, it records:

- hard and soft count error;
- absolute, squared, and normalized count error;
- exact-count accuracy;
- box precision, recall, and F1;
- candidate-pool recall;
- Brier score and negative log-likelihood when node labels are observable;
- pass count;
- selected EIG totals and means;
- realized-information totals and means.

Each run may save:

- `artifacts/visualizations/final_overlay.png`;
- `artifacts/visualizations/pass_timeline.png`;
- `artifacts/graph/final_graph.json`;
- canonical `run.json` and `summary.json`.

A root-level `runs.csv` is updated atomically. Failure to update this shared
index does not invalidate a completed canonical run.

## Aggregation and resume

`RunIndex` scans canonical runs and considers a result reusable only when all
of these match:

- successful status;
- dataset and split;
- sample key;
- image hash;
- resolved configuration hash.

`eval.sweep.run_sweep` uses that index for resume-safe execution.
`eval.reporting.aggregate_runs`, `write_aggregate`, and
`write_aggregate_csv` produce experiment-level tables and summaries directly
from saved runs.

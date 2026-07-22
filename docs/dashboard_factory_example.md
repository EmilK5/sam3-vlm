# Dashboard factory contract

A dashboard factory is intentionally project-specific. It must return a
`DashboardDependencies` object with:

1. a canonical `DatasetRegistry`;
2. a loaded `UnifiedRuntime`;
3. a configuration factory;
4. the directory containing experiment runs.

The configuration factory receives a `DashboardRunRequest` and the selected
`CanonicalSample`. It is responsible for choosing the frozen experiment preset
and applying only the explicit dashboard overrides. This keeps model paths,
Qwen credentials, calibrated kernels, and dataset-specific presets out of the
UI implementation.

The main UI does not expose deduplication thresholds. When expert overrides are
accepted, the factory must copy them into `resolved_overrides` so they are
visible in `config.json` and `run.json`.

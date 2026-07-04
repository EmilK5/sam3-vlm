# PROGRESS

Status legend: `[ ]` todo · `[~]` implemented, awaiting human verification ·
`[x]` verified by human (ONLY the human flips to `[x]`).

Full step specifications live in `docs/implementation_plan.md`. Do not start a
step whose predecessor in the same phase is still `[ ]`. Cross-phase order may
follow the "Suggested order of the first week" in the plan.

## Phase 0 — Scaffold & harness
- [x] 0.1 Package layout + config.py — check: `pytest -q` green; `python -c "from config import Config; print(Config())"` prints all grouped fields with sane defaults.
- [x] 0.2 CLI runner for the existing cascade (scripts/run_image.py, graph.to_dict) — check: run on a real orchard image with `--passes 3`; overlay resembles Gradio output; `out/<stem>_graph.json` opens and node count matches the logged pass stats.
- [ ] 0.3 Dataset loader (eval/datasets.py)

## Phase 1 — FM+V-IP verifier
- [x] 1.1 Query sets + generation script (verifier/queries.py, queries/green_citrus.json) — check: read queries/green_citrus.json by hand — every query answerable from a 256px crop; templates (T/D/S) match your intuition; edit freely. `pytest -q` green.
- [x] 1.2 Oracles (MockOracle, QwenOracle, Sam3Oracle) — check: `pytest -q tests/test_oracle.py`; then a REPL smoke test with a real QwenOracle on one fruit crop — answers visibly sane (round=yes, veins=no).
- [x] 1.3 Training-free V-IP core (verifier/vip.py, pure numpy) — check: `pytest -v tests/test_vip.py` — read the four test names; they are the spec.
- [x] 1.4 Verify API + demo (verifier/verify.py, scripts/demo_verify.py)  ← GO/NO-GO checkpoint — check: run demo on 3 hand-picked boxes (clear fruit, clear leaf, junk) with `--oracle qwen`; the three chains should read like sensible reasoning. Tweak queries/epsilon in the JSON and re-run. This is where you judge whether the whole idea works.
- [ ] 1.5 Pipeline integration behind verifier="vip" flag

## Phase 2 — Belief state
- [ ] 2.1 Support, jitter, signatures on nodes (graph.py, pipeline dedup branch)
- [ ] 2.2 belief.py: w_i, U, discovery curve, phi summary

## Phase 3 — Actions & cost
- [ ] 3.1 Action layer (agent/actions.py)
- [ ] 3.2 Cost meter (agent/budget.py)

## Phase 4 — Heuristic controller
- [ ] 4.1 VoI heuristic policy + episode runner

## Phase 5 — VLM orchestrator
- [ ] 5.1 Scene inspection z_t (agent/inspect.py)
- [ ] 5.2 VLM policy with strict validation (agent/policy_vlm.py)

## Phase 6 — Evaluation
- [ ] 6.1 Matching & detection metrics (eval/matching.py)
- [ ] 6.2 Sweep runner (eval/run_eval.py)
- [ ] 6.3 Results table & accuracy-vs-cost plot (eval/report.py)

## Notes / decisions log
<!-- Append dated one-liners here when a step deviates from the plan. -->
- 2026-07-04 (0.1): Config is a flat dataclass with grouped comments (not nested
  dataclasses) per the plan's field list; fields prefixed to avoid clashes
  across groups (e.g. `verifier_mode`, `oracle_base_url`, `oracle_model_name`).
  `oracle_base_url`/`oracle_model_name` default from `QWEN_BASE_URL`/`QWEN_MODEL`
  env vars; `QWEN_API_KEY` is intentionally NOT stored in Config (never logged/
  printed) and will be read directly from env by QwenOracle in step 1.2.
  `delta_U` had no default value specified in the plan prompt; picked `0.5` as
  a placeholder pending tuning in Phase 2/4.
- 2026-07-04 (0.2): This dev/test environment has no torch/transformers/scipy/
  gradio/openai installed (confirmed via `pip list`); `import inference` fails
  here. `scripts/run_image.py` is therefore syntax-checked only (`py_compile`),
  not executed or imported by pytest — matches the plan, which lists no test
  file for this step. Added `tests/test_graph_to_dict.py` (CPU-only, no torch)
  to satisfy the "one pytest file per step" ground rule by covering the one
  new pure piece of logic, `OrchardGraph.to_dict()`. `--conf` CLI default
  reads from `Config().conf` instead of hardcoding 0.35 again.
- 2026-07-04 (1.1): Added `class_names` as a QuerySet field + validation
  (plan's dataclass field list omitted it, but the authoritative JSON schema
  includes it and step 1.5 needs the target->fruit mapping). Also validate
  `epsilon in (0, 0.5)` per the proposal's noise model, and that each query's
  templates cover EXACTLY the declared classes (not just "all present"). Ordering
  deviation: did 1.1 before 0.3 per the plan's "Suggested order of the first week"
  and an explicit human choice. `scripts/gen_queries.py` imports `openai` lazily
  inside `main()` (not installed in this dev env); syntax-checked only, no test.
- 2026-07-04 (1.3): Did 1.3 before 1.2 per the plan's suggested week-1 order and
  an explicit human choice; `vip.py` is pure numpy with no dependency on the
  oracles (1.2). Tests generate the clean noise-free answer vectors directly
  from `query_set` templates (identical to what `MockOracle(noise=0)` will
  return in 1.2), keeping 1.3 self-contained. Entropies computed in nats
  (natural log) — internally consistent, does not affect argmax. `run_ip` also
  stops early when the best remaining info gain is <= 1e-12 (no informative
  query left), which guarantees an all-zero-template query is never selected.
- 2026-07-04 (verify 0.1-1.3): At the human's request during verification,
  hardened test coverage across all completed steps (test files only, no source
  changes): test_smoke +5 (env wiring, no api_key field, costs/agent fields,
  ioc default, unknown-key rejection), test_graph_to_dict +2 (JSON round-trip,
  leaf verdict), test_queries +7 (extra/empty/non-dict templates, class_names
  gap, epsilon boundaries, no dead queries), test_vip +6 (row sums, uniform
  uninformative row, CMI >= 0, empty-S == prior, max_q cap, prior-driven
  verdict), and new test_run_image.py (+3) exercising the argparse/path helpers
  via sys.modules stubs for torch/inference/pipeline. Suite: 14 -> 39, all green.
- 2026-07-04 (1.2): Did 1.2 after 1.3 per the plan's suggested week-1 order.
  `torch` and `openai` are imported LAZILY inside the methods that need them
  (Sam3Oracle._presence_score / QwenOracle._client), so oracle.py imports on a
  CPU box with neither installed and Mock/Qwen stay testable offline. QwenOracle
  takes an optional injectable `client` for offline testing (the canned fake).
  Sam3Oracle reads `sam3_phrase` via getattr(query, ...) — the Query dataclass
  (step 1.1) has no such field and 1.2 may not touch queries.py, so with the
  current query sets Sam3Oracle returns all-zeros and calls no model; a later
  step can add the field to Query. MockOracle "flip" = replace the clean
  template answer with a uniformly-chosen other value in {-1,0,1} (noise=1.0
  flips every query); Sam3Oracle presence answer = +1 if score>=tau else -1.
- 2026-07-04 (0.2 bugfix): `scripts/run_image.py` imported `config` BEFORE adding
  the repo root to sys.path, so `python scripts/run_image.py` (script invocation,
  which puts scripts/ — not the repo root — on sys.path) failed with
  "No module named 'config'". Moved the repo-root sys.path.append above the
  config import (matching demo_verify.py/gen_queries.py). Added a subprocess
  regression test (test_run_image.py) that runs the script from a foreign cwd;
  the prior unit tests missed it because pytest already has the repo root on
  the path. Fix is confined to 0.2's own file; step left [x].
- 2026-07-04 (1.4): `extract_crop` returns a PIL image (oracles consume crop_pil;
  mirrors Image.fromarray in verify_box_semantics) and guards degenerate/zero-area
  boxes so cv2.resize never fails. Prior is uniform (proposal allows uniform or
  confidence-derived; plan unspecified). Epsilon comes from query_set.epsilon via
  vip.likelihood_table (what 1.3 tested), NOT cfg.vip_epsilon — cfg.vip_epsilon is
  currently unused/redundant; flag for later reconciliation. Sequential mode
  re-implements the greedy loop here (interleaving oracle calls) using vip
  primitives; batched uses vip.run_ip. demo_verify displays the winning class's
  posterior mass max(posterior), not p_target, so a non-target verdict reads
  sensibly (e.g. "-> distractor (0.92)").

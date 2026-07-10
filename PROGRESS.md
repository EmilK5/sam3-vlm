# PROGRESS

Status legend: `[ ]` todo · `[~]` implemented, awaiting human verification ·
`[x]` verified by human (ONLY the human flips to `[x]`).

Full step specifications live in `docs/implementation_plan.md`. Do not start a
step whose predecessor in the same phase is still `[ ]`. Cross-phase order may
follow the "Suggested order of the first week" in the plan.

## Phase 0 — Scaffold & harness
- [x] 0.1 Package layout + config.py — check: `pytest -q` green; `python -c "from config import Config; print(Config())"` prints all grouped fields with sane defaults.
- [x] 0.2 CLI runner for the existing cascade (scripts/run_image.py, graph.to_dict) — check: run on a real orchard image with `--passes 3`; overlay resembles Gradio output; `out/<stem>_graph.json` opens and node count matches the logged pass stats.
- [x] 0.3 Dataset loader (eval/datasets.py) — check: `python -m eval.datasets --root <yolo-split> --fmt yolo`; the saved GT overlay boxes sit on real fruit. Also `run_image.py --draw-gt` saves out/<stem>_gt.jpg.

## Phase 1 — FM+V-IP verifier
- [x] 1.1 Query sets + generation script (verifier/queries.py, queries/green_citrus.json) — check: read queries/green_citrus.json by hand — every query answerable from a 256px crop; templates (T/D/S) match your intuition; edit freely. `pytest -q` green.
- [x] 1.2 Oracles (MockOracle, QwenOracle, Sam3Oracle) — check: `pytest -q tests/test_oracle.py`; then a REPL smoke test with a real QwenOracle on one fruit crop — answers visibly sane (round=yes, veins=no).
- [x] 1.3 Training-free V-IP core (verifier/vip.py, pure numpy) — check: `pytest -v tests/test_vip.py` — read the four test names; they are the spec.
- [x] 1.4 Verify API + demo (verifier/verify.py, scripts/demo_verify.py)  ← GO/NO-GO checkpoint — check: run demo on 3 hand-picked boxes (clear fruit, clear leaf, junk) with `--oracle qwen`; the three chains should read like sensible reasoning. Tweak queries/epsilon in the JSON and re-run. This is where you judge whether the whole idea works.
- [x] 1.5 Pipeline integration behind verifier="vip" flag — check: `scripts/run_image.py --passes 2` twice per verifier mode, diff the two overlays + graph JSONs, spot-check 5 differing nodes with demo_verify. NOTE: run_image.py has no --verifier flag yet (out of 1.5's file scope); see handoff for how to exercise the vip path meanwhile.

## Phase 2 — Belief state
- [x] 2.1 Support, jitter, signatures on nodes (graph.py, pipeline dedup branch) — check: 3-pass run; print top-10 nodes by `support` from the graph JSON — stable fruits should have support>=2, one-off junk support==1.
- [x] 2.2 belief.py: w_i, U, discovery curve, phi summary — check: pytest; then print φ (belief.summarize) after each pass in run_image — U (belief.uncertainty) should visibly drop across passes on an easy image.

## Phase 3 — Actions & cost
- [x] 3.1 Action layer (agent/actions.py) — check: REPL execute one QueryA on a quadrant of a real image (build ActionContext, execute(QueryA(region=quadrant,...), ctx)); overlay shows detections only inside that quadrant.
- [x] 3.2 Cost meter (agent/budget.py) — check: 2-pass run prints a cost line (CostMeter.total); recount by hand from the logs once.

## Phase 4 — Heuristic controller
- [x] 4.1 VoI heuristic policy + episode runner (agent/policy_heuristic.py, agent/runner.py) — check: run episodes on 3 images (sparse, dense, empty-of-fruit); read the JSON action traces — dense should trigger TileQuery/Subdivide, empty should stop within ~3 actions.

## Phase 5 — VLM orchestrator
- [x] 5.1 Scene inspection z_t (agent/inspect.py) — check: run inspect_scene on a dense-canopy image and a sparse one; compare the two JSONs against your own eyes.
- [x] 5.2 VLM policy with strict validation (agent/policy_vlm.py) — check: run a full VLM episode with prompt logging; read one full prompt+response pair; confirm every executed action was validated (grep the policy_vlm logs for "fallback"). NOTE: the VLM episode loop (inspect->z, log-prompts wiring) is not in 5.2's files; see handoff.

## Phase 6 — Evaluation
- [x] 6.1 Matching & detection metrics (eval/matching.py) — check: `pytest -q tests/test_matching.py`; then run matching on one real image vs GT — draw matched GT green, missed red, save to out/ (manual script).
- [x] 6.2 Sweep runner (eval/run_eval.py) [REDUCED: oneshot/cascade/tiled/convergence only] — check: `--limit 5` on citrus for {oneshot,cascade}x{ioc,vip}; open CSV — pool_recall(cascade) >= pool_recall(oneshot); vip-vs-ioc precision delta shows if the verifier earns its cost.
- [x] 6.3 Results table & accuracy-vs-cost plot (eval/report.py) — check: run report on your sweep CSV(s); the accuracy-vs-cost plot is the paper figure — confirm the policy story (one-shot → cascade → tiled → convergence → heuristic → VLM) is readable.

## Phase 7 — Guided-ROI policy rework
- [x] 7.1 LookROIA action + grounding enforcement (agent/actions.py, tests/test_look_roi.py) — implemented; tests/test_look_roi.py green (8). Check: REPL LookROIA on a dense corner of a real image; overlay shows SAM3 detections only inside the 10%-expanded box; the ROI box itself is not a node.
- [x] 7.2 Bootstrap global pass + runner auto-stop (agent/runner.py, tests/test_runner_bootstrap.py) — implemented; test green (4). Check: VLM episode's first log line is the global pass; a saturating run terminates without the model emitting stop.
- [x] 7.3 Guided-ROI VLM policy (agent/policy_vlm.py, agent/runner.py, eval/run_eval.py, tests/test_policy_vlm_roi.py; + tests/test_policy_vlm.py migration) — implemented; suite green (228). Check: run_eval --policy vlm --limit 1; after the bootstrap pass every look/tile/verify senses, no no-op spins, stops on saturation/budget.
- [x] 7.4 Config, thinking toggle, CLAUDE.md #3 reword (config.py, CLAUDE.md, agent/policy_vlm.py, tests/test_thinking_toggle.py) — implemented; suite green (232). Check: grep a policy-loop request log — disable-thinking arg present, no thinking trace; inspect/verify still think; pytest green. CAVEAT: verify the exact Ollama qwen3-vl disable-thinking key (see config.thinking_call_kwargs).

## Phase 8 — v2: full-history amortized policy (branch: v2)
- [x] 8.1 Action space v2: untiled look, delete tile/subdivide/verify (agent/actions.py, agent/policy_heuristic.py, tests/test_actions_v2.py + stale-test migration; scope widened w/ user approval to runner/policy_vlm/run_eval/orchestration_app ripple fixes) — check: REPL LookROIA logs a single global (not tiled) inference inside the expanded box; pytest green (264).
- [x] 8.2 Episode history + explicit 3-action bootstrap (agent/history.py, agent/runner.py, tests/test_history.py; migrated tests/test_runner_bootstrap.py) — implemented; suite green (271). Check: episode log starts canopy_roi/leaf_map/global_pass with a sane tree ROI box + leaf count; history length == log length.
- [x] 8.3 Full-history VLM policy + overlay image, retire inspect (agent/policy_vlm.py, agent/overlay.py, agent/runner.py; deleted agent/inspect.py + tests/test_inspect.py; new tests/test_policy_vlm_v2.py; migrated test_policy_vlm.py/test_policy_vlm_roi.py; trimmed 1 inspect test in test_review_fixes.py) — implemented; suite green (258). Check: one logged prompt carries raw+overlay images and the complete x/y history; no "fallback" lines on a healthy run.
- [x] 8.4 V-IP routing: CV + SAM3 channels, RouterOracle (verifier/queries.py, verifier/cv_answers.py, verifier/oracle.py, queries/green_citrus.json, tests/test_oracle_routing.py; scope: scripts/demo_verify.py --oracle router) — implemented; suite green (270). Check: demo_verify --oracle router shows ≤1 Qwen call per candidate; hand-review routed green_citrus.json.
- [~] 8.5 v2 wiring: config, eval, apps (config.py, eval/run_eval.py, orchestration_app.py, tests/test_v2_wiring.py; scope: tests/test_smoke.py for removed knobs) — implemented; suite green (277). runner.py needed no change. Check: run_eval --policy vlm --oracle router --limit 1 end-to-end.

## Notes / decisions log
<!-- Append dated one-liners here when a step deviates from the plan. -->
- 2026-07-10 (phase-8 review, user request): whole-phase code review found and
  fixed 2 bugs outside any step's file list: (1) citrus_orchestration_app.py
  still called the pre-8.3 make_vlm_policy(ctx, inspect_client=..., vlm_client=...)
  signature -> TypeError on its mock-VLM path; now (ctx, vlm_client=...).
  (2) orchestration_app.py ran VLM episodes WITHOUT the mandatory v2 bootstrap
  (violating the locked "3 bootstrap records always precede the policy" design;
  citrus app + run_eval both had it) -> run_episode now gets
  bootstrap_global_pass/auto_stop = (policy != "heuristic"). Suite green (277).
  Flagged-not-fixed runtime risks recorded in the review handoff: full-frame/16
  ROI area floor vs free VLM looks on high-res frames; guarded no-op looks are
  indistinguishable from zero-detection senses in the history; two full-res
  base64 images per policy call; per-look leaf-map regeneration (2 SAM3 calls
  per look); CSV n_actions counts the bootstrap as 3 records; untuned cv
  thresholds in green_citrus.json.
- 2026-07-10 (8.5): config.py removed small_area + c0 (only the deleted
  tile-menu heuristic used them; grep-confirmed dead) and added
  sam3_presence_tau=0.5 (promoted from the 8.4 getattr) + oracle_kind="qwen".
  run_eval: --oracle {qwen,router}; build_verifier gained oracle_kind+processor
  and builds RouterOracle(cfg, processor, vlm_oracle=QwenOracle) when selected;
  main() reordered to load SAM3 BEFORE build_verifier so the router gets the
  processor for its sam3 channel. orchestration_app.py: minimal fix of the
  broken make_vlm_policy(ctx, inspect_client=..., vlm_client=...) call ->
  (ctx, vlm_client=...) (inspect retired in 8.3). SCOPE WIDENED (forced):
  test_smoke.py dropped the small_area/c0 asserts. RESIDUALS left for a future
  cleanup (out of 8.5's scope): (a) c_inspect/n_inspect are now dead too
  (inspect deleted in 8.3) but still referenced by agent/budget.py +
  eval/metrics.py, so left in place (n_inspect is always 0 -> inert);
  (b) orchestration_app's _StubChatClient still emits a v1 "tile"/"assess" mock
  which now just falls back to the heuristic (harmless) -- a fuller v2 mock is a
  demo-only nicety. runner.py listed in the plan but needed no change (bootstrap/
  auto_stop + the router oracle already flow through run_eval -> ActionContext).
  Phase 8 code complete (8.1-8.5 all [~]/[x]).
- 2026-07-10 (8.4): four resolutions vs the plan text, all logged. (1) cv_check
  gains an optional "direction": "high"|"low" (default "high" == the plan's
  yes-above/no-below); "low" flips the answer sign so shape/texture queries
  whose "yes" is the LOW end (leaf-like = low circularity; smooth bg = low
  edge_density) can route to cv. (2) green_citrus.json gets NO sam3 route: it
  has no part-presence query, and adding one with a sam3_phrase would break the
  out-of-scope test_oracle.py assertion that the set is phrase-free — so the
  sam3 channel is exercised by a synthetic set in test_oracle_routing.py, and a
  part query is left for the human's JSON review. (3) SCOPE WIDENED (You-verify
  needs it, precedent 0.2/8.1): scripts/demo_verify.py gains --oracle router
  (RouterOracle with processor=None, vlm_oracle=QwenOracle) + a per-crop
  vlm/sam call-count print. (4) sam3 presence tau read via getattr(cfg,
  "sam3_presence_tau", 0.5) — promote to config.py in 8.5. RouterOracle exposes
  n_vlm_calls/n_sam_calls but wiring them into cost metering is NOT done here
  (verify.py/pipeline.py unchanged, still meter n_oracle_calls=1 per candidate);
  that reconciliation is a later step. green_citrus routing: q01/q03 circularity
  (high/low), q06 edge_density (low), q02/q04/q05 stay vlm.
- 2026-07-10 (8.3): policy_vlm.choose new signature is
  choose(phi, history, graph, cfg, client, image, sensed_rois) per the plan
  (partition dropped). The heuristic fallback needs a partition, so it is
  derived as [tree_roi] (the v2 partition IS the single tree ROI):
  tree_roi = graph.tree_roi, else the full frame when an image is present, else
  None. look is validated INSIDE tree_roi (not merely the frame). The prompt
  now sends TWO image parts (raw frame + agent/overlay.render_overlay drawing:
  candidate boxes by class, sensed ROIs blue, tree ROI magenta — PIL only).
  make_vlm_policy signature changed to (ctx, vlm_client=None) — inspect_client
  dropped; it reads ctx.history + ctx.sensed_rois via getattr. SCOPE WIDENED
  (same precedent as 8.1, out-of-scope but forced by the mandated deletion of
  agent/inspect.py): trimmed test_review_fixes.py's
  test_inspect_scene_survives_request_exceptions (its subject module is gone).
  orchestration_app.py / citrus_orchestration_app.py keep a now-dead "assess"
  branch in their mock-client dispatch (comment references agent.inspect, no
  import — no breakage); left for 8.5. run_eval calls make_vlm_policy(ctx)
  positionally, unaffected by the signature change.
- 2026-07-10 (8.2): two resolutions vs the plan text, both logged. (1) history
  new_nodes are computed by a pre/post node-id DIFF, not by filtering on
  found_in_pass — equivalent on the real pipeline but robust to stub executors
  that reuse a fixed pass number. (2) run_episode now decouples a per-record
  counter (one per log line / history record) from the sensing-budget counter
  (max_actions caps SENSING actions only): the bootstrap trio is 3 records but 1
  budget unit. ctx.history (EpisodeHistory) is set as a runtime attribute on
  ActionContext (not a declared field — actions.py is out of 8.2's scope; the
  dataclass isn't slotted so this is legal). run_episode return dict gains an
  additive "history" key; "log"/"history" are 1:1. KNOWN follow-up for 8.5:
  eval/run_eval.py:251 and orchestration_app.py report len(result["log"]) as
  "actions used", which now counts the bootstrap as 3 lines instead of 1 (a
  cosmetic reporting artifact — budget itself is enforced inside run_episode).
  bootstrap_global_pass default stays False (heuristic-baseline episodes
  unchanged); the v2 VLM path already passes True.
- 2026-07-10 (8.1, scope widened w/ user approval): deleting TileQueryA/
  SubdivideA/VerifyA broke module-level imports outside 8.1's named files, so
  the step also made minimal mechanical fixes there: agent/runner.py (drop
  TileQueryA from the import + _SENSING), agent/policy_vlm.py (menu/validation
  reduced to look/stop; tile/verify branches and their helpers _valid_conf/
  _valid_prompt/_valid_node_ids/_verifiable_ids removed — pulled forward from
  8.3), eval/run_eval.py (agent-episode --force-tile now warns + no-ops; fixed
  policies keep their tiled pass 1), orchestration_app.py (forced TileQueryA ->
  global QueryA over partition[0] — pulled forward from 8.5). Additionally
  migrated tests beyond the named four: test_look_roi.py (tiling False),
  test_runner_bootstrap.py (TileQueryA -> LookROIA), test_eval_sweep.py
  (force-tile agent test now asserts it is ignored), test_policy_vlm.py +
  test_policy_vlm_roi.py (tile/verify tests removed; "fell back" now asserted
  as equality with the heuristic's action, since the v2 heuristic itself emits
  LookROIA). Heuristic cell "cycling" is stateless: rank 2x2 cells by fewest
  candidates, rotate the pick by len(phi["D"]) % 4. 8.3/8.5 shrink accordingly.
- 2026-07-10 (Phase 8 planned, user decisions): v2 simplification per
  docs/active_perception_formulation.md. Locked with the user: full history
  (x_1^t, y_1^t) rebuilt into ONE stateless prompt per step (raw image +
  annotated overlay + JSON history); action menu collapses to look/stop with
  look = ONE UNTILED SAM3 query on a free ROI inside the tree ROI;
  TileQueryA/SubdivideA/VerifyA and agent/inspect.py are DELETED on the v2
  branch (user-approved amendment of the "old path stays default" rule for
  these); fixed 3-record bootstrap (canopy_roi, leaf_map, global_pass) always
  precedes the policy; V-IP queries get a per-query route field
  (cv/sam3/vlm) with a RouterOracle so Qwen only answers the residual.
  Plan only — no code changed this session.
- 2026-07-06 (7.1): roi_margin/roi_min_size/roi_max_depth/roi_dup_iou read via
  getattr with defaults (0.10 / 32px / 2 / 0.7); promoted to config.py in 7.4
  (matches policy_heuristic's "not yet in config" precedent). "Max 2 splits"
  encoded statelessly as an area floor = frame/4^depth (= frame/16). LookROIA is
  region-only; concept prompt + conf come from cfg (VLM chose only WHERE to look).
  A LookROIA reuses the QueryA executor with tiling=True, so it is exemplar-primed
  and counts as a sensing pass. FLAG (not from 7.1): running the full suite (now
  possible via a scratch venv) surfaced 2 pre-existing failures caused by the
  earlier user-requested green_citrus.json trim (22->6 queries) —
  test_queries.py::test_load_checked_in_green_citrus (asserts len>=20) and
  test_vip.py::test_posterior_recovers_true_class (asserts posterior>0.95, now
  0.941; verdict still correct). Both need a small out-of-scope test update.
- 2026-07-06 (query-trim test fixup, user-authorized): updated the two tests
  broken by the green_citrus.json 22->6 trim — test_queries.py len assertion
  20->5, test_vip.py clean-posterior threshold 0.95->0.90 (verdicts unchanged).
  Full suite green (215). Marked 7.1 [x] on the user's explicit instruction
  (automated tests only; the GPU-REPL "You verify" was not run here).
- 2026-07-06 (7.2): run_episode gained bootstrap_global_pass + auto_stop, BOTH
  default-off (old episodes unchanged per the "new behavior behind a flag" rule).
  Bootstrap = one non-tiled QueryA over partition[0] (canopy tree_roi / full frame),
  counted against budget. auto_stop ends on DiscoveryCurve.saturated over a non-empty
  graph; budget is the loop cap; "all ROIs sensed" is subsumed by saturation.
  LookROIA added to the sensing set that feeds the discovery curve. FLAG for 7.3:
  enabling these flags for the actual VLM episode is a one-line change in
  eval/run_eval.py::_run_agent_policy (out of 7.2/7.3's named files) — 7.3 will need
  its scope widened to include run_eval.py, or a follow-up. Marked [x] on user
  instruction (automated tests only; GPU "You verify" not run here).
- 2026-07-06 (7.3, scope widened w/ user approval): the guided-ROI VLM policy needs
  the image + the 7.2 flags wired through non-policy files, so 7.3 touched, beyond
  its named files: agent/runner.py (make_vlm_policy passes image=ctx.image_pil),
  eval/run_eval.py (_run_agent_policy enables bootstrap_global_pass + auto_stop only
  for policy=="vlm"; heuristic unchanged), and tests/test_policy_vlm.py (removed the
  now-obsolete query/subdivide legal-mapping tests; fixed _build_messages call to the
  new no-partition signature). policy_vlm reworked: choose() gains an image param;
  menu is look/tile/verify/stop (query + subdivide removed); "look" -> LookROIA is
  validated in-bounds against image_size and is a sensing target only (grounding
  invariant preserved). Suite green (228).
- 2026-07-06 (7.4, scope widened w/ user approval): added roi_margin/roi_min_size/
  roi_max_depth/roi_dup_iou to config.py (were getattr defaults in 7.1; values match)
  + policy_enable_thinking (default False) + module helper thinking_call_kwargs().
  Widened beyond config.py/CLAUDE.md to agent/policy_vlm.py so the toggle actually
  applies: _request now passes the disable-thinking extra_body for the policy loop
  (inspect + verify oracle untouched -> keep thinking on). CLAUDE.md #3 reworded to
  the ROI-as-sensing-target wording. CAVEAT: thinking_call_kwargs uses the vLLM/Qwen
  convention (extra_body.chat_template_kwargs.enable_thinking); the live Ollama
  qwen3-vl endpoint may want top-level {"think": false} instead -- change only that
  one helper if so. Suite green (232). Phase 7 code complete (all 4 steps [~]/[x]).
- 2026-07-05 (functional-review fixes, user request): (1) VerifyA now raises a
  clear ValueError without oracle/query_set, and BOTH policies only propose/accept
  "verify" when cfg.verifier_mode=="vip" (run_eval only wires an oracle for vip;
  previously --policy vlm/heuristic --verifier ioc/off could crash mid-sweep).
  (2) Leaf-map cache is keyed by its generating ROI (graph.cached_leaf_roi) and
  regenerated when the ROI changes — cross-frame leaf boxes corrupted IoC verdicts
  and neg exemplars in agent episodes with region queries; canopy-ROI runs are
  byte-for-byte unchanged. (3) Cross-pass dedup under vip/off also matches
  "unresolved" tracks (IoC still fruit-only, golden path untouched): "off" no
  longer re-registers everything each pass, and convergence can converge. (4) Cost
  is now metered from actual per-pass calls via new additive PassStats fields
  n_sam_calls/n_verify_calls (canopy + leaf map + proposal; real vip oracle calls
  incl. inline episode verifications, excl. dedup-rejected/tiny-skipped) — fixed
  policies and episodes share one scale; agent episodes start their partition at
  [tree_roi] (plan 4.1) when a processor is present, metering the canopy call.
  (5) run_eval --prompt/--conf now reach heuristic/vlm via cfg.target_prompt/conf.
  (6) getattr-default tunables promoted to config.py (target_prompt, c0,
  small_area, tau_w, k_min, tau_high, area_min/max — inert defaults). vip_epsilon
  default changed 0.15 -> None: None defers to the query set's epsilon, a float
  overrides it (the knob was previously dead). (7) queries.py preserves optional
  sam3_phrase (validated non-empty str) so Sam3Oracle is reachable. (8) QwenOracle
  and inspect_scene retry request/network exceptions like parse failures (all-
  zeros / neutral-z fallback instead of aborting the pass). (9) Episode pass_number
  now counts sensing passes only (ActionContext.n_passes): Verify/Subdivide/Stop
  no longer inflate pass numbers (the discovery curve still records every action,
  per the proposal). (10) vip leaf_verification uses the actual "distractor" index
  (0.0 if that class is absent — index 0 was the target). Tests updated to the new
  metering/pass semantics + new tests/test_review_fixes.py. Suite 178 -> 200.
- 2026-07-05 (mask-overlap switch, user request): new cfg.overlap_mode
  ("box" default | "mask"): NMS Gate A IoU / Gate B IoM measured on SAM3 instance
  masks via an additive masks= param on apply_nms_dualgate (None -> byte-for-byte
  box behavior; boxes still bound candidate pairs, concentric sub-gate stays box-
  based), and cross-pass dedup via pipeline.compute_mask_iou on box-cropped masks
  stored as OrchardNode.mask (frame-independent; not serialized in to_dict).
  Masks flow run_raw_inference(return_masks=True) -> dual-gate NMS
  (return_indices) -> box-cropped -> register_and_verify_candidates
  (candidate_masks=). Scope: global (non-tiled) dualgate passes only — tiled or
  nms_mode="iou" passes log a warning and fall back to box overlap (tile masks
  would need global-frame stitching); the IoC leaf-containment gate stays
  box-based (leaf map has no masks). tests/test_mask_overlap.py covers the gates,
  alignment through the confidence filter, dedup, and an execute_pass round trip.
- 2026-07-05 (testing docs + mask CLI, user request): rewrote docs/tier_b_testing.md
  as an Ollama-first numbered walkthrough (T0-T12) — each test states Run/Expect/
  Artifacts/If-it-fails, every command tees to out/T<N>.log, and a failure-report
  template + env block sits at the top (vLLM demoted to an appendix; fixed the
  Ollama tag to qwen3-vl:8b / qwen2.5vl, no hyphen). docs/manual_tests.md updated
  to 204 tests and now references the T-ids; added sections for the review fixes
  and the mask switch. To make mask mode runnable from the CLI, added additive
  --overlap-mode {box,mask} flags to scripts/run_image.py (outputs tagged
  <stem>_<verifier>_mask_* so box/mask runs don't overwrite) and eval/run_eval.py
  (CSV verifier column becomes e.g. "ioc+mask" via verifier_label() so mask rows
  don't collide with box rows on resume). Suite 200 -> 204.
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
- 2026-07-04 (1.5): Branch on `cfg.verifier_mode` (actual field name; plan wrote
  cfg.verifier). VIP path is a pure early-continue inserted after add_candidate;
  the IoC gate below is byte-for-byte unchanged (verified via git diff: no `-`
  lines in the gate). 12px skip applies ONLY to the VIP branch (applying it to
  IoC would break the byte-for-byte guarantee). verifier_mode="vip" without
  oracle/query_set/image_np raises ValueError rather than silently falling back
  (IoC stays the safe default; misconfig surfaces loudly). Threaded
  cfg/oracle/query_set through execute_pass as optional None-default params +
  image_np=img_np. NOT DONE (out of 1.5's file scope): run_image.py has no
  --verifier flag, so the plan's "run_image.py per verifier mode" manual check
  can't be run as written yet — needs a small step-0.2-file follow-up.
- 2026-07-04 (6.3): eval/report.py reads sweep CSVs, prints a markdown table
  (policy x verifier: MAE/RMSE/Exact/F1/pool_recall/mean cost) and saves
  out/accuracy_vs_cost.png via a headless Agg Figure (no seaborn); verifier =
  marker style, policy annotated. Added tests/test_report.py (plan named no test
  file; ground rule requires one).
- 2026-07-04 (follow-ups A+B, user request): (A) agent/runner.make_vlm_policy wraps
  should_inspect->inspect_scene->policy_vlm.choose into a runner-compatible
  callable(phi,partition,cfg) with injectable clients — the full VLM episode loop.
  (B) eval/run_eval now uses belief.count_estimates (real N_supp/N_cons, no longer
  == N_obs) and CostMeter.total for cost; --policy heuristic/vlm now run real
  episodes via runner.run_episode (previously raised NotImplementedError). run_policy
  returns (graph, CostMeter, n_actions); split into _run_fixed_policy/_run_agent_policy;
  evaluate_image gained episode_execute_fn injection. Updated test_eval_sweep
  accordingly (cost.n_* asserts, heuristic-episode test) + make_vlm_policy test.
  This lifts the 6.2 "REDUCED" caveat.
- 2026-07-04 (5.2): policy_vlm.choose(phi,z,partition,graph,cfg) one text call, no
  image; NO retry — any parse failure or validation violation falls straight back to
  policy_heuristic.choose (the safety net). VLM references regions/nodes by id only
  (no action accepts coordinates -> no box injection). Validation: region_id indexes
  partition, conf in [0.1,0.9] (bool rejected), optional prompt must == target concept
  (getattr target_prompt default "green fruit"), verify node_ids must be in-graph AND
  unresolved-or-low-w (support_score < tau_w). Injectable client for tests; openai
  lazy; logs prompt+response+"fallback" for the manual grep. NOT in 5.2's files
  (deferred): the VLM episode loop that runs should_inspect->inspect_scene->passes z,
  and a --log-prompts CLI — needs a runner/orchestrator touch beyond policy_vlm.py.
- 2026-07-04 (5.1): inspect_scene mirrors QwenOracle (base64 image, strict JSON
  parse + retry + neutral fallback with recommend="query"); injectable client for
  offline tests; openai imported lazily. Strict _parse_z rejects missing keys / bad
  enums / non-bool target_present, truncates notes to 200. should_inspect lacks the
  policy's action scores, so condition 2 ("tiling/subdivide decision pending") is
  proxied by last_z.recommend in {tile, subdivide}, and "discovery saturated" by
  no new candidates in the last _SATURATION_WINDOW=3 passes (sum(D[-3:])==0) — no
  cfg needed. Both documented as proxies in the docstring.
- 2026-07-04 (4.1): Prompt required estimators "in belief.py" (not in 4.1's Files
  line) -> added count_estimates there (N_obs/N_supp/N_cons over non-leaf/non-spurious
  nodes, consistent with 6.2's predicted_nodes; thresholds tau_w=0.5/k_min=2/
  tau_high=0.5 via getattr, to be promoted to config later). Extended belief.summarize
  to add U/ids/centers/area/classification so choose(phi,partition,cfg) is
  self-contained (the region VoI proxies + VerifyA node_ids need per-node geometry;
  existing summarize tests check key-presence, unaffected). policy_heuristic proxies
  all documented in the choose docstring; c0/target_prompt/small_area via getattr
  defaults. TileQuery cost proxy = c_tile*4 (real tile count only known post-hoc).
  runner takes an injectable execute_fn; policy is a callable(phi,partition,cfg).
  Stop-by-step-5 test relaxes delta_U so the stop hinges on discovery saturation.
- 2026-07-04 (3.2): Files named budget.py/actions.py/test, but the Prompt required
  the tile count "from tiled_engine" -> additive pipeline.py edits (all backward
  compatible, hard-constraint #1 permits additive tiled_engine params): tiled_engine
  gains return_tile_count=False (default still returns the 2-tuple; n_tiles =
  len(x_offsets)*len(y_offsets)); execute_pass captures it; PassStats gains n_tiles=0.
  CostMeter lives on ActionContext (default_factory) so execute() meters in place;
  n_orch += 1 per executed action (incl. Subdivide/Stop). Leaf-map +1 (n_sam) on
  pass 1 applies to BOTH Query and TileQuery (it's the same physical global call in
  execute_pass); the Prompt only named it under QueryA. VerifyA uses
  result["n_oracle_calls"] (1 batched / chain length sequential) so both modes are
  handled uniformly. Canopy Pass-0 SAM3 call is NOT separately metered (matches the
  Prompt's simplified model; QueryA skips canopy anyway via roi_override).
- 2026-07-04 (3.1): Files line named only actions.py + test, but the Prompt required
  adding roi_override to execute_pass (pipeline.py) — did that additive change
  (default None = canopy-gated behavior byte-for-byte; verified via diff + IoC golden
  still green). partition lives on ActionContext (plan: "on the episode state");
  SubdivideA mutates it in place and execute still returns int 0. pass_number for
  Query/TileQuery derived as len(discovery.counts)+1 (no explicit episode pass
  counter). TileQueryA = global tiled (roi_override=None). actions.py imports pipeline
  LAZILY (torch) but verify_candidate at top (torch-free), so the module + Subdivide/
  Verify are CPU-testable; Query/TileQuery tested via a sys.modules pipeline stub.
  SubdivideA raises ValueError if the region isn't in the partition (surfaces policy bugs).
- 2026-07-04 (2.2): belief.py pure functions, formulas verbatim from the proposal.
  s_bar = node.scores["detection_confidence"] (no running-aggregate score exists yet).
  support_score's plausible-area term reads cfg.area_min/area_max via getattr with
  inert defaults [0, inf] (config has no area bounds and 2.2 can't touch config.py),
  so the lambda_A term is a no-op until bounds are added. uncertainty sums over
  non-spurious nodes (per plan); summarize's w/s/k/delta lists cover ALL K_t nodes
  (per proposal phi). tiling_status derived from signatures (any ':tiled:' present),
  since summarize isn't passed tiling state. saturated(m, delta) requires a full
  window (len >= m) so it can't fire before m passes. NOTE for later: 6.2's
  N_supp/N_cons can now be upgraded to use support_score/support k (still deferred
  until a step revisits run_eval).
- 2026-07-04 (2.1): OrchardNode gains support(k)/signatures(set)/jitter/area +
  reinforce(box, signature). jitter = running mean of Euclidean center displacement
  of re-detections vs the representative box center. Also record the CREATING
  signature on the accepted node (pipeline accept branch) so support == |signatures|
  (proposal k=|Q|); the plan only named reinforce, noting the addition. to_dict now
  emits support/jitter/area/signatures (signatures as sorted list) — needed for the
  "top-10 by k from graph JSON" check; existing to_dict tests check key presence not
  exclusivity, so unaffected. Dedup branch: captures the matched fruit node and calls
  reinforce; acceptance/rejection behavior byte-for-byte unchanged (signature defaults
  None -> no-op on the IoC golden path). Signature built in execute_pass:
  f"{pass_number}:{mode}:{prompt}:{conf:.2f}", mode = tiled/global.
- 2026-07-04 (6.2 REDUCED): Implemented per the plan's suggested-order milestone
  "6.2 with {oneshot, cascade} only" because Phases 2-5 don't exist yet. IN:
  metrics.py (mae/rmse/exact + normalized_cost from a call-counts dict + cfg
  ratios) and run_eval.py with policies oneshot/cascade/tiled/convergence, CSV +
  aggregate + resume-safe + per-image numpy seed. DEFERRED (all forward-compatible):
  (a) N_supp/N_cons are placeholders == N_obs until support tracking (2.1) +
  belief.py estimators (2.2); estimate_counts() is the single swap point.
  (b) --policy heuristic/vlm kept in the CLI but raise NotImplementedError until
  the runner (4.1/5.2). (c) cost is a pass-count approximation (n_global/n_tile/
  n_verify) pending CostMeter (3.2) — normalized_cost already matches the proposal
  formula, so CostMeter just feeds it real counts later. N_obs = predicted nodes =
  classification in {fruit, unresolved} (so verifier="off" still yields a count).
  execute_pass is injected (execute_pass_fn) so run_policy is testable without torch.
- 2026-07-04 (0.3 refactor, user request): datasets now use the layout
  root/images/<split>/ + root/labels/<split>/ (+ masks/<split>/), split in
  train/val/test. load_split gained a `split` param (default "train"); falls back
  to flat root/images if the split subdir is absent, else raises FileNotFoundError.
  main() gained --split. run_image find_label_file now maps images/<split>/img.png
  -> labels/<split>/img.txt by swapping the 'images' path component for 'labels'.
- 2026-07-04 (6.1): Did 6.1 before the remaining Phase 2-5 steps per the plan's
  suggested week-1 order (quantify vip-vs-ioc early). `scipy` (already listed in
  requirements.txt) was not installed in this dev env; installed scipy 1.18.0 to
  run tests — no new dependency, no code/requirements change. match() maximizes
  total IoU via linear_sum_assignment then drops sub-threshold pairs (one-to-one).
  pool_recall is COVERAGE (each GT matched by any candidate), not one-to-one, per
  the proposal. Empty-GT -> recall/pool_recall 0.0 (guarded). per_pass_pool_recall
  is cumulative (pool at pass p = candidates with found_in_pass <= p), over ALL
  node boxes regardless of classification (discovery, not verification).
- 2026-07-04 (0.3): eval/datasets.py imports inference.yolo_to_xyxy LAZILY so the
  module (and its MinneApple/overlay paths) stay torch-free; the YOLO test stubs
  torch+transformers to load the real yolo_to_xyxy. YOLO loader supports both
  images/+labels/ subdirs and same-dir layouts. MinneApple boxes use half-open
  max [xmin,ymin,xmax+1,ymax+1] per instance id; handles grayscale (value=id) and
  RGB (color=id) masks. draw_gt_overlay uses a headless Agg Figure (no global
  matplotlib.use) so tests need no display. run_image --draw-gt auto-locates the
  image's YOLO <stem>.txt (labels/ sibling, then same dir) and saves
  out/<stem>_gt.jpg via the datasets helper.
- 2026-07-04 (post-1.5, user request): Added verifier "off" mode (config.py
  comment + pipeline.py): registers candidates but does no classification, so
  nodes stay "unresolved" (no-verifier baseline). Added `--verifier {ioc,vip,off}`
  + `--oracle {mock,qwen}` + `--query-file` + `--mock-class` to run_image.py
  (user authorized touching this 0.2 file), threading cfg/oracle/query_set into
  execute_pass. run_image now tags output files with the verifier mode
  (`<stem>_<mode>_overlay.jpg`) so ioc/vip/off runs don't overwrite each other —
  this makes the plan's 1.5 "diff the two overlays" check runnable; default
  output filename thus changed from `<stem>_overlay.jpg` to `<stem>_ioc_...`.
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

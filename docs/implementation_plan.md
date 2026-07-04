# Implementation Plan — Zero-Shot Active Perception Counting (SAM3 + Qwen-3-VL + FM+V-IP Verifier)

Full-system plan covering the FM+V-IP verifier, the belief/graph upgrades, the
action layer, the heuristic and VLM orchestrator policies, and the evaluation
harness — built incrementally on top of the existing `app.py`, `pipeline.py`,
`inference.py`, `graph.py`.

## Ground rules (paste these into every Claude Code session)

```
RULES FOR THIS REPO:
- Simplicity over cleverness. Plain functions, plain dicts, numpy + PIL. No new
  frameworks. No abstract base classes. No async.
- Never modify validated code paths (apply_nms, apply_nms_dualgate, tiled_engine,
  run_raw_inference). New behavior always goes behind a config flag with the old
  behavior as default.
- Every step ships: the code, one small pytest file, and nothing else.
- All model calls (SAM3, Qwen) must be mockable: pass the model/oracle object in
  as an argument, never import-and-call globally.
- All new tunables live in config.py, never as magic numbers in function bodies.
- Do not touch files outside the ones named in the step.
```

## Design decisions (already made — do not relitigate in-session)

1. **No CLIP.** Query oracle = Qwen-3-VL (already the orchestrator model), via an
   OpenAI-compatible endpoint so any serving stack works (vLLM, llama.cpp,
   DashScope). SAM3 presence scores are an optional secondary channel, off by
   default.
2. **Batched answering + post-hoc IP.** One structured Qwen call per crop answers
   all M queries; the greedy IP chain is computed afterwards over the stored
   answer vector. Identical chain/posterior to sequential IP, cost = 1 VLM call
   per candidate. Sequential mode is a flag for ablation only.
3. **Offline query generation.** Query sets and class answer templates are
   generated once by an LLM (script provided), reviewed by you, and checked into
   the repo as JSON. Runtime never calls an LLM to make queries — deterministic
   and still zero-shot (no labeled data ever used).
4. **Training-free V-IP.** 3 classes (`target`/`distractor`/`spurious`), M ≈ 20–30
   queries, LLM templates + noise rate ε → closed-form naive-Bayes posterior and
   greedy conditional-mutual-information selection. No querier network.
5. **Old verifier stays.** The IoC/occlusion gate remains the default
   (`verifier="ioc"`); the new one is `verifier="vip"`. Every experiment can A/B.

## Target file tree

```
orchard/
├── app.py  pipeline.py  inference.py  graph.py     # existing (minimal edits)
├── config.py                                       # all tunables, one dataclass
├── verifier/
│   ├── queries.py        # load query sets + templates from JSON
│   ├── oracle.py         # QwenOracle, Sam3Oracle (optional), MockOracle
│   ├── vip.py            # posterior, info gain, greedy chain, stop rule (pure numpy)
│   └── verify.py         # crop -> (verdict, posterior, chain); called by pipeline
├── agent/
│   ├── belief.py         # w_i, U(b~), discovery curve, region stats
│   ├── actions.py        # Query/TileQuery/Subdivide/Verify/Stop + executor
│   ├── budget.py         # cost meter
│   ├── inspect.py        # z_t: Qwen scene assessment (structured JSON)
│   ├── policy_heuristic.py
│   ├── policy_vlm.py
│   └── runner.py         # episode loop: while not stop: act
├── eval/
│   ├── datasets.py       # MinneApple / citrus (YOLO txt) loaders
│   ├── matching.py       # Hungarian matching, P/R/F1, pool recall
│   ├── metrics.py        # MAE/RMSE/Exact, normalized cost
│   └── run_eval.py       # CLI: policy × estimator sweep -> CSV
├── scripts/
│   ├── gen_queries.py    # one-time LLM query/template generation
│   ├── run_image.py      # run N passes on one image, dump overlay + graph JSON
│   └── demo_verify.py    # verify 3 crops, print chains
├── queries/
│   └── green_citrus.json # generated + hand-reviewed
└── tests/
```

## JSON schema for a query set (fix this now, everything depends on it)

```json
{
  "concept": "green citrus",
  "classes": ["target", "distractor", "spurious"],
  "class_names": {"target": "fruit", "distractor": "leaf", "spurious": "clutter"},
  "epsilon": 0.15,
  "queries": [
    {
      "id": "q01",
      "text": "Is the central object in this crop approximately round or spherical?",
      "templates": {"target": 1, "distractor": -1, "spurious": 0}
    }
  ]
}
```

---

# Phase 0 — Scaffold & harness (no model changes)

### Step 0.1 — Package layout + config
**Files:** `config.py`, empty `verifier/ agent/ eval/ scripts/ tests/` packages,
`tests/test_smoke.py`.
**Prompt:**
> Create the package layout above with empty `__init__.py` files. Write
> `config.py` containing a single `@dataclass Config` with fields grouped by
> comments: paths (sam3_repo, queries_dir, output_dir), detection (conf=0.35,
> nms_mode="dualgate"), verifier (mode="ioc", vip_query_file, vip_epsilon=0.15,
> vip_stop=0.10, vip_max_queries=10, crop_scale=1.4, crop_size=256,
> answer_mode="batched"), oracle (base_url, model_name, temperature=0.0,
> max_retries=2), agent (window_m=3, delta_disc=1.0, delta_U, lambdas as a dict,
> budget_max_actions=12), costs (c_sam=1.0, c_tile=0.25, c_verify=0.5,
> c_inspect=0.5, c_orch=0.05). Add `Config.load(path)` reading overrides from a
> JSON file. One smoke pytest that instantiates Config.
**You verify:** `pytest` green; `python -c "from config import Config; print(Config())"`.

### Step 0.2 — CLI runner for the existing cascade
**Files:** `scripts/run_image.py`; tiny addition to `graph.py` (`OrchardGraph.to_dict()`).
**Prompt:**
> Add `OrchardGraph.to_dict()` returning `{"nodes": [node.to_dict() ...]}`. Write
> `scripts/run_image.py` with argparse: `--image --prompt --passes N --tiling
> --clahe --conf`. It loads SAM3 once via inference.load_sam3_model, runs
> pipeline.execute_pass N times reusing one graph, prints each PassStats.as_row(),
> and saves `out/<stem>_overlay.jpg` (inference.plot_graph_scene) and
> `out/<stem>_graph.json`. No changes to execute_pass.
**You verify:** run on one orchard image with `--passes 3`; overlay looks like the
Gradio output; JSON opens and node count matches the log.

### Step 0.3 — Dataset loader
**Files:** `eval/datasets.py`, `tests/test_datasets.py`, `scripts/` gets `--draw-gt` in run_image.
**Prompt:**
> Write `eval/datasets.py` with `load_split(root, fmt)` where fmt ∈
> {"minneapple", "yolo"}. Return a list of dicts
> `{"image_path", "gt_boxes": Nx4 float xyxy, "count": int}`. For "yolo", parse
> per-image txt files using inference.yolo_to_xyxy. For "minneapple", parse its
> instance masks into boxes (bounding box of each instance id). Include a
> `main()` that prints dataset size and per-image counts for the first 5 images,
> and saves one GT overlay jpg. Pytest with 2 tiny synthetic YOLO txt fixtures.
**You verify:** run `python -m eval.datasets --root ... --fmt yolo`; GT overlay
boxes sit on real fruit.

---

# Phase 1 — FM+V-IP verifier

### Step 1.1 — Query sets + generation script
**Files:** `verifier/queries.py`, `scripts/gen_queries.py`, `queries/green_citrus.json`, `tests/test_queries.py`.
**Prompt:**
> Implement `queries.py`: `load_query_set(path) -> QuerySet` (plain dataclass:
> concept, classes, epsilon, list of Query(id, text, templates dict)). Validate
> the schema strictly (templates only in {-1,0,1}, all classes present, unique
> ids) and raise ValueError with a helpful message. Write
> `scripts/gen_queries.py` that calls an OpenAI-compatible chat endpoint with a
> prompt modeled on FM+V-IP ("List the useful visual attributes and their values
> of ...") but asking directly for the JSON schema above, for classes
> fruit/leaf/clutter given `--concept`. Also hand-write
> `queries/green_citrus.json` with ~22 sensible queries (roundness, waxy sheen,
> specular highlight, uniform color, visible veins, flat/thin shape, serrated
> edge, stem attachment, branch/bark texture, sky/soil background, partial
> occlusion by leaves, blur, etc.) so the repo works fully offline. Pytest:
> loading the checked-in file succeeds; a corrupted copy fails validation.
**You verify:** read `green_citrus.json` yourself — every query should be
answerable by looking at a 256px crop, and templates should match your intuition.
This file is the single highest-leverage artifact in the verifier; edit freely.

### Step 1.2 — Oracles
**Files:** `verifier/oracle.py`, `tests/test_oracle.py`.
**Prompt:**
> Implement three classes with one shared method
> `answer_batch(crop_pil, query_set) -> np.ndarray of int8 in {-1,0,1} (len M)`:
> (1) `MockOracle(true_class, noise=0.0, seed=0)` answers from the templates with
> optional flips — for tests. (2) `QwenOracle(cfg)` sends ONE chat completion:
> system prompt fixes the task; user content = the crop image (base64) + a
> numbered list of the M yes/no questions + instruction to reply ONLY with JSON
> `{"answers": {"q01": "yes"|"no"|"unsure", ...}}`. temperature=0, strict
> json parsing, on parse failure retry up to cfg.max_retries then return all
> zeros and log a warning. Map yes/no/unsure -> 1/-1/0; missing ids -> 0.
> (3) `Sam3Oracle(processor, tau)` (optional channel): only answers queries whose
> JSON entry has an optional "sam3_phrase" field, by thresholding the SAM3
> presence score on the crop (reuse the pattern from
> inference.verify_box_semantics); returns 0 for all other queries. Pytest covers
> MockOracle determinism and QwenOracle JSON parsing against a canned fake
> response object (no network).
**You verify:** `pytest`; then a 5-line REPL smoke test: QwenOracle on one real
fruit crop — answers should be visibly sane (round=yes, veins=no).

### Step 1.3 — Training-free V-IP core
**Files:** `verifier/vip.py`, `tests/test_vip.py`.
**Prompt:**
> Pure numpy, no model calls. Implement: `likelihood_table(query_set) ->
> (K, M, 3)` array of P(answer ∈ {-1,0,1} | class) built from templates and
> epsilon per the noise model: agree -> 1-eps, contradict -> eps, either side
> uninformative (template 0 or answer 0) -> 1/2 (then normalize rows over the 3
> answer values so it is a proper distribution). `posterior(answers, S, table,
> prior)` -> length-K probs using only indices in S, computed in log space.
> `info_gain(m, S, answers, table, prior)` -> closed-form conditional mutual
> information I(a_m; v | a_S) = H(a_m|a_S) - sum_k P(k|a_S) H(a_m|k).
> `run_ip(answers, table, prior, stop_eps, max_q)` -> dict {verdict_idx,
> posterior, chain: [(query_idx, answer), ...]} implementing greedy selection
> with the stopping rule max_k P >= 1 - stop_eps. Pytests: (a) MockOracle
> template answers for class k -> posterior(k) > 0.95 within max_q; (b) a query
> with template 0 for all classes is never selected; (c) chain length strictly
> shorter than M for clean answers; (d) permutation-invariance: posterior over
> full S equals product regardless of order.
**You verify:** `pytest -v` — read the four test names; they are the spec.

### Step 1.4 — Verify API + demo
**Files:** `verifier/verify.py`, `scripts/demo_verify.py`, `tests/test_verify.py`.
**Prompt:**
> `verify.py`: `extract_crop(image_np, box, scale, size)` (center-scale, clamp,
> cv2 INTER_CUBIC — mirror the geometry in inference.verify_box_semantics), and
> `verify_candidate(image_np, box, oracle, query_set, cfg)` -> dict {verdict:
> "target"/"distractor"/"spurious", p_target: float, posterior: list, chain:
> [{"q": text, "a": "yes/no/unsure"}], n_oracle_calls: int}. In batched mode:
> one oracle call then vip.run_ip; in sequential mode: call oracle per selected
> query (oracle gets a single-query QuerySet view). `demo_verify.py`: argparse
> --image --boxes x1,y1,x2,y2 (repeatable) --oracle mock|qwen; pretty-print each
> chain like:  [q03 round? yes][q07 waxy? yes][q11 veins? no] -> target (0.94).
> Pytest with MockOracle end-to-end.
**You verify:** run demo on 3 hand-picked boxes (clear fruit, clear leaf, junk)
with the real Qwen oracle. The three chains should read like sensible reasoning.
This is the step where you actually judge whether the whole idea works — spend
time here, tweak queries/epsilon in the JSON, re-run.

### Step 1.5 — Pipeline integration behind a flag
**Files:** `pipeline.py` (surgical), `graph.py` (store chain), `tests/test_pipeline_vip.py`.
**Prompt:**
> In graph.py add optional fields to OrchardNode: `vip_chain=None`,
> `vip_posterior=None`; include them in to_dict when set. In pipeline.py,
> thread `cfg` and an optional `oracle`/`query_set` down into
> register_and_verify_candidates. If cfg.verifier == "vip": after registration,
> replace the entire occlusion-aware logic-gate block with: result =
> verify.verify_candidate(...); map target->"fruit", distractor->"leaf",
> spurious->"spurious"; write scores fruit_verification=p_target,
> leaf_verification=posterior[distractor]; store chain. If cfg.verifier ==
> "ioc" (default): behavior byte-for-byte unchanged (keep the IoC computation
> inside the ioc branch only). Also skip verification (keep "unresolved") when
> node box < 12px a side. Pytest: run register_and_verify_candidates with
> MockOracle("target") and check all nodes classified fruit; with
> verifier="ioc" and no oracle, results equal the pre-change golden values
> (hardcode a tiny fixture).
**You verify:** `scripts/run_image.py --passes 2` twice, once per verifier mode,
same image. Diff the two overlays and the two graph JSONs; spot-check 5 nodes
where verdicts differ by opening those crops with demo_verify.

---

# Phase 2 — Belief state (graph upgrades from the proposal)

### Step 2.1 — Support, jitter, signatures
**Files:** `graph.py`, `pipeline.py` (dedup branch), `tests/test_graph_support.py`.
**Prompt:**
> Extend OrchardNode with: `support k` (int, starts 1), `signatures` (set of
> strings), `jitter delta` (float, running mean of center displacement of
> re-detections), `area` (float, from box). In pipeline
> register_and_verify_candidates, when a detection is rejected as a cross-pass
> duplicate of node n: instead of only counting it, call new method
> `n.reinforce(box, signature)` which increments k, adds the signature, and
> updates delta with the center distance. Signature string =
> f"{pass_number}:{mode}:{prompt}:{conf:.2f}" passed down from execute_pass
> (mode = "tiled"/"global"). No behavior change for acceptance/rejection.
> Pytest: feed the same box twice with different signatures -> k==2, delta small;
> shifted box -> delta grows.
**You verify:** 3-pass run; print top-10 nodes by k from the graph JSON — stable
fruits should have k≥2, one-off junk k==1.

### Step 2.2 — belief.py: w_i, U, discovery curve, φ summary
**Files:** `agent/belief.py`, `tests/test_belief.py`.
**Prompt:**
> Pure functions over the graph, formulas verbatim from the proposal:
> `support_score(node, cfg)` = s̄ + λk·log(1+k) − λΔ·Δ − λA·1[area outside
> plausible range]; `uncertainty(graph, discovery, cfg)` = λD·mean(last m of
> discovery) + λS·Σ(1/(1+k) + αΔ·Δ + αs·(1−s̄)) over non-spurious nodes;
> `DiscoveryCurve` tiny class: append(n_new), mean_recent(m), saturated(m,
> delta). `summarize(graph, discovery, budget, cfg) -> dict φ` with K, n_t,
> D, lists of w/s/k/delta (rounded), tiling_status, remaining_budget — this
> dict is what policies consume and must be json-serializable. Pytest with a
> hand-built 3-node graph: adding support lowers U; flat discovery lowers U.
**You verify:** pytest; then print φ after each pass in run_image — U should
visibly drop across passes on an easy image.

---

# Phase 3 — Actions & cost

### Step 3.1 — Action layer
**Files:** `agent/actions.py`, `tests/test_actions.py`.
**Prompt:**
> Plain dataclasses: QueryA(region, prompt, conf), TileQueryA(prompt, conf),
> SubdivideA(region), VerifyA(node_ids), StopA(estimate_name). Region = xyxy
> tuple; maintain a partition list on the episode state starting as [tree_roi],
> SubdivideA splits one region 2×2 and returns the new partition (no model
> call). `execute(action, ctx) -> n_new_candidates` where ctx bundles
> processor, image, graph, cfg, oracle, query_set, discovery. QueryA/TileQueryA
> call pipeline.execute_pass restricted to the region (add an optional
> `roi_override` param to execute_pass that skips canopy detection and uses the
> given region — default None keeps behavior identical). VerifyA runs
> verify_candidate on the named unresolved/low-w nodes and updates them.
> Pytest with mocks: dispatch table correct; Subdivide never calls models;
> execute returns ints.
**You verify:** REPL: execute one QueryA on a quadrant of a real image; overlay
shows detections only inside that quadrant.

### Step 3.2 — Cost meter
**Files:** `agent/budget.py`, wiring in actions.py, `tests/test_budget.py`.
**Prompt:**
> `CostMeter` with counters n_sam, n_tile, n_verify, n_inspect, n_orch and
> `total(cfg)` implementing the normalized cost formula from the proposal
> (weights = cfg.costs ratios). Increment inside execute(): QueryA -> n_sam+=1
> (+1 more on pass 1 for the leaf map), TileQueryA -> n_tile += number of tiles
> actually run (return it from tiled_engine via a small counter, do not change
> its outputs), VerifyA -> n_verify += len(node_ids) in batched mode / chain
> lengths in sequential, every policy decision -> n_orch += 1. Pytest: scripted
> sequence of mocked actions produces exactly the hand-computed total.
**You verify:** 2-pass run prints a cost line; recount by hand from the logs once.

---

# Phase 4 — Heuristic controller (the non-VLM baseline)

### Step 4.1 — VoI heuristic policy + episode runner
**Files:** `agent/policy_heuristic.py`, `agent/runner.py`, `tests/test_policy_heuristic.py`.
**Prompt:**
> `policy_heuristic.choose(phi, partition, cfg) -> action`: score each concrete
> candidate action with the VoI-per-cost ratio using cheap predicted-ΔU proxies
> (document each in a docstring): QueryA(r): λD·recent local discovery +
> λU·region-restricted instability + λC·crowding (formulas V_query(r) from the
> proposal); TileQueryA: global version, boosted when median candidate area is
> small; VerifyA: sum over unresolved/low-w nodes of their instability term;
> SubdivideA(r): fraction of r's candidates with k==1. Divide each by
> (c0 + cost_sense(a)). Stop when discovery.saturated(m, δ_disc) AND U ≤ δ_U,
> returning StopA. `runner.run_episode(image, ctx, policy, max_actions)` loops
> choose -> execute -> update discovery/φ, logs one JSON line per step
> {t, action, n_new, U, cost_so_far}, returns final graph + counts
> {N_obs, N_supp, N_cons} (implement the three estimators in belief.py from the
> proposal's formulas). Pytest: with a stub executor whose n_new goes 5,2,0,0,
> the policy stops by step 5.
**You verify:** run episodes on 3 images (sparse, dense, empty-of-fruit). Read
the action traces: dense image should trigger TileQuery/Subdivide; empty image
should stop within ~3 actions.

---

# Phase 5 — VLM orchestrator

### Step 5.1 — Scene inspection z_t
**Files:** `agent/inspect.py`, `tests/test_inspect.py`.
**Prompt:**
> `inspect_scene(image_or_crop, phi, oracle_cfg) -> dict z` via one Qwen call
> returning ONLY JSON: {target_present: bool, density: "sparse|medium|dense",
> object_scale: "large|medium|small", occlusion: "low|medium|high",
> recommend: "query|tile|subdivide|verify|stop", notes: str<=200}. Same strict
> parse/retry/fallback pattern as QwenOracle (fallback = neutral z with
> recommend="query"). `should_inspect(t, phi, last_z) -> bool` implementing the
> fixed protocol from the proposal: t==1, or a tiling/subdivide decision is
> pending (top two action scores within 10%), or discovery saturated while
> density=="dense". Pytest for should_inspect logic and JSON fallback only (no
> network).
**You verify:** run inspect_scene on a dense-canopy image and a sparse one;
compare the two JSONs against your own eyes.

### Step 5.2 — VLM policy with validation
**Files:** `agent/policy_vlm.py`, `tests/test_policy_vlm.py`.
**Prompt:**
> `policy_vlm.choose(phi, z, partition, graph, cfg) -> action`: one Qwen text
> call (no image) whose prompt contains: compact φ (drop per-node lists beyond
> top-15 by uncertainty), z, the partition with region ids, verifiable node ids,
> remaining budget, and a menu of legal actions with their exact JSON forms.
> Response must be JSON {"action": ..., "args": ...}. VALIDATE hard: region id
> must exist, node ids must be in graph and unresolved/low-w, prompt strings
> must equal the configured target concept, conf within [0.1, 0.9]. Any
> violation or parse failure -> log and fall back to policy_heuristic.choose.
> Never let the VLM inject boxes: there is no action that accepts coordinates
> from the model. Pytest: canned malformed / illegal responses all fall back;
> a canned legal response maps to the right dataclass.
**You verify:** run a full VLM episode with `--log-prompts`; read one full
prompt+response pair; confirm every executed action was validated (grep the log
for "fallback").

---

# Phase 6 — Evaluation

### Step 6.1 — Matching & detection metrics
**Files:** `eval/matching.py`, `tests/test_matching.py`.
**Prompt:**
> `match(pred_boxes, gt_boxes, iou_thr=0.5)` via scipy.optimize
> linear_sum_assignment on the IoU matrix (pairs below thr unmatched);
> return precision, recall, f1, matched flags. `pool_recall(all_candidate_boxes,
> gt)` and `per_pass_pool_recall(graph, gt)` using node.found_in_pass. Pytest
> with 4 hand-drawn toy cases including the nested-box case.
**You verify:** pytest; then matching on one real image vs GT overlay — matched
GT drawn green, missed drawn red, saved to out/.

### Step 6.2 — Sweep runner
**Files:** `eval/metrics.py`, `eval/run_eval.py`.
**Prompt:**
> metrics.py: mae, rmse, exact from paired counts; cost from CostMeter.
> run_eval.py CLI: --root --fmt --policy
> {oneshot, cascade, tiled, convergence, heuristic, vlm} --verifier {ioc,vip}
> --limit N --out csv. oneshot = 1 global pass; cascade = fixed 4 passes;
> tiled = fixed 4 passes tiling; convergence = cascade until n_t==0;
> heuristic/vlm = runner episodes. For each image write one CSV row: image,
> policy, verifier, N_gt, N_obs, N_supp, N_cons, precision, recall, f1,
> pool_recall, per-pass pool recalls (json col), cost, n_actions, seconds.
> Aggregate block printed at the end. Deterministic seeds. Resume-safe (skip
> images already in the CSV).
**You verify:** `--limit 5` on the citrus set for {oneshot, cascade} × {ioc, vip}
= 4 runs. Open the CSV: pool_recall(cascade) ≥ pool_recall(oneshot) must hold,
and vip-vs-ioc precision difference tells you immediately if the verifier earns
its cost.

### Step 6.3 — Results table & accuracy-vs-cost plot
**Files:** `eval/report.py`.
**Prompt:**
> Read one or more sweep CSVs; print a markdown table (rows = policy×verifier,
> cols = MAE, RMSE, Exact, F1, pool recall, mean cost) and save
> out/accuracy_vs_cost.png (matplotlib, one point per policy, MAE on y, cost on
> x, verifier as marker style). No seaborn.
**You verify:** the plot is the figure that goes in the paper — check that the
six-policy story (one-shot → cascade → tiled → convergence → heuristic → VLM)
is readable from it.

---

## Suggested order of the first week

0.1 → 0.2 → 1.1 → 1.3 → 1.2 → 1.4 (the go/no-go checkpoint: do the chains make
sense on real crops?) → 1.5 → 0.3 → 6.1 → 6.2 with {oneshot, cascade} only.
That gives you a quantified answer to "does FM+V-IP verification beat the IoC
gate" before any agent code exists. Phases 2–5 then only make sense if that
answer is yes or fixable-by-editing-queries.

## Known risks & their cheap mitigations

- **Qwen JSON drift** → strict schema + retry + all-zeros fallback (step 1.2);
  all-zeros answers make VIP return the prior, i.e. "unresolved", never a wrong
  confident verdict.
- **Conditional-independence violations** (correlated queries) → harmless for
  ranking but can overconfidence the posterior; if observed, raise ε or cap
  chain length via vip_max_queries. Both are config knobs, no code.
- **Verification cost blow-up on dense images** → VerifyA takes node_ids; the
  policies already select only unresolved/low-w nodes, and budget_max_actions
  hard-caps the episode.
- **torch.compile + many small verify crops is slow** → crops go through Qwen,
  not SAM3, in the default config; Sam3Oracle is opt-in.

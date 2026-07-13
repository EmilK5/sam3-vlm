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

---

# Phase 7 — Guided-ROI policy rework

Motivation: in the t9 VLM episode the policy wasted 9/12 actions on no-op
`subdivide`s and never stopped. Root cause was policy design, not model
strength (qwen3-vl-thinking is capable). This phase replaces id-based quadrant
subdivision with image-grounded region-of-interest proposals, seeds exemplars
with a mandatory global pass, and makes the runner terminate on its own.

Design decisions locked in with the user (do not relitigate):
- **One model only: qwen3-vl-thinking.** Never swap models. Disable thinking
  *per-call* via the endpoint's API arg for the structured policy-loop
  decisions; keep thinking on for `inspect` (z) and the `verify` oracle.
- **VLM sees the image** in the policy call and may emit region boxes as
  **sensing targets only**. A VLM box parameterizes a SAM3 query and is
  code-guaranteed never to become a graph node. Candidates stay SAM3-only.
  This amends hard-constraint #3 (done in 7.4).
- **Mandatory global (non-tiled) pass first**, seeding candidates + exemplars.
- **One ROI proposal = one real tiled sense** (+10% margin), so no action is a
  no-op. Refinement bounded to 2 nesting levels / a min ROI size (the user's
  "max 2 splits / 16 quadrants" rule). `subdivide` is removed.
- **Heuristic policy stays as the validated fallback.**

### Step 7.1 — LookROIA action + grounding enforcement
**Files:** `agent/actions.py`, `tests/test_look_roi.py`.
**Prompt:**
> Add `LookROIA(region)` where region is a VLM-proposed xyxy box (a sensing
> target, never a candidate). In `execute()`: expand region by `cfg.roi_margin`
> (default 0.10) about its center, clamp to image bounds, then call
> `pipeline.execute_pass` with `roi_override=<expanded box>` and tiling on, using
> the graph's current exemplars; return n_new. HARD GUARANTEE: no code path adds
> the ROI box itself to the graph — candidates come only from SAM3 inside the
> ROI. Track sensed ROIs on ctx; return 0 (log, no model call) when the ROI is
> below `cfg.roi_min_size`, nested past `cfg.roi_max_depth` (=2) levels, or
> overlaps an already-sensed ROI above an IoU threshold. Pytest with a stub
> pipeline + MockOracle: margin/clamp math is correct; a LookROIA never creates a
> node from its own coordinates; a too-small / duplicate ROI is a no-op.
**You verify:** REPL: `LookROIA` on a hand-picked dense corner of a real image;
the overlay shows SAM3 detections only inside the 10%-expanded box, and the ROI
box itself is not a node in the graph JSON.

### Step 7.2 — Bootstrap global pass + runner auto-stop
**Files:** `agent/runner.py`, `tests/test_runner_bootstrap.py`.
**Prompt:**
> Add an opt-in `bootstrap_global_pass` to `run_episode` (default off so the
> heuristic-baseline tests are unaffected): when set, execute one global
> non-tiled `QueryA` over the full tree_roi at `cfg.conf` before the policy loop,
> seeding candidates + exemplars, logged as its own step. Add runner-level
> auto-stop backstops that end the episode regardless of the policy's action:
> discovery saturation (a full sensing pass adds < `cfg.delta_disc` new) OR all
> proposed ROIs sensed OR budget exhausted. Keep one JSON log line per step.
> Pytest with a stub executor: with bootstrap on, the first executed action is
> the global query; a saturating discovery script auto-stops even when the
> policy never returns StopA.
**You verify:** run a VLM episode; the first log line is the global pass; a
saturating run terminates without the model emitting `stop`.

### Step 7.3 — Guided-ROI VLM policy
**Files:** `agent/policy_vlm.py`, `tests/test_policy_vlm_roi.py`.
**Prompt:**
> Rework `policy_vlm.choose` to send the image (base64 data URL, same encoding
> `inspect_scene` uses) alongside phi/z. New menu: `look{region:[x1,y1,x2,y2]}`
> (sensing ROI only), `tile{conf}`, `verify{node_ids}` (vip only), `stop`.
> Remove `subdivide`. Hard-validate `look`: region is four numbers, in-bounds
> after clamp, passes the 7.1 depth/min-size/overlap guards -> `LookROIA`; any
> parse/validation failure -> heuristic fallback (unchanged path). The grounding
> guarantee holds: the look box is a sensing target, never a candidate. Pytest
> with a fake client + stub: a legal `look` maps to `LookROIA` with the proposed
> box; out-of-bounds / too-small / duplicate `look` falls back; the request
> includes the image; `subdivide` is absent from the emitted menu.
**You verify:** `run_eval --policy vlm --limit 1`; read the trace — after the
bootstrap global pass, every `look`/`tile`/`verify` senses (n_new changes or
exemplars grow), there are no no-op spins, and it stops on saturation/budget.

### Step 7.4 — Config, thinking toggle, and constraint reword
**Files:** `config.py`, `CLAUDE.md`.
**Prompt:**
> config.py: add `roi_margin=0.10`, `roi_min_size` (px), `roi_max_depth=2`, and a
> per-callsite thinking toggle — a helper that appends the endpoint's
> disable-thinking argument to a `chat.completions` call (verify the exact param
> against the live Ollama qwen3-vl endpoint: likely
> `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` or a `think`
> field). Default: thinking ON for `inspect` + the `verify` oracle, OFF for
> policy-loop decisions. Keep a single model id; never switch models. CLAUDE.md:
> reword hard-constraint #3 to "candidates originate only from SAM3; no VLM box
> may become a candidate/node; VLM boxes are allowed solely as sensing ROIs that
> parameterize a SAM3 query."
**You verify:** grep a policy-loop request log — the disable-thinking arg is
present and the response carries no thinking trace; `inspect`/`verify` requests
still think. `pytest -q` green.

---

# Phase 8 — v2: full-history amortized policy (active-perception formulation)

Motivation: `docs/active_perception_formulation.md` recasts the system as He
et al.-style active perception — the sensing action $x_t$ is a SAM3 query, and
Qwen-3-VL is an amortized policy $\pi_\theta(u_t \mid \xi, x_{past}, y_{past})$
that must therefore SEE the full past $x_1^t, y_1^t$. v2 collapses the agent to
exactly that formulation and simplifies everything around it. Branch: `v2`.

Design decisions locked in with the user (2026-07-10, do not relitigate):
- **Full history each step, rebuilt prompt.** Every policy call is one
  stateless prompt containing: the raw frame, an annotated overlay image
  (current candidate boxes colored by class + already-sensed ROIs + tree ROI),
  and a JSON log of every past action $x_1^t$ and observation $y_1^t$ (ROI,
  n_new, new boxes with id/conf/class, running totals) plus the current belief
  summary. No multi-turn chat state; each prompt is self-contained and
  auditable from the episode log.
- **One policy action: `look`.** Menu = {`look(region)`, `stop`}. The region is
  any xyxy box inside the tree ROI; a look = ONE untiled SAM3 query on the
  (margin-expanded, clamped) ROI — no tiling, since selecting a box already IS
  tiling. `TileQueryA` / `SubdivideA` / `VerifyA` are **deleted** on v2 (a
  user-approved amendment of the "old path stays default" rule for these three;
  the fixed run_eval baselines call `execute_pass` directly and are unaffected).
  Verification stays automatic inside `execute_pass` (`verifier_mode`), never a
  policy action. Runner auto-stop (saturation/budget) stays as the backstop.
- **Fixed 3-action bootstrap** before the policy ever acts, each recorded as an
  explicit history entry the VLM will see: (1) canopy/tree-ROI detection,
  (2) global leaf map, (3) one global untiled SAM3 pass on the tree ROI.
  (Physically these already happen at episode setup / inside the first
  `execute_pass`; v2 makes them explicit $x_1..x_3$ records with observations.)
- **V-IP answers routed to the cheapest able channel.** A per-query `route`
  field in the query-set JSON: `"cv"` (deterministic crop/mask features —
  color/shape/texture, no model call), `"sam3"` (presence score via
  `sam3_phrase`), `"vlm"` (default). A `RouterOracle` composes the channels;
  Qwen receives only the residual `vlm` queries, in one batched call.
- **`inspect_scene` / z is retired.** The full-history prompt subsumes the
  separate scene-inspection call; `agent/inspect.py` and its wiring are deleted.
- Grounding invariant unchanged: candidates originate only from SAM3; a VLM
  region is a sensing target only (CLAUDE.md hard-constraint #3).
- `orchestration_app.py` imports the deleted actions; it gets a minimal import
  fix only (no restructuring) in 8.5.

### Step 8.1 — Action space v2: untiled look, delete legacy actions
**Files:** `agent/actions.py`, `agent/policy_heuristic.py`,
`tests/test_actions_v2.py`; migrate/trim stale tests: `tests/test_actions.py`,
`tests/test_policy_heuristic.py`, `tests/test_budget.py`,
`tests/test_review_fixes.py`.
**Prompt:**
> In `agent/actions.py`: change `_execute_look` to run ONE untiled SAM3 query
> (`tiling=False`) on the margin-expanded ROI via `execute_pass`
> `roi_override` — everything else about LookROIA (margin, clamp, min-size /
> depth-floor / duplicate guards, grounding invariant, metering) is unchanged.
> Delete `TileQueryA`, `SubdivideA`, `VerifyA`, `subdivide_region`,
> `_execute_subdivide`, `_execute_verify`, and their `execute()` branches.
> `QueryA` stays (it is the bootstrap/global sensing primitive). Rewrite
> `policy_heuristic.choose` as the minimal look/stop fallback: stop when
> discovery is saturated (or the graph is empty of unsensed evidence), else
> look at the cell of a fixed 2x2 grid over `partition[0]` (the tree ROI) with
> the fewest registered candidates, cycling cells so repeats don't hit the
> duplicate-ROI guard. Migrate the named test files (delete tests of removed
> actions; keep every guard/grounding/metering test alive). New
> `tests/test_actions_v2.py`: a LookROIA executes exactly one untiled pass
> (stub pipeline asserts `tiling is False` and `roi_override` equals the
> expanded box); the removed action names are gone from the module; the
> heuristic emits only LookROIA/StopA and stops on a saturated curve.
**You verify:** REPL LookROIA on a dense corner of a real image — overlay shows
detections only inside the expanded box, log shows a single global (not tiled)
inference; `pytest -q` green.

### Step 8.2 — Episode history record + explicit 3-action bootstrap
**Files:** `agent/history.py` (new), `agent/runner.py`,
`tests/test_history.py`; migrate `tests/test_runner_bootstrap.py`.
**Prompt:**
> New `agent/history.py`: an `EpisodeHistory` (list of plain dicts, JSON-ready)
> with one record per executed action:
> `{t, x: {action, params...}, y: {n_new, new_nodes: [{id, box, conf, class}],
> totals: {K, N_obs}}}` — boxes in global-frame xyxy pixels; `new_nodes` are the
> graph nodes whose `found_in_pass` equals the pass just executed.
> In `runner.run_episode`: store the history on `ctx.history`; after every
> executed action append its record. Replace the single bootstrap QueryA with
> the fixed 3-action bootstrap, always on for this runner path (flag
> `bootstrap_global_pass` may remain as the switch, still counted against
> budget as ONE sensing action — the canopy and leaf-map records document what
> the pass did, they are not separately billed): record (1) `canopy_roi` with
> the tree ROI box actually used (`graph.tree_roi` or full frame), (2)
> `leaf_map` with the leaf-box count (`graph.cached_leaf_boxes`), (3)
> `global_pass` with the standard observation record. Keep one JSON log line
> per record. Policy callable signature stays `(phi, partition, cfg)`; policies
> that need history read `ctx.history` via their closure (make_vlm_policy).
> Pytest with a stub executor: bootstrap yields exactly 3 leading records with
> those action names; a look appends a record whose `new_nodes` match the stub
> graph; records are `json.dumps`-able.
**You verify:** run one episode; the log's first three lines are
canopy_roi/leaf_map/global_pass with a sane tree ROI box and leaf count;
history length == log length.

### Step 8.3 — Full-history VLM policy + overlay image
**Files:** `agent/policy_vlm.py`, `agent/overlay.py` (new), `agent/runner.py`
(drop inspect wiring), delete `agent/inspect.py` + `tests/test_inspect.py`;
`tests/test_policy_vlm_v2.py`; migrate/trim `tests/test_policy_vlm.py`,
`tests/test_policy_vlm_roi.py`.
**Prompt:**
> New `agent/overlay.py`: `render_overlay(image_pil, graph, sensed_rois,
> tree_roi) -> PIL.Image` — PIL ImageDraw only; candidate boxes colored by
> classification (fruit/leaf/spurious/unresolved), sensed ROIs and the tree ROI
> in distinct styles; no matplotlib, no model. Rework `policy_vlm`: `choose(phi,
> history, graph, cfg, client=None, image=None, sensed_rois=None)` builds ONE
> stateless prompt per step: system prompt = active-perception orchestrator
> (you pick the next region $x_t$ to sense, guided by the FULL past); user
> content = JSON body {target_concept, image_size, tree_roi, history (complete
> $x_1^t, y_1^t$), phi (compact belief), action_menu: {look: {region:
> \[x1,y1,x2,y2\]}, stop: {estimate_name}}} + the raw image + the overlay
> image (two image parts). Delete the z/inspect plumbing and the tile/verify
> menu entries. Validation: look region must be four numbers inside the tree
> ROI (not merely the frame); stop only on a non-empty graph; any
> parse/validation failure falls back to `policy_heuristic.choose` (unchanged
> safety net). `runner.make_vlm_policy` drops should_inspect/inspect_scene and
> passes `ctx.history` + `ctx.sensed_rois`. Pytest with a fake client: the
> request carries exactly two image parts and the full history JSON verbatim;
> a legal look inside the tree ROI maps to LookROIA; a look outside the tree
> ROI (but inside the frame) falls back; menu contains only look/stop.
**You verify:** run a VLM episode with prompt logging; read one full prompt —
history matches the episode log line-for-line, overlay image opens and shows
the boxes; every executed action validated (grep for "fallback").

### Step 8.4 — V-IP routing: CV + SAM3 channels, RouterOracle
**Files:** `verifier/queries.py`, `verifier/cv_answers.py` (new),
`verifier/oracle.py`, `queries/green_citrus.json`,
`tests/test_oracle_routing.py`.
**Prompt:**
> `queries.py`: add optional per-query `route` in {"cv","sam3","vlm"} (default
> "vlm"; validate: route=="sam3" requires `sam3_phrase`; route=="cv" requires a
> new `cv_check` dict naming a feature + thresholds). New
> `verifier/cv_answers.py`: deterministic features on the 256px crop —
> `hue_fraction(lo,hi)` (color), `circularity` of the dominant contour after
> Otsu/HSV threshold (shape), `edge_density` via cv2.Laplacian (texture);
> `answer(crop_pil, cv_check) -> {-1,0,+1}` with a yes-above / no-below /
> unsure-between threshold pair. New `RouterOracle(cfg, processor=None,
> vlm_oracle=None)` in `oracle.py` exposing the standard
> `answer_batch(crop_pil, query_set)`: cv queries answered locally; sam3
> queries by the existing presence-score pattern (0 when no processor); the
> residual vlm queries answered by ONE `QwenOracle` call on a subset view of
> the query set (0-fill when no vlm oracle); expose `n_vlm_calls` (0 or 1) and
> `n_sam_calls` from the last batch so verify metering counts only real model
> calls. Update `queries/green_citrus.json`: route the color/shape/texture
> queries to cv, part-presence queries to sam3 (add phrases), leave the
> genuinely semantic ones on vlm — the human re-reviews the JSON by hand.
> V-IP math (`vip.py`, `verify.py`) unchanged: only the SOURCE of answers
> moves. Pytest: routing partitions the query ids exactly; a synthetic green
> disc crop answers yes-green/yes-round without any oracle; the fake vlm
> client receives ONLY the vlm-routed queries; missing channels yield 0
> (never a wrong confident answer).
**You verify:** `scripts/demo_verify.py --oracle router` (add the choice) on a
clear fruit crop and a leaf crop — chains sensible, log shows ≤1 Qwen call per
candidate; hand-review the routed green_citrus.json.

### Step 8.5 — v2 wiring: config, eval, apps, docs
**Files:** `config.py`, `eval/run_eval.py`, `agent/runner.py`,
`orchestration_app.py` (import fix only), `PROGRESS.md`,
`tests/test_v2_wiring.py`.
**Prompt:**
> `config.py`: remove knobs orphaned by the deletions (anything only the
> tile/subdivide/verify menu used), add `oracle_kind`/router option and any 8.4
> thresholds that were getattr defaults. `run_eval`: `--policy vlm` runs the v2
> episode (3-record bootstrap + look/stop loop + auto-stop) and `--oracle
> router` wires RouterOracle; heuristic/fixed policies keep working.
> `orchestration_app.py`: fix imports of deleted actions minimally (drop the
> broken menu paths). Offline end-to-end pytest with stub processor +
> MockOracle + fake VLM client: a full v2 episode produces history =
> [canopy_roi, leaf_map, global_pass, look..., stop/auto-stop], counts sane,
> cost metered, everything `json.dumps`-able.
**You verify:** `run_eval --policy vlm --oracle router --limit 1` on a real
image end-to-end; read the episode JSON — bootstrap trio first, every look is
inside the tree ROI, Qwen verify calls ≤ 1 per candidate, episode stops on its
own.

## Phase 9 — Prompt-refinement active loop (VLM refines the SAM3 text prompt)

**Design (locked with the human).** The active arm senses the SAME region every
step — the canopy tree ROI, ONE global pass — and the VLM refines *what* it asks
SAM3: a 1-2 adjective + noun text prompt and a detection threshold. It is NOT a
sub-ROI zoom policy. Rationale: (a) a global tree-ROI pass keeps every found
positive exemplar in-frame, so exemplar priming is never lost — SAM3 exemplars are
welded to the query frame (roi_align'd against that image's own features; there is
no cross-image exemplar conditioning, confirmed against the HF `modeling_sam3.py`
source), so a sub-ROI crop would silently drop out-of-crop exemplars; (b) fixing
the region isolates the causal effect of prompt refinement from any tiling/
ensembling confound, which is exactly the claim under test. Sub-ROI zoom may be
re-added later behind a flag. Purpose of the phase: a three-window GUI (ground
truth | fixed generic cascade | VLM-refine loop) to build intuition on whether
prompt refinement helps; the statistical claim (adaptive vs. equal-budget random/
fixed prompt schedule) is a later `eval/` ablation, NOT this GUI.

### Step 9.1 — v3 prompt-refinement policy + config
**Files:** `config.py`, `agent/policy_vlm_v3.py` (new),
`tests/test_policy_vlm_v3.py` (new).
**Prompt:**
> `config.py`: add a phase-9 group — `seed_conf=0.65` (confident-seed / bootstrap
> threshold), `refine_conf_default=0.40`, `refine_conf_min=0.30`,
> `refine_conf_max=0.70`, `refine_min_words=1`, `refine_max_words=3`. New
> `agent/policy_vlm_v3.py`: `choose(phi, history, graph, cfg, client=None,
> image=None)`, shape mirroring `policy_vlm` but menu = {refine, stop}. Body =
> {initial_concept, image_size, tree_roi, prompts_tried (distilled prompt→yield
> trajectory), history (full x/y), phi, action_menu} + raw image + overlay (two
> image parts). A legal `refine` validates the prompt to a
> refine_min_words..refine_max_words plain-word noun phrase and clamps the
> threshold to [refine_conf_min, refine_conf_max] (default when missing/non-numeric),
> then returns a GLOBAL `QueryA(region=tree_roi, prompt, conf)` — text only, never a
> box (CLAUDE.md #3). `stop` obeys the non-empty-graph guard + valid estimator.
> Any parse/validation failure OR a refine with no image falls back to
> `policy_heuristic.choose` (unchanged safety net). Reuse policy_vlm's request
> plumbing (`_build_client/_request/_data_url/_compact_phi/_history_records/
> _resolve_tree_roi/_log_text/_coerce_json_string/_ESTIMATORS`); do NOT modify
> policy_vlm (v2 look/stop stays the default policy). Pytest with a fake client:
> two image parts + full history + prompts_tried in the body; menu is refine/stop;
> a legal refine → global QueryA over the tree ROI with normalized prompt + clamped
> conf; missing threshold → default; over-long/empty prompt → fallback; empty-graph
> stop → fallback; no image → refine absent from the menu and a refine response
> falls back.
**You verify:** read one built prompt (log it) — the body carries the raw+overlay
images, the full history, and the prompt→yield trajectory; the menu is refine/stop;
`pytest -q` green; grep the run for "fallback" on a healthy episode (none).

### Step 9.2 — Three-window citrus experiment app
**Files:** `citrus_orchestration_app.py`, `agent/runner.py` (additive
`make_refine_policy`), `tests/test_refine_app.py` (new).
**Prompt:**
> `agent/runner.py`: add `make_refine_policy(ctx, vlm_client=None)` mirroring
> `make_vlm_policy` but dispatching to `policy_vlm_v3.choose` (no sensed_rois);
> `make_vlm_policy` unchanged. `citrus_orchestration_app.py`: render THREE panels
> per image — (1) ground truth (existing `draw_gt_view`); (2) generic SAM3 arm: a
> fixed two-call cascade on ONE prompt — canopy ROI → leaf map → global pass @
> `seed_conf` (seeds confident exemplars) → tiled pass @ `refine_conf_default`;
> (3) active arm: the v3 refine loop (bootstrap global @ seed_conf on the tree ROI,
> then `make_refine_policy` with `bootstrap_global_pass=True, auto_stop=True` for
> the hybrid VLM-stop / saturation / budget termination). Both arms run on separate
> OrchardGraphs; show each arm's N_obs vs GT and stream the VLM prompt→yield trace
> to the verbose log. Offline pytest with a stub processor + stubbed VLM client:
> the three views render and the active arm's history is
> [canopy_roi, leaf_map, global_pass, (refine…), stop/auto-stop] with every refine
> a global QueryA over the tree ROI.
**You verify:** launch the app on the GPU box; step a few images — the generic arm
matches the old cascade, the active arm's log shows the prompt evolving
(green fruit → …) with N_obs moving toward GT; the region never changes (always
the tree ROI); no "fallback" spam on a healthy run.

### Step 9.3 — Per-prompt dedup feedback + distinct-prompt rule
**Files:** `agent/actions.py`, `agent/runner.py`, `agent/history.py`,
`agent/policy_vlm_v3.py`, `citrus_orchestration_app.py`,
`tests/test_prompt_feedback.py` (new); + `tests/test_policy_vlm_v3.py` update.
**Prompt:**
> The VLM can only judge a prompt if it sees how many ALREADY-KNOWN objects that
> prompt re-found (dedup re-detections), not just how many new ones it added.
> Thread `PassStats.duplicates_rejected` (and `post_nms`) through to the policy.
> `actions.py`: add `ActionContext.last_pass_stats` (default None); `execute` clears
> it each action; `_execute_query` stashes `{n_new, n_redetected(=duplicates_rejected),
> n_detections(=post_nms)}`. `runner.py`: `_obs(ctx, n_new, new_nodes)` builds the
> observation y including n_redetected/n_detections (zeros when no pass ran); used for
> the bootstrap global_pass record and every loop record; canopy_roi/leaf_map carry
> zeros. `history.py`: document the new y fields. `policy_vlm_v3.py`: `_prompts_tried`
> now carries {prompt, conf, n_detections, n_new, n_redetected}; add `_tried_prompt_set`;
> `_parse_and_validate` rejects a refine prompt already tried (normalized) so each step
> tries a NEW wording; system prompt tells the VLM to judge a prompt by n_redetected
> (good wording re-finds most known targets AND adds new) and that its prompt must
> differ from all tried. `citrus_orchestration_app.py`: show det/new/redet per step in
> the trajectory log. Pytest (CPU, `pipeline` stubbed in sys.modules / execute_fn
> injected): `_execute_query` stashes the feedback; a Stop clears it; run_episode's y
> carries n_redetected/n_detections; a repeated prompt falls back; prompts_tried shows
> the dedup counts.
**You verify:** on the GPU box, read one active-arm prompt — prompts_tried lists each
prior wording with n_detections/n_new/n_redetected; the VLM never repeats a wording
(each step is new); watch a run where an early prompt has low n_redetected and the
VLM pivots to a wording that re-finds more of the known fruit.

### Step 9.4 — Ban LookROIA in the active arm + tiling for recall
**Files:** `config.py`, `agent/actions.py`, `agent/runner.py`,
`agent/policy_vlm_v3.py`, `citrus_orchestration_app.py`,
`tests/test_active_arm_tiling.py` (new); + `tests/test_policy_vlm_v3.py` migration.
**Prompt:**
> Observed: the active arm's fallback called policy_heuristic, which emits LookROIA
> (a sub-ROI zoom that drops exemplars and finds nothing), and recall trailed the
> generic cascade because the arm never tiled. Fix both. `policy_vlm_v3._fallback`
> returns `StopA` (never the look heuristic) so the arm's action space is strictly
> {refine, stop} -- LookROIA can never appear; drop the empty-graph stop guard (the
> bootstrap always senses first, so an empty graph == empty orchard, stop with 0 is
> correct). `config.py`: `refine_tiling=True`. `actions.py`: `QueryA.tiling` field
> (default False, additive); `execute` forwards it. `policy_vlm_v3`: a refine builds
> `QueryA(..., tiling=cfg.refine_tiling)` over the whole tree ROI (tiling is the recall
> mechanism, NOT a sub-ROI zoom); system prompt tells the VLM to keep proposing new
> wordings and that the only actions are refine/stop. `runner.py`: add
> `bootstrap_tiled_pass` -- after the global seed, run one TILED pass over the tree ROI
> with the seed prompt at refine_conf_default (a "tiled_seed_pass" record) so the arm's
> recall floor equals the generic cascade before any refine. `citrus_orchestration_app`:
> run_active_arm sets bootstrap_tiled_pass=True. Pytest (CPU, pipeline stubbed /
> execute_fn injected): QueryA.tiling threads to execute_pass; bootstrap adds the tiled
> seed record (untiled global then tiled seed); a refine is tiled (off when
> refine_tiling=False); an invalid response returns StopA, never LookROIA.
**You verify:** on the GPU box, the active-arm trajectory shows canopy_roi / leaf_map /
global_pass / tiled_seed_pass then refines -- NO LookROIA anywhere; active-arm recall
is >= the generic cascade (the tiled seed floor guarantees it) and rises further when
the VLM finds a good wording; an invalid/repeated VLM reply ends the arm cleanly (a
single "stopping" log line, not LookROIA spam).

### Step 9.5 — Refine threshold rises with each new prompt
**Files:** `config.py`, `agent/policy_vlm_v3.py`, `tests/test_refine_threshold.py`
(new); + `tests/test_policy_vlm_v3.py` / `tests/test_runner_refine.py` migration.
**Prompt:**
> With tiling added, the old 0.40 refine threshold admits too much clutter on later
> passes (more, smaller boxes). Raise the range and make the threshold FLOOR rise per
> new prompt. `config.py`: refine_conf_default 0.40->0.50, refine_conf_min 0.30->0.45,
> refine_conf_max 0.70->0.85, add refine_conf_step=0.05. `policy_vlm_v3`: add
> `_n_prior_refines(history)` (count of QueryA records); `_validate_threshold(threshold,
> cfg, n_prior_refines)` computes floor = clamp(refine_conf_default + n_prior_refines*
> step, [min,max]) and returns clamp(chosen, [floor, max]) (missing/non-numeric ->
> floor). So each new prompt is at least one step stricter; the VLM may go higher but
> never lower. System prompt tells the VLM to use higher thresholds later. Pytest: a
> below-floor / missing value is raised to the floor; the floor rises by step per prior
> refine; a stricter choice is kept; the ceiling is refine_conf_max.
**You verify:** on the GPU box, the active trajectory shows refine thresholds climbing
(≈0.50, 0.55, 0.60 …) and fewer clutter boxes on later passes than before; the tiled
seed floor and the generic tiled pass both now run at 0.50.

### Step 9.6 — Per-pass animation + separate leaf map (app)
**Files:** `citrus_orchestration_app.py`.
**Prompt:**
> Add a scrollable per-pass animation and a separate leaf-map panel to the citrus app.
> `draw_sam3_view` gains `max_pass=None` -> draw only nodes with found_in_pass <=
> max_pass (the graph is monotone, so this reconstructs the belief AS OF each pass).
> `render_pass_frames(image, graph, history)` -> list of (image, caption), one per
> sensing pass (global_pass/tiled_seed_pass/QueryA in order; found_in_pass == the pass
> index), captioned with action/prompt/threshold and that pass's det/new/redet + N_obs.
> `render_leaf_map(image, graph)` -> the cached_leaf_boxes (ROI-relative -> global) on a
> separate frame. Wire a gr.Gallery (active arm, scroll pass by pass) + a gr.Image (leaf
> map) into run_experiment's outputs and clear them on navigation. App-only; GPU
> hand-verified (the app loads SAM3 at import, so no offline test).
**You verify:** run the app on the GPU box; the gallery lets you click pass 1..N and
watch boxes accumulate as the prompt/threshold change; the leaf map shows the green-leaf
inhibitors on its own panel; navigating to a new image clears both.

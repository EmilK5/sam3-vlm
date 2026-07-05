# Manual test checklist — per step

Two tiers per step:
- **Tier A — CPU pytest.** No models, no network; runs anywhere. `pytest -q`
  for the whole suite should print **204 passed** in a couple of seconds.
- **Tier B — real SAM3 + Qwen VLM via Ollama.** Runs on the GPU box. The
  thorough, copy-paste walkthrough (with expected output and what to send back
  when something fails) is [tier_b_testing.md](tier_b_testing.md) — steps below
  reference its test IDs (**T0–T12**).

## Setup (Tier B)

Start Ollama with a vision-capable Qwen model (details + context-length notes
in [tier_b_testing.md](tier_b_testing.md) Part 1), then:

```bash
export QWEN_BASE_URL=http://localhost:11434/v1
export QWEN_MODEL=qwen3-vl:8b     # must exactly match `ollama list`
                                  # (fallback tag: qwen2.5vl:7b — no hyphen)
export IMG=dataset/images/val/REPLACE.png
export STEM=$(basename "$IMG" | sed 's/\.[^.]*$//')
```

No API key needed. vLLM works too (appendix of the Tier B guide); only the two
env vars change.

---

## Phase 0 — scaffold

### 0.1 config
- **A:** `pytest -q tests/test_smoke.py`
- **B:** `python -c "from config import Config; print(Config())"` — all fields
  sane; `oracle_base_url` echoes `QWEN_BASE_URL`; `overlap_mode='box'` and
  `vip_epsilon=None` are the defaults (**T0**).

### 0.2 run_image CLI
- **A:** `pytest -q tests/test_run_image.py`
- **B:** **T1** — 3-pass run + overlay + graph JSON; node count matches the
  logged pass stats.

### 0.3 dataset loader
- **A:** `pytest -q tests/test_datasets.py`
- **B:** `python -m eval.datasets --root dataset --fmt yolo --split val` → GT
  overlay (cyan) sits on real fruit; `--draw-gt` covered inside **T1**.

---

## Phase 1 — FM+V-IP verifier

### 1.1 query set
- **A:** `pytest -q tests/test_queries.py`
- **B (read):** open `queries/green_citrus.json` — every query answerable from
  a 256px crop; T/D/S templates match intuition. Optional `sam3_phrase` fields
  are allowed per query (they enable the Sam3Oracle channel).

### 1.2 oracles
- **A:** `pytest -q tests/test_oracle.py`
- **B:** **T2** — QwenOracle on one fruit crop; answers visibly sane
  (round=+1, veins=−1). All-zeros = parse/request fallback fired; see the log.

### 1.3 training-free V-IP core
- **A:** `pytest -v tests/test_vip.py` — the four test names ARE the spec.

### 1.4 verify API + demo  ← GO/NO-GO
- **A:** `pytest -q tests/test_verify.py`
- **B:** **T3** — three chains (fruit/leaf/junk) read like sensible reasoning.
  Tweak `queries/green_citrus.json` templates/epsilon and re-run if not; or
  override epsilon without editing the JSON via `Config(vip_epsilon=0.25)`.

### 1.5 pipeline integration (verifier flag)
- **A:** `pytest -q tests/test_pipeline_vip.py`
- **B:** **T4** — same image under `--verifier ioc / vip / off`; diff overlays
  and graphs; spot-check differing verdicts with `demo_verify`. The `off` run
  is also the dedup regression check: node count must stay ~flat across passes.

---

## Phase 2 — belief state

### 2.1 support / jitter / signatures
- **A:** `pytest -q tests/test_graph_support.py`
- **B:** **T5** (first half) — top-10 nodes by support: stable fruit ≥ 2,
  one-off junk == 1; signatures record `pass:mode:prompt:conf`.

### 2.2 belief.py (w, U, discovery, φ)
- **A:** `pytest -q tests/test_belief.py`
- **B:** **T5** (second half) — U visibly drops across 3 passes on an easy
  image.

---

## Phase 3 — actions & cost

### 3.1 action layer
- **A:** `pytest -q tests/test_actions.py`
- **B:** **T6** — one `QueryA` on the top-left quadrant: detections only inside
  that quadrant. Note: episode pass numbers advance only with sensing actions
  (`ctx.n_passes`), not with Verify/Subdivide/Stop.

### 3.2 cost meter
- **A:** `pytest -q tests/test_budget.py`
- **B:** **T6** — after the single region query expect
  `n_sam == 2, n_tile == 0, n_orch == 1 → total 2.05`. The meter now counts
  *actual* SAM3 calls from `PassStats` (canopy when run + leaf map when
  generated + proposal + real vip oracle calls), identically for fixed policies
  and episodes.

---

## Phase 4 — heuristic controller

### 4.1 VoI heuristic + episode runner
- **A:** `pytest -q tests/test_policy_heuristic.py`
- **B:** **T7** — episodes via
  `python -m eval.run_eval --policy heuristic ...` on sparse/dense/empty
  images: dense triggers `TileQueryA`/`SubdivideA`, empty stops within ~3
  actions. Under `--verifier ioc/off` the trace must contain **no `VerifyA`**
  (verify is vip-only now) and must not crash. Episodes anchor their partition
  to the canopy ROI automatically.

---

## Phase 5 — VLM orchestrator

### 5.1 scene inspection z_t
- **A:** `pytest -q tests/test_inspect.py`
- **B:** **T8** — `inspect_scene` on a dense vs a sparse image; compare the two
  JSONs against your own eyes. Persistent `notes='fallback'` on a healthy
  server is reportable.

### 5.2 VLM policy with validation
- **A:** `pytest -q tests/test_policy_vlm.py`
- **B:** **T9** — full VLM episode; read one prompt+response pair; grep the log
  for `fallback`. With `--verifier ioc` the prompt's action menu must not
  contain `verify`.

---

## Phase 6 — evaluation

### 6.1 matching & metrics
- **A:** `pytest -q tests/test_matching.py`
- **B:** run `match()` on one image vs GT (draw matched green / missed red).

### 6.2 sweep runner
- **A:** `pytest -q tests/test_eval_sweep.py`
- **B:** **T11** — `--limit 5` sweep; `pool_recall(cascade) ≥
  pool_recall(oneshot)`; the vip-vs-ioc precision delta shows if the verifier
  earns its cost. `--prompt`/`--conf` now apply to heuristic/vlm too. Costs sit
  on one unified scale (oneshot ≈ 3.0, not 1.0 — canopy + leaf map counted).

### 6.3 report + plot
- **A:** `pytest -q tests/test_report.py`
- **B:** **T11** — `python -m eval.report --csv out/sweep.csv`; the
  accuracy-vs-cost plot is the paper figure; the six-policy story must be
  readable. Mask-mode rows appear as `verifier="ioc+mask"` etc.

---

## 2026-07-05 review fixes + mask-overlap switch

### Functional-review fixes
- **A:** `pytest -q tests/test_review_fixes.py` (13 tests: VerifyA guard +
  vip-only gating, ROI-keyed leaf cache, unresolved dedup under vip/off,
  PassStats call counts, vip_epsilon override, request-exception fallbacks,
  sam3_phrase round-trip).
- **B:** **T12** — (a) `--policy heuristic --verifier off` completes without a
  traceback; (b) `--policy convergence --verifier off` terminates in < 8
  passes with a sane `N_obs`; (c) a custom `target_prompt` shows up in the
  episode's node signatures.

### Mask-based IoU/IoM (`overlap_mode="mask"`)
- **A:** `pytest -q tests/test_mask_overlap.py` (9 tests: mask IoU math, mask
  dual-gate NMS + alignment through the confidence filter, mask dedup,
  execute_pass round trip).
- **B:** **T10** — `run_image.py --overlap-mode mask` on a clustered image:
  outputs tagged `_ioc_mask_*`; touching-but-distinct fruits survive that box
  mode suppressed; `--tiling --overlap-mode mask` warns and falls back to box.

---

## Full runs

Whole CPU suite:
```bash
pytest -q            # expect 204 passed
```

Full six-policy sweep (needs the Ollama server for vip / vlm; resume-safe):
```bash
for pol in oneshot cascade tiled convergence heuristic vlm; do for ver in ioc vip; do
  python -m eval.run_eval --root dataset --fmt yolo --split val --limit 5 \
    --policy $pol --verifier $ver --prompt "green fruit" --out out/sweep.csv
done; done
# optional mask-mode line for the figure:
python -m eval.run_eval --root dataset --fmt yolo --split val --limit 5 \
  --policy cascade --verifier ioc --overlap-mode mask --out out/sweep.csv
python -m eval.report --csv out/sweep.csv --out out/accuracy_vs_cost.png
```

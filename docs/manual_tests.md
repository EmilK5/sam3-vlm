# Manual test checklist — per step

Two tiers per step:
- **Tier A — CPU pytest.** No models, no network; runs anywhere. `pytest -q` for
  the whole suite should print **175 passed**.
- **Tier B — real SAM3 + Qwen VLM.** Runs on the GPU box with a VLM server up
  (vLLM or Ollama — see [tier_b_testing.md](tier_b_testing.md)).

## Setup (Tier B)

Point the project at your VLM server (no API key needed):

```bash
# Ollama
export QWEN_BASE_URL=http://localhost:11434/v1
export QWEN_MODEL=qwen2.5-vl:7b        # must match `ollama list`
# (vLLM: QWEN_BASE_URL=http://localhost:8000/v1, QWEN_MODEL=qwen3-vl)

export IMG=dataset/images/val/REPLACE.png
export STEM=$(basename "$IMG" | sed 's/\.[^.]*$//')
```

---

## Phase 0 — scaffold

### 0.1 config
- **A:** `pytest -q tests/test_smoke.py`
- **B:** `python -c "from config import Config; print(Config())"` — all fields sane;
  `oracle_base_url` is empty until you `export QWEN_BASE_URL`.

### 0.2 run_image CLI
- **A:** `pytest -q tests/test_run_image.py`
- **B:** `python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 3`
  → overlay resembles the Gradio app; `out/${STEM}_ioc_graph.json` node count
  matches the logged pass stats.

### 0.3 dataset loader
- **A:** `pytest -q tests/test_datasets.py`
- **B:** `python -m eval.datasets --root dataset --fmt yolo --split val`
  → GT overlay (cyan) sits on real fruit. Also
  `python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 1 --draw-gt`
  writes `out/${STEM}_gt.jpg`.

---

## Phase 1 — FM+V-IP verifier

### 1.1 query set
- **A:** `pytest -q tests/test_queries.py`
- **B (read):** open `queries/green_citrus.json` — every query answerable from a
  256px crop; T/D/S templates match intuition.

### 1.2 oracles
- **A:** `pytest -q tests/test_oracle.py`
- **B (Qwen sanity):**
  ```python
  import os, numpy as np; from PIL import Image; from config import Config
  from verifier.queries import load_query_set; from verifier.oracle import QwenOracle
  from verifier.verify import extract_crop
  qs = load_query_set("queries/green_citrus.json"); cfg = Config()
  img = np.array(Image.open(os.environ["IMG"]).convert("RGB"))
  crop = extract_crop(img, [120, 340, 175, 400], cfg.crop_scale, cfg.crop_size)  # a fruit box
  print(list(zip([q.id for q in qs.queries], QwenOracle(cfg).answer_batch(crop, qs))))
  ```
  Answers visibly sane: round=+1, veins=−1.

### 1.3 training-free V-IP core
- **A:** `pytest -v tests/test_vip.py` — the four test names ARE the spec.

### 1.4 verify API + demo  ← GO/NO-GO
- **A:** `pytest -q tests/test_verify.py`
- **B:** pick 3 boxes (fruit / leaf / junk), then
  ```bash
  python scripts/demo_verify.py --image "$IMG" \
    --boxes B1 --boxes B2 --boxes B3 --oracle qwen
  ```
  The three query→answer chains should read like sensible reasoning. Tweak
  `queries/green_citrus.json` and re-run if not.
  Get candidate boxes to paste:
  ```bash
  python -c "import os,json; g=json.load(open(f'out/{os.environ[\"STEM\"]}_ioc_graph.json')); [print(','.join(str(int(v)) for v in n['box'])) for n in g['nodes'][:5]]"
  ```

### 1.5 pipeline integration (verifier flag)
- **A:** `pytest -q tests/test_pipeline_vip.py`
- **B:**
  ```bash
  python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 --verifier ioc
  python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 --verifier vip --oracle qwen
  diff "out/${STEM}_ioc_graph.json" "out/${STEM}_vip_graph.json" | head -40
  ```
  Spot-check 5 differing nodes with `demo_verify`.

---

## Phase 2 — belief state

### 2.1 support / jitter / signatures
- **A:** `pytest -q tests/test_graph_support.py`
- **B:** 3-pass run, then:
  ```python
  import os, json; g = json.load(open(f"out/{os.environ['STEM']}_ioc_graph.json"))
  for n in sorted(g["nodes"], key=lambda x: x["support"], reverse=True)[:10]:
      print(n["support"], round(n["jitter"], 1), n["area"], n["classification"])
  ```
  Stable fruit `support ≥ 2`; one-off junk `support == 1`.

### 2.2 belief.py (w, U, discovery, φ)
- **A:** `pytest -q tests/test_belief.py`
- **B:** print `belief.uncertainty(graph, discovery, cfg)` and
  `belief.summarize(...)` after each pass — U visibly drops across passes on an
  easy image.

---

## Phase 3 — actions & cost

### 3.1 action layer
- **A:** `pytest -q tests/test_actions.py`
- **B:** build an `ActionContext`, `execute(QueryA(region=<a quadrant>, ...), ctx)`,
  then `inference.plot_graph_scene(...)` — detections appear only inside that
  quadrant.

### 3.2 cost meter
- **A:** `pytest -q tests/test_budget.py`
- **B:** after a 2-pass episode: `print(ctx.cost.as_dict(), ctx.cost.total(cfg))`;
  recount by hand from the logs once.

---

## Phase 4 — heuristic controller

### 4.1 VoI heuristic + episode runner
- **A:** `pytest -q tests/test_policy_heuristic.py`
- **B:**
  ```python
  from agent.actions import ActionContext; from agent.belief import DiscoveryCurve
  from agent.policy_heuristic import choose; from agent import runner
  from graph import OrchardGraph; from config import Config
  # processor = inference.load_sam3_model(...); img = Image.open(os.environ["IMG"]).convert("RGB"); W,H = img.size
  ctx = ActionContext(processor=processor, image_pil=img, graph=OrchardGraph(), cfg=Config(),
                      discovery=DiscoveryCurve(), partition=[(0, 0, W, H)])
  out = runner.run_episode(img, ctx, choose, max_actions=12)
  for e in out["log"]: print(e)
  print(out["counts"], "cost", out["cost"])
  ```
  Run on sparse / dense / empty images: dense triggers `TileQueryA`/`SubdivideA`;
  empty stops within ~3 actions.

---

## Phase 5 — VLM orchestrator

### 5.1 scene inspection z_t
- **A:** `pytest -q tests/test_inspect.py`
- **B:** `inspect_scene(dense_img, phi, cfg)` vs `inspect_scene(sparse_img, phi, cfg)`
  — compare the two JSONs against your own eyes.

### 5.2 VLM policy with validation
- **A:** `pytest -q tests/test_policy_vlm.py`
- **B:** run a VLM episode via `runner.make_vlm_policy(ctx)` with logging on;
  read one prompt+response pair; grep the logs for `fallback` to confirm every
  executed action was validated.

---

## Phase 6 — evaluation

### 6.1 matching & metrics
- **A:** `pytest -q tests/test_matching.py`
- **B:** run `match()` on one image vs GT (draw matched green / missed red).

### 6.2 sweep runner
- **A:** `pytest -q tests/test_eval_sweep.py`
- **B:** `--limit 5` sweep of `{oneshot,cascade} × {ioc,vip}`; open the CSV —
  `pool_recall(cascade) ≥ pool_recall(oneshot)`, and the vip-vs-ioc precision
  delta tells you if the verifier earns its cost.

### 6.3 report + plot
- **A:** `pytest -q tests/test_report.py`
- **B:** `python -m eval.report --csv out/sweep.csv` — the accuracy-vs-cost plot
  is the paper figure; confirm the six-policy story is readable.

---

## Full runs

Whole CPU suite:
```bash
pytest -q            # expect 175 passed
```

Full six-policy sweep (needs the VLM server for vip / vlm):
```bash
for pol in oneshot cascade tiled convergence heuristic vlm; do for ver in ioc vip; do
  python -m eval.run_eval --root dataset --fmt yolo --split val --limit 5 \
    --policy $pol --verifier $ver --out out/sweep.csv
done; done
python -m eval.report --csv out/sweep.csv --out out/accuracy_vs_cost.png
```

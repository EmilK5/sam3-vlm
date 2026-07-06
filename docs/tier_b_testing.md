# Tier B Testing Guide — Ollama (Qwen VLM) + real SAM3

End-to-end manual testing on the GPU box. Tier B covers everything the CPU
pytest suite cannot: real SAM3 detections, real Qwen answers, real episodes.

The tests are numbered **T0–T12**. Run them in order the first time (later ones
reuse earlier outputs). Every test states **Run / Expect / Artifacts / If it
fails**. Nothing here modifies the repo; everything lands in `out/`.

The project talks to the VLM through `QWEN_BASE_URL` + `QWEN_MODEL` (an
OpenAI-compatible endpoint). This guide assumes **Ollama**; a vLLM appendix is
at the bottom for reference.

---

## How to report a failure

When any test fails, send back:

1. **The test ID and the exact command** you ran (copy the line).
2. **The full terminal output.** Every shell command below ends in
   `2>&1 | tee out/T<N>.log` where output matters — just attach that file.
   For Python-REPL tests, copy the whole traceback.
3. **The artifacts** listed under that test (files in `out/`).
4. This environment block:
   ```bash
   git rev-parse --short HEAD
   python --version
   pip show torch transformers openai 2>/dev/null | grep -E '^(Name|Version)'
   ollama --version && ollama list
   echo "QWEN_BASE_URL=$QWEN_BASE_URL  QWEN_MODEL=$QWEN_MODEL  IMG=$IMG"
   ```
5. One sentence on what you expected vs. what you saw (e.g. "T4: node count
   grew every pass under --verifier off; expected flat").

---

## Part 0 — One-time setup (Terminal B, project root)

```bash
pip install -r requirements.txt
mkdir -p out
```

Dataset layout expected by the commands below (adjust paths if yours differ):

```
dataset/
  images/val/*.png|jpg
  labels/val/*.txt        # YOLO: "cls xc yc w h" normalized
```

SAM3 (`facebook/sam3`) downloads from Hugging Face on first use — several
minutes is normal, as is a long `torch.compile` warm-up on the first pass.

---

## Part 1 — Start the Ollama VLM server (Terminal A)

1. Install Ollama from <https://ollama.com>, then pull a **vision-capable**
   Qwen model. Preferred (matches the paper's orchestrator):

   ```bash
   ollama pull qwen3-vl:8b
   ```

   If your Ollama doesn't have `qwen3-vl` yet, use `ollama pull qwen2.5vl:7b`
   (note: **no hyphen** — the library tag is `qwen2.5vl`). On a small GPU, the
   `:4b` / `:3b` variants also work; answers just get noisier.

2. **Raise the context window.** Ollama's default (~4k tokens) can truncate the
   VLM-policy prompt (φ + region table + menu), which shows up as garbage/cut
   JSON and constant fallbacks. Easiest fix — set it server-wide and start the
   server:

   ```bash
   OLLAMA_CONTEXT_LENGTH=8192 ollama serve
   ```

   (If `ollama serve` says the port is busy, Ollama already runs as a service.
   Either stop the service and rerun with the env var, or bake the context into
   a model instead:
   `printf 'FROM qwen3-vl:8b\nPARAMETER num_ctx 8192\n' > Modelfile && ollama create qwen3-vl-8k -f Modelfile`
   — then use `QWEN_MODEL=qwen3-vl-8k`.)

3. Leave this terminal running. `ollama ps` shows what's loaded.

---

## Part 2 — Point the project at the server (Terminal B)

```bash
export QWEN_BASE_URL=http://localhost:11434/v1
export QWEN_MODEL=qwen3-vl:8b      # MUST exactly match a NAME from `ollama list`
# no QWEN_API_KEY needed — the code sends a placeholder that Ollama ignores

export IMG=dataset/images/val/REPLACE_ME.png   # pick one real test image
export STEM=$(basename "$IMG" | sed 's/\.[^.]*$//')
```

`STEM` drives all output filenames; the run_image outputs are tagged by mode so
nothing overwrites: `out/${STEM}_<verifier>[_mask]_{overlay.jpg,graph.json}`.

---

## T0 — Preflight (no SAM3 yet)

**Run:**

```bash
pytest -q 2>&1 | tee out/T0_pytest.log                  # CPU suite
python -c "from config import Config; print(Config())"  # config sanity
curl -s "$QWEN_BASE_URL/models" | tee out/T0_models.log; echo
python - <<'EOF' 2>&1 | tee out/T0_vlm_smoke.log
# Minimal VQA round-trip: isolates server problems from project problems.
import base64, io, os
from openai import OpenAI
from PIL import Image
img = Image.new("RGB", (64, 64), (40, 160, 60))          # a plain green square
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
client = OpenAI(base_url=os.environ["QWEN_BASE_URL"], api_key="EMPTY")
r = client.chat.completions.create(model=os.environ["QWEN_MODEL"], temperature=0,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What color is this image? Reply with one word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
print(r.choices[0].message.content)
EOF
```

**Expect:**
- pytest prints **`204 passed`** (in ~1–2 s, no network).
- `print(Config())` shows one dataclass with all fields; `oracle_base_url` /
  `oracle_model_name` echo your env vars; `verifier_mode='ioc'`,
  `overlap_mode='box'`, `vip_epsilon=None`.
- `curl` returns JSON listing your model id.
- The VQA smoke prints something like `Green` / `green.`.

**If it fails:** the VQA smoke failing means server/model trouble, not project
trouble — send `out/T0_vlm_smoke.log` + `ollama list`. A wrong `QWEN_MODEL`
gives a 404 mentioning the model name.

---

## T1 — SAM3 cascade + GT overlay  (steps 0.2, 0.3)

**Run:**

```bash
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 3 --draw-gt \
  2>&1 | tee out/T1.log
python -c "
import os, json
g = json.load(open(f\"out/{os.environ['STEM']}_ioc_graph.json\"))
print('nodes:', len(g['nodes']))
print('by class:', {c: sum(1 for n in g['nodes'] if n['classification'] == c)
                    for c in set(n['classification'] for n in g['nodes'])})
"
```

**Expect:**
- Three `Pass N: {'raw_proposals': ..., 'post_nms': ..., 'duplicates_rejected': ...,
  'accepted': ...}` lines in the log; `accepted` shrinks across passes as dedup
  kicks in.
- Total node count == sum of the three `accepted` values.
- `out/${STEM}_ioc_overlay.jpg`: green solid boxes = fruit, red dashed = other;
  looks like the Gradio app output.
- `out/${STEM}_gt.jpg`: cyan GT boxes sit on real fruit.

**Artifacts:** `out/T1.log`, both jpgs, `out/${STEM}_ioc_graph.json`.

**If it fails:** first-run model download / compile can take minutes — only
report after it errors. `--draw-gt: no YOLO label found` means the label path
doesn't mirror `images/<split>/ -> labels/<split>/`.

---

## T2 — Qwen oracle answers one crop sanely  (step 1.2)

Grab candidate boxes to reuse in T2/T3 (prints `x1,y1,x2,y2` per node):

```bash
python -c "
import os, json
g = json.load(open(f\"out/{os.environ['STEM']}_ioc_graph.json\"))
for n in g['nodes'][:8]:
    print(','.join(str(int(v)) for v in n['box']), n['classification'])
"
```

**Run** (replace the box with a clear-fruit one from above):

```bash
python - <<'EOF' 2>&1 | tee out/T2.log
import os
import numpy as np
from PIL import Image
from config import Config
from verifier.queries import load_query_set
from verifier.oracle import QwenOracle
from verifier.verify import extract_crop
qs = load_query_set("queries/green_citrus.json")
img = np.array(Image.open(os.environ["IMG"]).convert("RGB"))
box = [16, 603, 32, 620]        # <-- REPLACE with a fruit box from the list
crop = extract_crop(img, box, Config().crop_scale, Config().crop_size)
crop.save("out/T2_crop.jpg")      # so you can see what the oracle saw
for q, a in zip(qs.queries, QwenOracle(Config()).answer_batch(crop, qs)):
    print(f"{q.id}  {a:+d}  {q.text}")
EOF
```

**Expect:** answers visibly sane for a fruit crop — round/spherical `+1`,
visible veins `-1`, thin/flat `-1`. A few `0`s are fine; **all zeros is a
failure** (it means every response failed to parse and the all-zeros fallback
fired — check `out/T2.log` for `QwenOracle: unparseable response` /
`request failed` warnings).

**Artifacts:** `out/T2.log`, `out/T2_crop.jpg`.

---

## T3 — GO/NO-GO: interpretable verifier chains  (step 1.4)

**Run** with three boxes from T2's list — ideally one clear fruit, one leaf,
one junk:

```bash
python scripts/demo_verify.py --image "$IMG" \
  --boxes FRUIT_BOX --boxes LEAF_BOX --boxes JUNK_BOX --oracle qwen \
  2>&1 | tee out/T3.log
```

**Expect:** per box, a chain like
`[Is the central object roun yes][...] -> target (0.94)` plus the three class
posteriors. Fruit → `target` with high mass; leaf → `distractor`; junk →
`spurious` or a hedged posterior. The chain should read like sensible
reasoning — this is where you judge the whole idea.

**If it fails:** verdicts are wrong but the *answers* in the chain look right →
edit templates/epsilon in `queries/green_citrus.json` and re-run (no code
change). Answers themselves are nonsense → report with `out/T3.log` +
`out/T2_crop.jpg`-style crops of the boxes you used.

---

## T4 — Verifier three-way: ioc vs vip vs off  (step 1.5 + off-dedup fix)

**Run:**

```bash
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 --verifier ioc 2>&1 | tee out/T4_ioc.log
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 --verifier vip --oracle qwen 2>&1 | tee out/T4_vip.log
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 3 --verifier off 2>&1 | tee out/T4_off.log
python -c "
import os, json
stem = os.environ['STEM']
for mode in ('ioc', 'vip', 'off'):
    g = json.load(open(f'out/{stem}_{mode}_graph.json'))
    sup = sorted((n['support'] for n in g['nodes']), reverse=True)
    print(mode, '->', len(g['nodes']), 'nodes; top support:', sup[:8])
"
```

**Expect:**
- ioc vs vip node counts are comparable; verdicts differ on some nodes. Spot
  check 3–5 differing nodes with `demo_verify` — the vip verdict should be the
  more defensible one.
- vip graph JSON nodes carry `vip_chain` + `vip_posterior`; ioc nodes don't.
- **off (regression check):** across passes 2–3 the `accepted` count in
  `out/T4_off.log` drops to ~0 and `duplicates_rejected` rises; the final node
  count is roughly the pass-1 count (NOT ~3× it), and re-detected nodes show
  `support >= 2`. If the node count grows linearly with passes, the
  unresolved-dedup fix regressed — report immediately.

**Artifacts:** the three logs, three graph JSONs, three overlays.

---

## T5 — Belief state: support / jitter / U drops  (steps 2.1, 2.2)

Uses the 3-pass ioc graph from T1.

**Run:**

```bash
python -c "
import os, json
g = json.load(open(f\"out/{os.environ['STEM']}_ioc_graph.json\"))
rows = sorted(g['nodes'], key=lambda n: n['support'], reverse=True)[:10]
print('support jitter   area   class     signatures')
for n in rows:
    print(f\"{n['support']:>7} {n['jitter']:>6.1f} {n['area']:>6.0f}   {n['classification']:<9} {len(n['signatures'])}\")
"
```

Then U across passes (one REPL, loads SAM3 once):

```bash
python - <<'EOF' 2>&1 | tee out/T5.log
import os
import torch
from PIL import Image
import inference, pipeline
from config import Config
from graph import OrchardGraph
from agent import belief
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, processor = inference.load_sam3_model(f"{cfg.sam3_repo}/assets/bpe_simple_vocab_16e6.txt.gz", cfg.conf, device=device)
img = Image.open(os.environ["IMG"]).convert("RGB")
g, disc = OrchardGraph(), belief.DiscoveryCurve()
for p in (1, 2, 3):
    n = pipeline.execute_pass(processor, img, g, cfg.conf, False, False, p, "green fruit", cfg=cfg)
    disc.append(int(n))
    print(f"pass {p}: new={int(n)}  U={belief.uncertainty(g, disc, cfg):.2f}")
EOF
```

**Expect:** stable fruit has `support >= 2` with small jitter; one-off junk has
`support == 1`. `U` visibly drops from pass 1 to pass 3 on an easy image
(mostly because discovery flattens; the per-node sum grows slightly with K —
a modest late rise is fine, a monotone climb is not).

**Artifacts:** the support table (copy it), `out/T5.log`.

---

## T6 — Action layer + cost meter  (steps 3.1, 3.2)

**Run:**

```bash
python - <<'EOF' 2>&1 | tee out/T6.log
import os
import torch
from PIL import Image
import inference
from config import Config
from graph import OrchardGraph
from agent.actions import ActionContext, QueryA, execute
from agent.belief import DiscoveryCurve
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, processor = inference.load_sam3_model(f"{cfg.sam3_repo}/assets/bpe_simple_vocab_16e6.txt.gz", cfg.conf, device=device)
img = Image.open(os.environ["IMG"]).convert("RGB"); W, H = img.size
ctx = ActionContext(processor=processor, image_pil=img, graph=OrchardGraph(), cfg=cfg,
                    discovery=DiscoveryCurve(), partition=[(0, 0, W, H)])
n = execute(QueryA(region=(0, 0, W // 2, H // 2), prompt="green fruit", conf=cfg.conf), ctx)
print("new candidates:", n)
print("cost:", ctx.cost.as_dict(), " total:", ctx.cost.total(cfg))
inference.plot_graph_scene(img, ctx.graph, output_path="out/T6_quadrant.jpg")
EOF
```

**Expect:**
- `out/T6_quadrant.jpg`: every box strictly inside the **top-left quadrant**.
- Cost after this single region query: `n_sam == 2` (leaf map + the proposal
  call — a region query skips canopy detection), `n_tile == 0`, `n_orch == 1`,
  so `total == 2.05` with default weights. These are the post-fix semantics —
  the meter counts *actual* SAM3 calls now.
- Hand-check: the log shows exactly two `inference` sweeps ("Generating global
  leaf map", "Global inference started").

**Artifacts:** `out/T6.log`, `out/T6_quadrant.jpg`.

---

## T7 — Heuristic episode  (step 4.1)

**Run** on three images — sparse, dense, and empty-of-fruit:

```bash
python -m eval.run_eval --root dataset --fmt yolo --split val --limit 3 \
  --policy heuristic --verifier ioc --prompt "green fruit" \
  --out out/T7.csv 2>&1 | tee out/T7.log
grep -o '"action": "[A-Za-z]*"' out/T7.log | sort | uniq -c
```

**Expect:**
- Completes with per-step JSON log lines `{"t": 1, "action": "...", "n_new": ...,
  "U": ..., "cost_so_far": ...}` and one CSV row per image. **This exact
  configuration used to crash with `AttributeError: 'NoneType' ... 'classes'`
  — any traceback here is a regression, report it.**
- The action histogram contains **no `VerifyA`** (verify needs `--verifier vip`).
- Dense image → `TileQueryA`/`SubdivideA` appear; empty image → `StopA` within
  ~3 actions; `n_actions` in the CSV ≤ 12 (the budget).
- Optional vip variant (VerifyA now allowed):
  `python -m eval.run_eval --root dataset --fmt yolo --split val --limit 1 --policy heuristic --verifier vip --out out/T7vip.csv 2>&1 | tee out/T7vip.log`

**Artifacts:** `out/T7.log`, `out/T7.csv` (+ vip variants).

---

## T8 — Scene inspection z_t  (step 5.1)

**Run** (once with a dense-canopy image path, once with a sparse one):

```bash
python - <<'EOF' 2>&1 | tee out/T8.log
import os
from PIL import Image
from config import Config
from agent.inspect import inspect_scene
img = Image.open(os.environ["IMG"]).convert("RGB")
print(inspect_scene(img, {"K": 0, "n_t": 0, "tiling_status": False, "remaining_budget": 12}, Config()))
EOF
```

**Expect:** a dict with the exact keys `target_present / density / object_scale /
occlusion / recommend / notes`, values matching your own read of the image.
`notes == 'fallback'` + `recommend == 'query'` means every attempt failed to
parse (or the request errored) — that's the safe fallback, but for a healthy
server it should be rare; report if it's persistent.

---

## T9 — VLM policy episode with validation  (step 5.2)

**Run:**

```bash
python -m eval.run_eval --root dataset --fmt yolo --split val --limit 1 \
  --policy vlm --verifier vip --prompt "green fruit" \
  --out out/T9.csv 2>&1 | tee out/T9.log
grep -c "policy_vlm prompt"   out/T9.log     # prompts sent
grep -c "policy_vlm response" out/T9.log     # responses received
grep -c "fallback"            out/T9.log     # heuristic fallbacks
```

**Expect:**
- At least one full prompt+response pair in the log — read one: the prompt
  contains compact φ, z, the region table, and the action menu; the response is
  bare JSON `{"action": ..., "args": ...}`.
- Every executed action is either a validated VLM choice or an explicitly
  logged fallback — there is no third path. A few fallbacks are the safety net
  working; **near-100% fallback** means the model can't hold the JSON contract
  (try a bigger tag or raise the context length, see Part 1).
- With `--verifier vip`, the menu includes `verify`; re-run with
  `--verifier ioc` and confirm the logged prompts contain **no** `"verify"` in
  `action_menu` and the run still completes (this used to be crashable).

**Artifacts:** `out/T9.log`, `out/T9.csv`.

---

## T10 — Mask-based IoU/IoM switch  (new feature)

Box overlap stays the default; this verifies the opt-in mask mode end to end.
Best on an image with a **tight cluster of touching fruit**.

**Run:**

```bash
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 \
  --verifier ioc --overlap-mode mask 2>&1 | tee out/T10_mask.log
python -c "
import os, json
stem = os.environ['STEM']
for tag in ('ioc', 'ioc_mask'):
    g = json.load(open(f'out/{stem}_{tag}_graph.json'))
    print(tag, '->', len(g['nodes']), 'nodes')
"
# tiled passes must fall back to box overlap with a warning, not crash:
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 1 \
  --tiling --overlap-mode mask 2>&1 | tee out/T10_tiled.log
grep "not supported with tiling" out/T10_tiled.log
```

**Expect:**
- Outputs are separately tagged: `out/${STEM}_ioc_mask_overlay.jpg` /
  `_graph.json` alongside the box-mode `_ioc_` files from T4.
- Compare the two overlays around fruit clusters: mask mode should **keep
  touching-but-distinct fruits** whose boxes overlap heavily but whose masks
  don't (box mode suppresses one of them). Node counts typically equal or
  slightly higher in mask mode; verdicts for unchanged nodes should match.
- Dedup still works across passes in mask mode (pass-2 `duplicates_rejected`
  > 0 in `out/T10_mask.log`).
- The tiled run prints the `overlap_mode='mask' is not supported with tiling`
  warning and completes normally in box mode.

**Artifacts:** both T10 logs, both overlays, both graph JSONs. When reporting a
suppression you disagree with, include a crop/screenshot of the cluster.

---

## T11 — Full sweep + paper figure  (steps 6.2, 6.3)

**Run** (the sweep is resume-safe — rerunning skips finished rows):

```bash
for pol in oneshot cascade tiled convergence heuristic vlm; do for ver in ioc vip; do
  python -m eval.run_eval --root dataset --fmt yolo --split val --limit 5 \
    --policy $pol --verifier $ver --prompt "green fruit" --out out/sweep.csv \
    2>&1 | tee -a out/T11.log
done; done
# optional: a mask-mode line for the figure (rows tagged verifier="ioc+mask")
python -m eval.run_eval --root dataset --fmt yolo --split val --limit 5 \
  --policy cascade --verifier ioc --overlap-mode mask --out out/sweep.csv 2>&1 | tee -a out/T11.log

python -m eval.report --csv out/sweep.csv --out out/accuracy_vs_cost.png 2>&1 | tee out/T11_report.log
```

**Expect:**
- One CSV row per image×policy×verifier with all `CSV_FIELDS`;
  aggregate blocks printed after each combo.
- Sanity relations: `pool_recall(cascade) >= pool_recall(oneshot)` per image;
  `convergence` uses `n_actions <= 8`; heuristic/vlm respect the 12-action
  budget; the vip-vs-ioc precision delta tells you if the verifier earns its
  cost.
- **Costs are on one unified scale now** (canopy + leaf map + proposals + real
  verify calls, for fixed policies *and* episodes) — so absolute numbers are
  higher than pre-fix runs; only cross-policy comparisons matter. A oneshot
  ioc image costs ~3.0 (canopy + leaf map + 1 proposal), not 1.0.
- `out/accuracy_vs_cost.png`: MAE (y) vs cost (x); the six-policy story
  (one-shot → cascade → tiled → convergence → heuristic → VLM) reads cleanly.

**Artifacts:** `out/sweep.csv`, `out/accuracy_vs_cost.png`, both T11 logs.

---

## T12 — Regression pack (fixes from the 2026-07-05 review)

Quick checks that each fixed bug stays fixed on real runs:

```bash
# (a) agent policies without the vip oracle must not crash (was: AttributeError)
python -m eval.run_eval --root dataset --fmt yolo --split val --limit 1 \
  --policy heuristic --verifier off --out out/T12a.csv 2>&1 | tee out/T12a.log

# (b) convergence under --verifier off must terminate early (was: always 8 passes)
python -m eval.run_eval --root dataset --fmt yolo --split val --limit 1 \
  --policy convergence --verifier off --out out/T12b.csv 2>&1 | tee out/T12b.log
python -c "import csv; r = list(csv.DictReader(open('out/T12b.csv')))[-1]; print('n_actions:', r['n_actions'], ' N_obs:', r['N_obs'], ' N_gt:', r['N_gt'])"

# (c) --prompt must reach agent episodes (was: silently 'green fruit')
python - <<'EOF' 2>&1 | tee out/T12c.log
import dataclasses, os
import torch
from PIL import Image
import inference, pipeline
from config import Config
from graph import OrchardGraph
from agent.actions import ActionContext
from agent.belief import DiscoveryCurve
from agent.policy_heuristic import choose
from agent import runner
import numpy as np
cfg = dataclasses.replace(Config(), target_prompt="apple")   # <- custom concept
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, processor = inference.load_sam3_model(f"{cfg.sam3_repo}/assets/bpe_simple_vocab_16e6.txt.gz", cfg.conf, device=device)
img = Image.open(os.environ["IMG"]).convert("RGB")
g = OrchardGraph()
roi = pipeline.initialize_canopy_roi(processor, np.array(img), g)
ctx = ActionContext(processor=processor, image_pil=img, graph=g, cfg=cfg,
                    discovery=DiscoveryCurve(), partition=[tuple(roi)])
out = runner.run_episode(img, ctx, choose, max_actions=4)
prompts = {s.split(":")[2] for n in g.nodes.values() for s in n.signatures}
print("prompts used by the episode:", prompts)   # expect {'apple'}
EOF
```

**Expect:**
- (a) completes with CSV rows, zero tracebacks, and no `VerifyA` in the log.
- (b) `n_actions` **< 8** (typically 2–3: the pass after dedup saturates
  discovers 0), and `N_obs` in the same ballpark as `N_gt` — not a multiple
  of it.
- (c) prints `prompts used by the episode: {'apple'}` — the custom concept
  reached SAM3 (detections may be poor if the image has no apples; only the
  prompt set matters here).

**Artifacts:** the three T12 logs + `out/T12b.csv`.

---

## Stopping

- Terminal B: nothing to stop.
- Ollama: runs as a service; `ollama stop $QWEN_MODEL` unloads the model from
  VRAM (or just leave it — it idles out).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `curl "$QWEN_BASE_URL/models"` fails | Server not up / wrong port. Ollama listens on **11434**; `QWEN_BASE_URL` must end in `/v1`. |
| 404 "model not found" on any VLM call | `QWEN_MODEL` must exactly match a NAME in `ollama list` (tag included, e.g. `qwen3-vl:8b`). |
| Every oracle answer is `0` (T2) | Model returned unparseable JSON every attempt. Check `out/T2.log` for `unparseable response` warnings; try a larger tag. `request failed` warnings instead → server/network issue (the run continues on the all-zeros fallback by design). |
| Answers ignore the image | The tag isn't vision-capable. Use `qwen3-vl:*` / `qwen2.5vl:*`. |
| VLM policy: constant `fallback` (T9) | JSON contract too hard for the model or prompt truncated. Raise context (Part 1 step 2) and/or use a bigger tag. Occasional fallbacks are fine — that's the validation layer doing its job. |
| GPU OOM on the Ollama side | Use a smaller tag (`:4b`, `:3b`); don't run SAM3 and a large VLM on the same small GPU. |
| SAM3 first pass extremely slow | Expected: weight download + `torch.compile` warm-up. Subsequent passes are fast. |
| `No module named 'config'` from scripts | Run from the project root, or use the module form (`python -m eval.run_eval ...`). |
| Sweep reruns skip everything | That's resume: rows keyed by (image, policy, verifier[+mask]) already exist in the CSV. Use a new `--out` file to force a fresh run. |

---

## Appendix — vLLM instead of Ollama

Everything above works identically with vLLM; only the server start and two env
vars differ:

```bash
python -m venv ~/vllm-env && source ~/vllm-env/bin/activate   # separate venv (own torch)
pip install --upgrade vllm                                    # needs a recent release for Qwen3-VL
vllm serve Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl --port 8000
# then:
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=qwen3-vl
```

OOM → `--max-model-len 8192` or a smaller `-4B`/`-2B` checkpoint; multi-GPU →
`--tensor-parallel-size N`. Stop with Ctrl-C.

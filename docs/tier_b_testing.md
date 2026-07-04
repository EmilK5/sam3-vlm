# Tier B Testing Guide — vLLM + Qwen + real SAM3

This walks you, end to end, through:
1. Starting a **local Qwen vision server** with vLLM (no API key).
2. Running **all the Tier B tests** (the real-model checks for Phases 0 and 1).

Everything is copy-paste. Placeholders are only `IMG` (your test image) — every
other value is derived automatically.

---

## Before you start

You need a machine with an **NVIDIA GPU**. You'll use **two terminals**:
- **Terminal A** — runs the Qwen vLLM server (stays open the whole time).
- **Terminal B** — runs the project tests.

In **Terminal B**, from the project root, make sure the project deps are installed:

```bash
pip install -r requirements.txt
```

SAM3 loads from Hugging Face (`facebook/sam3`) the first time you run it — that
download can take a few minutes. That's normal.

---

## Part 1 — Start the Qwen vision server (Terminal A)

### 1.1 Install vLLM (in its OWN environment)

vLLM ships its own copy of torch, so install it in a **separate** virtualenv to
avoid disturbing the project's environment:

```bash
python -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade vllm
```

> Use a **recent** vLLM — Qwen3-VL support landed in late-2025 releases. If the
> serve command later complains about an unknown architecture, upgrade vLLM.

### 1.2 Launch the server

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl \
  --port 8000
```

- The **first launch downloads the model** (several GB). Wait until you see a
  line like `Uvicorn running on http://0.0.0.0:8000`.
- `--served-model-name qwen3-vl` is what makes `QWEN_MODEL=qwen3-vl` work below,
  regardless of which Qwen model you actually load.

**If you run out of GPU memory**, either add `--max-model-len 8192`, or pick a
smaller model — swap the first argument for one of:
`Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-2B-Instruct`.
**Multiple GPUs?** add `--tensor-parallel-size N`.

Leave this terminal running.

---

## Part 2 — Point the project at the server (Terminal B)

```bash
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=qwen3-vl
# No API key needed for a local vLLM server.
```

Confirm the server is reachable — this should list `qwen3-vl`:

```bash
curl http://localhost:8000/v1/models
```

---

## Part 3 — Run the Tier B tests (Terminal B)

Pick one test image from your dataset and set it once. Everything else derives
the output filenames automatically:

```bash
export IMG=dataset/images/val/REPLACE_ME.png
export STEM=$(basename "$IMG" | sed 's/\.[^.]*$//')
```

### Test 1 — SAM3 cascade + GT overlay  (steps 0.2, 0.3)

```bash
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 3 --draw-gt
```

**Check** in `out/`:
- `${STEM}_ioc_overlay.jpg` — detections; should resemble the Gradio app.
- `${STEM}_gt.jpg` — ground-truth boxes (cyan) sit on real fruit.
- `${STEM}_ioc_graph.json` — node count matches the logged pass stats.

### Test 2 — the GO / NO-GO: interpretable verifier chains  (step 1.4)

First, grab a few candidate boxes from the graph you just produced (prints
ready-to-paste `x1,y1,x2,y2` strings for the first 5 nodes):

```bash
python -c "
import os, json
stem = os.environ['STEM']
g = json.load(open(f'out/{stem}_ioc_graph.json'))
for n in g['nodes'][:5]:
    print(','.join(str(int(v)) for v in n['box']))
"
```

Now verify 3 boxes (ideally one clear fruit, one leaf, one junk) with the real
Qwen oracle:

```bash
python scripts/demo_verify.py --image "$IMG" \
  --boxes BOX1 --boxes BOX2 --boxes BOX3 --oracle qwen
```

**Check:** each printed chain reads like sensible reasoning, e.g.
`[round? yes][waxy? yes][veins? no] -> target (0.94)`. This is where you decide
whether FM+V-IP actually works. If a chain misfires, edit templates/epsilon in
`queries/green_citrus.json` and re-run — no code change needed.

### Test 3 — Qwen oracle answers a single crop sanely  (step 1.2)

Uses the first box from Test 2's list; answers should be visibly sane
(round = yes/+1, veins = no/-1):

```bash
python -c "
import os; from config import Config
from verifier.queries import load_query_set
from verifier.oracle import QwenOracle
from verifier.verify import extract_crop
import numpy as np; from PIL import Image
qs = load_query_set('queries/green_citrus.json')
img = np.array(Image.open(os.environ['IMG']).convert('RGB'))
box = [BOX1_COORDS]   # e.g. [120, 340, 175, 400]
crop = extract_crop(img, box, Config().crop_scale, Config().crop_size)
ans = QwenOracle(Config()).answer_batch(crop, qs)
print(list(zip([q.id for q in qs.queries], [q.text for q in qs.queries], ans)))
"
```

### Test 4 — ioc vs vip comparison  (step 1.5)

Runs the same image under both verifiers (output files are mode-tagged so they
don't overwrite each other) and diffs the graphs:

```bash
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 --verifier ioc
python scripts/run_image.py --image "$IMG" --prompt "green fruit" --passes 2 --verifier vip --oracle qwen
diff "out/${STEM}_ioc_graph.json" "out/${STEM}_vip_graph.json" | head -40
```

**Check:** open both overlays (`out/${STEM}_ioc_overlay.jpg` vs
`out/${STEM}_vip_overlay.jpg`). Spot-check 5 nodes whose verdicts differ by
re-running `demo_verify` on their boxes — the vip verdict should be the more
defensible one.

---

## Part 4 — Stopping

- In **Terminal B**, nothing to stop.
- In **Terminal A**, press **Ctrl-C** to shut down the vLLM server.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `curl .../v1/models` fails | Server still loading, or wrong port. Wait for the `Uvicorn running` line; make sure `QWEN_BASE_URL` ends in `/v1`. |
| Every oracle answer is `0` / `unsure` | The model isn't returning valid JSON. Confirm `QWEN_MODEL` matches `--served-model-name`; try a larger Qwen3-VL model. |
| vLLM: "unknown model architecture" | Upgrade vLLM: `pip install --upgrade vllm`. |
| GPU out of memory on the server | Add `--max-model-len 8192`, or use `Qwen/Qwen3-VL-4B-Instruct` / `-2B-Instruct`. |
| SAM3 first run is slow | Expected — weight download + `torch.compile` warm-up. |
| `run_image.py` can't find the image | Use a path relative to where you run the command, e.g. `dataset/images/val/xyz.png`. |

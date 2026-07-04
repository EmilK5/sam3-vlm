# Tier B Testing Guide — Qwen VLM server (vLLM or Ollama) + real SAM3

This walks you, end to end, through:
1. Starting a **local Qwen vision server** — pick **Option A (vLLM)** or
   **Option B (Ollama)**. Both expose an OpenAI-compatible endpoint, so the
   project code is identical; only two env vars differ. No API key either way.
2. Running **all the Tier B tests** (the real-model checks for Phases 0 and 1).

The project talks to whichever server through `QWEN_BASE_URL` + `QWEN_MODEL`;
nothing in the code is vLLM- or Ollama-specific.

Everything is copy-paste. Placeholders are only `IMG` (your test image) — every
other value is derived automatically.

---

## Before you start

SAM3 itself needs a machine that can load `facebook/sam3` (a CUDA GPU in
practice). For the **VLM server**: vLLM needs an NVIDIA GPU; Ollama also runs on
Apple Silicon / AMD / CPU (vision models are slow on CPU but fine for a few test
images). You'll use **two terminals**:
- **Terminal A** — runs the Qwen VLM server (stays open the whole time).
- **Terminal B** — runs the project tests.

In **Terminal B**, from the project root, make sure the project deps are installed:

```bash
pip install -r requirements.txt
```

SAM3 loads from Hugging Face (`facebook/sam3`) the first time you run it — that
download can take a few minutes. That's normal.

---

## Part 1 — Start the Qwen vision server (Terminal A)

Do **either** Option A or Option B, then continue to Part 2.

### Option A — vLLM (NVIDIA GPU)

vLLM ships its own copy of torch, so install it in a **separate** virtualenv to
avoid disturbing the project's environment:

```bash
python -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade vllm
```

> Use a **recent** vLLM — Qwen3-VL support landed in late-2025 releases. If the
> serve command later complains about an unknown architecture, upgrade vLLM.

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl \
  --port 8000
```

- First launch **downloads the model** (several GB). Wait for `Uvicorn running on
  http://0.0.0.0:8000`.
- `--served-model-name qwen3-vl` is what makes `QWEN_MODEL=qwen3-vl` work below.
- OOM? add `--max-model-len 8192` or use `Qwen/Qwen3-VL-4B-Instruct` /
  `-2B-Instruct`. Multi-GPU: `--tensor-parallel-size N`.

For vLLM, in Part 2 use `QWEN_BASE_URL=http://localhost:8000/v1` and
`QWEN_MODEL=qwen3-vl`.

### Option B — Ollama (NVIDIA / AMD / Apple Silicon / CPU)

```bash
# install Ollama (https://ollama.com), then:
ollama pull qwen2.5-vl:7b        # a VISION model (needed for the oracle + inspect crops)
ollama serve                     # usually already running as a service
```

- The **model tag** you pull (e.g. `qwen2.5-vl:7b`) is exactly what `QWEN_MODEL`
  must be — check with `ollama list`. Use whatever Qwen-VL tag your Ollama has;
  it **must be vision-capable**.
- Ollama serves the OpenAI-compatible API at **port 11434** by default.
- **Context length:** Ollama models default to a small `num_ctx`. The VLM-policy
  prompt (φ) can be long; if you see truncated/garbage JSON, raise it via a
  Modelfile:
  ```bash
  printf 'FROM qwen2.5-vl:7b\nPARAMETER num_ctx 8192\n' > Modelfile
  ollama create qwen2.5-vl-8k -f Modelfile
  # then use QWEN_MODEL=qwen2.5-vl-8k
  ```

For Ollama, in Part 2 use `QWEN_BASE_URL=http://localhost:11434/v1` and
`QWEN_MODEL=<your pulled tag>`.

Leave this terminal running.

---

## Part 2 — Point the project at the server (Terminal B)

Use the pair matching the option you started (no API key needed either way):

```bash
# Option A — vLLM
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=qwen3-vl

# Option B — Ollama
export QWEN_BASE_URL=http://localhost:11434/v1
export QWEN_MODEL=qwen2.5-vl:7b        # must match `ollama list`
```

Confirm the server is reachable — this should list your model:

```bash
curl "$QWEN_BASE_URL/models"
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
- **vLLM** (Terminal A): press **Ctrl-C**.
- **Ollama**: it runs as a background service; `ollama stop <model>` unloads the
  model from memory (or leave it — it idles out).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `curl "$QWEN_BASE_URL/models"` fails | Server still loading, or wrong port/URL. vLLM: wait for `Uvicorn running`; Ollama: `ollama serve` running on 11434. Make sure `QWEN_BASE_URL` ends in `/v1`. |
| Every oracle answer is `0` / `unsure` | The model isn't returning valid JSON. Confirm `QWEN_MODEL` matches the served name (`ollama list` for Ollama); try a larger Qwen-VL model. |
| Answers ignore the image / describe nothing | The model isn't vision-capable, or the endpoint dropped the image. Use a `*-vl` / `-vision` tag; on Ollama confirm the tag supports images. |
| VLM policy JSON truncated / cut off | Prompt exceeds the model context. Ollama: raise `num_ctx` via a Modelfile (see Option B). vLLM: raise `--max-model-len`. |
| vLLM: "unknown model architecture" | Upgrade vLLM: `pip install --upgrade vllm`. |
| Ollama: `model not found` | Pull it first (`ollama pull <tag>`) and set `QWEN_MODEL` to the exact `ollama list` tag. |
| GPU out of memory on the server | vLLM: `--max-model-len 8192` or a smaller model. Ollama: use a smaller tag (e.g. `:3b`). |
| SAM3 first run is slow | Expected — weight download + `torch.compile` warm-up. |
| `run_image.py` can't find the image | Use a path relative to where you run the command, e.g. `dataset/images/val/xyz.png`. |

"""
citrus_orchestration_app.py

Interactive Gradio sandbox that runs the FULL active-perception orchestration
pipeline of this repo (agent episode: policy -> action -> belief update) on the
local green-fruit/citrus dataset, with a verbose log window exposing every
action, per-pass SAM3 diagnostic, verifier call, and count estimate. Built to
watch the VLM (Qwen-3-VL) orchestration policy decide -- prompts, responses,
and fallbacks are all in the verbose log.

This is a standalone driver, structured like orchestration_app.py (does NOT
touch app.py, pipeline.py, or any validated core module). It only *uses* their
public entry points:

    agent.runner.run_episode                  - the orchestration loop
    agent.policy_heuristic / policy_vlm        - the two policies
    agent.actions.ActionContext                - the episode sensing state
    agent.belief.count_estimates                - N_obs / N_supp / N_cons
    pipeline.initialize_canopy_roi             - canopy ROI for the initial partition
    eval.datasets.load_split                   - YOLO-format ground truth loader
    verifier.queries.load_query_set            - FM+V-IP query set (vip mode)
    verifier.oracle.MockOracle/QwenOracle
    inference.load_sam3_model

Dataset: a local YOLO-format green-fruit/citrus split
    root/images/<split>/*.jpg
    root/labels/<split>/*.txt   ("cls xc yc w h", normalized)
Root defaults to "datasets/citruses_dataset" (override with env var
CITRUS_DATASET_ROOT); split falls back to a flat root/images layout when no
train/val/test subdirectory exists (see eval.datasets._split_dir).

View: left panel is the SAM3 sensor view (canopy ROI box + every candidate's
box, with a translucent mask fill when overlap_mode="mask" produced one, color
coded by verifier classification); right panel is the ground-truth overlay for
the same image. Run this on the GPU box -- SAM3 (and, for the vlm policy, a
live Qwen-3-VL endpoint) must be reachable.
"""

import dataclasses
import json
import logging
import os
import sys

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 0. Paths (SAM3 repo/BPE come from config.py, per CLAUDE.md's step 0.1 move)
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

_PATH_CFG = Config()
SAM3_REPO_ROOT = _PATH_CFG.sam3_repo
BPE_PATH = f"{SAM3_REPO_ROOT}/assets/bpe_simple_vocab_16e6.txt.gz"

if SAM3_REPO_ROOT not in sys.path:
    sys.path.append(SAM3_REPO_ROOT)

# Local citrus/green-fruit dataset root (YOLO format: images/<split>, labels/<split>).
DATASET_ROOT = os.environ.get("CITRUS_DATASET_ROOT", "datasets/citruses_dataset")
DEFAULT_PROMPT = "green fruit"

from graph import OrchardGraph
from agent.actions import ActionContext
from agent.belief import DiscoveryCurve
from agent import runner, policy_heuristic
from eval.datasets import load_split
import inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# Load SAM3 once. If weights are unavailable, keep the UI alive and surface the
# failure at run time instead of crashing on import (mirrors orchestration_app.py).
try:
    MODEL, PROCESSOR = inference.load_sam3_model(BPE_PATH, 0.35, device=device)
    _LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - depends on machine weights
    MODEL, PROCESSOR = None, None
    _LOAD_ERROR = str(exc)
    logger.warning("SAM3 model failed to load: %s", exc)


# ==========================================
# 1. Dataset access layer
# ==========================================
_dataset_cache = {}


def get_samples(split):
    """YOLO-format samples for one split: [{"image_path", "gt_boxes", "count"}, ...]."""
    if split not in _dataset_cache:
        try:
            _dataset_cache[split] = load_split(DATASET_ROOT, "yolo", split)
        except FileNotFoundError as exc:
            logger.error("Failed to load citrus dataset split '%s': %s", split, exc)
            _dataset_cache[split] = []
    return _dataset_cache[split]


# ==========================================
# 2. Rendering
# ==========================================
_CLASS_COLORS = {
    "fruit": (57, 255, 20),        # verified target
    "leaf": (255, 59, 59),         # verified distractor
    "spurious": (154, 160, 166),   # rejected
    "unresolved": (0, 229, 255),   # unverified candidate (still counted toward N_obs)
}
_ROI_COLOR = (255, 215, 0)
_GT_COLOR = (0, 229, 255)


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_sam3_view(image_pil, graph, prompt_str):
    """Left panel: canopy ROI box + every candidate's box, colored by
    classification, with a translucent mask fill wherever the node carries one
    (overlap_mode='mask', non-tiled passes); boxless candidates just get an
    outline."""
    canvas = image_pil.convert("RGBA").copy()
    font = _font()

    # Mask fills first, so box outlines/text drawn afterward stay crisp.
    mask_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for node in graph.nodes.values():
        if node.mask is None:
            continue
        rgb = _CLASS_COLORS.get(node.classification, (255, 255, 255))
        x1, y1 = int(round(node.box[0])), int(round(node.box[1]))
        alpha = (node.mask.astype("uint8") * 140)
        alpha_img = Image.fromarray(alpha, mode="L")
        color_layer = Image.new("RGBA", alpha_img.size, rgb + (0,))
        color_layer.putalpha(alpha_img)
        mask_overlay.paste(color_layer, (x1, y1), color_layer)
    canvas = Image.alpha_composite(canvas, mask_overlay).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    roi = getattr(graph, "tree_roi", None)
    if roi is not None:
        draw.rectangle([int(v) for v in roi], outline=_ROI_COLOR, width=4)
        draw.text((roi[0] + 4, max(0, roi[1] - 16)), "canopy ROI", fill=_ROI_COLOR, font=font)

    tally = {}
    for node in graph.nodes.values():
        x1, y1, x2, y2 = node.box
        rgb = _CLASS_COLORS.get(node.classification, (255, 255, 255))
        tally[node.classification] = tally.get(node.classification, 0) + 1
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
        tag = f"P{node.found_in_pass} {node.classification[:1]}:{node.scores['detection_confidence']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=rgb, font=font)

    banner = f"SAM3 sensor view | prompt='{prompt_str}' | {tally or 'no candidates yet'}"
    draw.text((8, 8), banner, fill=(255, 255, 255), font=font)
    return canvas


def draw_gt_view(image_pil, gt_boxes, gt_count, pred_count):
    """Right panel: ground-truth overlay, with a match/mismatch banner once a
    prediction is available."""
    canvas = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for box in gt_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=_GT_COLOR, width=3)

    banner = f"GROUND TRUTH | count={gt_count}"
    color = (255, 255, 255)
    if pred_count is not None:
        match = pred_count == gt_count
        banner += f" | SAM3 N_obs={pred_count} | {'MATCH' if match else 'MISMATCH'}"
        color = (46, 204, 113) if match else (231, 76, 60)
    draw.text((8, 8), banner, fill=color, font=font)
    return canvas


# ==========================================
# 3. Offline mock client (sandbox testing without a live Qwen endpoint)
# ==========================================
_MOCK_INSPECT_Z = {
    "target_present": True, "density": "medium", "object_scale": "medium",
    "occlusion": "medium", "recommend": "tile", "notes": "mock (offline stub)",
}


class _Resp:
    """Minimal stand-in for an OpenAI chat completion response."""
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _StubChatClient:
    """Deterministic offline stand-in for the Qwen endpoint.

    Distinguishes an inspection call (returns a neutral z) from an orchestration
    call (returns a short tile-then-stop plan), so both the inspect and VLM-policy
    parse/validation paths are exercised without any network. Used only when the
    Mock toggle is on and policy='vlm'.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self._orch_calls = 0
        self.chat = self
        self.completions = self

    def create(self, model=None, temperature=None, messages=None, **kwargs):
        system = (messages[0]["content"] if messages else "").lower()
        if "assess" in system:  # agent.inspect.SYSTEM_PROMPT ("You assess images...")
            return _Resp(json.dumps(_MOCK_INSPECT_Z))
        # agent.policy_vlm.SYSTEM_PROMPT ("You are the orchestrator...")
        self._orch_calls += 1
        conf = float(min(0.9, max(0.1, self.cfg.conf)))
        if self._orch_calls >= 2:
            action = {"action": "stop", "args": {"estimate_name": "N_obs"}}
        else:
            action = {"action": "tile", "args": {"conf": round(conf, 2)}}
        return _Resp(json.dumps(action))


# ==========================================
# 4. Orchestration driver
# ==========================================
class _ListLogHandler(logging.Handler):
    """Captures formatted log records emitted during one episode run."""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


def _build_cfg(prompt, verifier, overlap_mode, gate_mode, iou_threshold, iom_threshold,
               conf, budget, query_file, enable_thinking):
    return dataclasses.replace(
        Config(),
        verifier_mode=verifier,
        overlap_mode=overlap_mode,
        gate_mode=gate_mode,
        nms_iou_threshold=float(iou_threshold),
        nms_iom_threshold=float(iom_threshold),
        conf=float(conf),
        target_prompt=prompt,
        budget_max_actions=int(budget),
        vip_query_file=query_file or Config().vip_query_file,
        policy_enable_thinking=bool(enable_thinking),
    )


def _build_oracle(cfg, verifier, use_mock, mock_true_class):
    """Return (oracle, query_set) for the run. Both None unless verifier='vip'."""
    if verifier != "vip":
        return None, None
    from verifier.queries import load_query_set
    query_set = load_query_set(cfg.vip_query_file)
    if use_mock:
        from verifier.oracle import MockOracle
        oracle = MockOracle(true_class=mock_true_class)
    else:
        from verifier.oracle import QwenOracle
        oracle = QwenOracle(cfg)
    return oracle, query_set


def _run_episode(image_pil, cfg, policy, oracle, query_set, use_mock):
    """Build the ActionContext, run the episode, return (result, ctx)."""
    from agent.budget import CostMeter

    graph = OrchardGraph()
    cost = CostMeter()

    if PROCESSOR is not None:
        from pipeline import initialize_canopy_roi
        roi = initialize_canopy_roi(PROCESSOR, np.array(image_pil), graph)
        cost.n_sam += 1
        partition = [tuple(int(v) for v in roi)]
    else:
        w, h = image_pil.size
        partition = [(0, 0, w, h)]

    ctx = ActionContext(
        processor=PROCESSOR, image_pil=image_pil, graph=graph, cfg=cfg,
        oracle=oracle, query_set=query_set, discovery=DiscoveryCurve(),
        partition=partition, cost=cost,
    )

    if policy == "heuristic":
        pol = policy_heuristic.choose
        bootstrap, auto_stop = False, False
    else:
        if use_mock:
            stub = _StubChatClient(cfg)
            pol = runner.make_vlm_policy(ctx, inspect_client=stub, vlm_client=stub)
        else:
            pol = runner.make_vlm_policy(ctx)
        # Guided-ROI VLM policy episode wiring (mirrors eval.run_eval._run_agent_policy,
        # PROGRESS.md 7.3): the heuristic baseline is untouched.
        bootstrap, auto_stop = True, True

    result = runner.run_episode(image_pil, ctx, pol, max_actions=cfg.budget_max_actions,
                                bootstrap_global_pass=bootstrap, auto_stop=auto_stop)
    return result, ctx


def _classification_tally(graph):
    tally = {}
    for node in graph.nodes.values():
        tally[node.classification] = tally.get(node.classification, 0) + 1
    return tally


def run_orchestration(split, idx, prompt, policy, verifier, overlap_mode, gate_mode,
                      iou_threshold, iom_threshold, conf, budget, enable_thinking, use_mock,
                      query_file, mock_true_class):
    """Full-pipeline episode on one dataset image. Returns (sam3_view, gt_view, status, verbose)."""
    idx = int(idx)
    samples = get_samples(split)
    if not samples:
        placeholder = Image.new("RGB", (640, 400), "#c0392b")
        return (placeholder, placeholder, "Dataset unavailable.",
                f"ERROR: no samples loaded for split '{split}' from root '{DATASET_ROOT}'. "
                "Check CITRUS_DATASET_ROOT / the images+labels layout.")

    sample = samples[idx]
    image_pil = Image.open(sample["image_path"]).convert("RGB")
    gt_boxes, gt_count = sample["gt_boxes"], sample["count"]
    gt_view_plain = draw_gt_view(image_pil, gt_boxes, gt_count, None)

    if PROCESSOR is None:
        placeholder = Image.new("RGB", (640, 400), "#2c3e50")
        return (placeholder, gt_view_plain, "SAM3 model not loaded on this machine.",
                f"[FATAL] SAM3 weights did not load:\n{_LOAD_ERROR}\n\n"
                "Run this app on the GPU box with the SAM3 repo present.")

    target = (prompt or "").strip() or DEFAULT_PROMPT
    cfg = _build_cfg(target, verifier, overlap_mode, gate_mode, iou_threshold, iom_threshold,
                     conf, budget, query_file, enable_thinking)

    try:
        oracle, query_set = _build_oracle(cfg, verifier, use_mock, mock_true_class)
    except Exception as exc:
        return image_pil, gt_view_plain, f"ERROR building verifier: {exc}", str(exc)

    # Capture everything logged during the episode into the verbose window,
    # including policy_vlm's prompt/response lines -- this is the VLM decision trace.
    handler = _ListLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    header = [
        "=" * 78,
        f"RUN  dataset=citrus split={split} idx={idx}  policy={policy}  verifier={verifier}"
        f"  overlap={overlap_mode}  gate={gate_mode} (iou_t={cfg.nms_iou_threshold:.2f} "
        f"iom_t={cfg.nms_iom_threshold:.2f})  conf={cfg.conf:.2f}  budget={cfg.budget_max_actions}"
        f"  thinking={'on' if enable_thinking else 'off'}  mock={'on' if use_mock else 'off'}",
        f"TARGET CONCEPT: '{target}'   IMAGE: {os.path.basename(sample['image_path'])}",
        "=" * 78,
    ]
    try:
        result, ctx = _run_episode(image_pil, cfg, policy, oracle, query_set, use_mock)
        graph = result["graph"]
        counts = result["counts"]
    except Exception as exc:
        root.removeHandler(handler)
        root.setLevel(prev_level)
        detail = "\n".join(header + ["", "[PIPELINE EXCEPTION]", str(exc), "",
                                     "--- captured log ---"] + handler.records)
        return image_pil, gt_view_plain, f"Pipeline error: {exc}", detail
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    pred = counts["N_obs"]

    step_lines = ["", "--- ACTION LOG (per step) ---"]
    for e in result["log"]:
        step_lines.append(
            f"  t={e['t']:>2}  {e['action']:<11}  new={e['n_new']:>3}  "
            f"U={e['U']:.3f}  cost={e['cost_so_far']:.3f}"
        )

    footer = [
        "",
        "--- COUNT ESTIMATES ---",
        f"  N_obs={counts['N_obs']}   N_supp={counts['N_supp']}   N_cons={counts['N_cons']}"
        f"   (headline = N_obs)   GT={gt_count}",
        f"  classification tally: {_classification_tally(graph)}",
        f"  actions used: {len(result['log'])}   normalized cost: {ctx.cost.total(cfg):.3f}",
        f"  cost breakdown: {ctx.cost.as_dict()}",
    ]

    verbose = "\n".join(header + ["", "--- CAPTURED LOG (chronological) ---"]
                        + handler.records + step_lines + footer)

    sam3_view = draw_sam3_view(image_pil, graph, target)
    gt_view = draw_gt_view(image_pil, gt_boxes, gt_count, pred)
    if pred == gt_count:
        status = f"EXACT MATCH  |  N_obs={pred} == GT={gt_count}"
    else:
        direction = "under" if pred < gt_count else "over"
        status = f"MISMATCH ({direction})  |  N_obs={pred} vs GT={gt_count}   (N_supp={counts['N_supp']}, N_cons={counts['N_cons']})"
    return sam3_view, gt_view, status, verbose


# ==========================================
# 5. Navigation
# ==========================================
def load_sample_view(split, idx):
    """Load an image + GT overlay without running the pipeline (on-demand only)."""
    samples = get_samples(split)
    n = len(samples)
    if n == 0:
        placeholder = Image.new("RGB", (640, 400), "#c0392b")
        return (placeholder, placeholder, f"### split '{split}' empty/unavailable",
                f"Dataset unavailable: check CITRUS_DATASET_ROOT='{DATASET_ROOT}' and its "
                "images/<split> + labels/<split> layout.", 0)
    idx = max(0, min(int(idx), n - 1))
    sample = samples[idx]
    image_pil = Image.open(sample["image_path"]).convert("RGB")
    gt_view = draw_gt_view(image_pil, sample["gt_boxes"], sample["count"], None)
    banner = (f"### [{split}] index {idx} / {n - 1} | "
              f"{os.path.basename(sample['image_path'])} | GT count: {sample['count']}")
    return image_pil, gt_view, banner, "Loaded. Press ▶ Run Orchestration.", idx


def nav_next(split, idx):
    return load_sample_view(split, int(idx) + 1)


def nav_prev(split, idx):
    return load_sample_view(split, int(idx) - 1)


def nav_jump(split, idx_text):
    idx = int(idx_text) if str(idx_text).strip().lstrip("-").isdigit() else 0
    return load_sample_view(split, idx)


def on_split_change(split):
    return load_sample_view(split, 0)


# ==========================================
# 6. Gradio interface
# ==========================================
_QUERY_FILES = sorted(
    os.path.join("queries", f) for f in os.listdir("queries")
    if f.endswith(".json")
) if os.path.isdir("queries") else [Config().vip_query_file]
_DEFAULT_QUERY_FILE = "queries/green_citrus.json" if "queries/green_citrus.json" in _QUERY_FILES else _QUERY_FILES[0]

with gr.Blocks(theme=gr.themes.Soft(), title="Citrus Orchestration Sandbox") as app:
    idx_state = gr.State(value=0)

    gr.Markdown("# \U0001f34a Citrus Active-Perception Orchestration Sandbox")
    gr.Markdown(
        "Runs the **full agent orchestration** (policy -> action -> belief update) on "
        f"the local green-fruit dataset (`{DATASET_ROOT}`), with every action, per-pass "
        "SAM3 stat, verifier call, and VLM prompt/response streamed to the verbose window. "
        "Left = SAM3 sensor view (canopy ROI + candidate boxes/masks). Right = ground truth. "
        "Navigation only loads the image; press **Run Orchestration** to execute the pipeline."
    )
    if _LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **SAM3 not loaded:** `{_LOAD_ERROR}` -- runs will report this.")

    with gr.Row():
        with gr.Column(scale=5):
            banner_md = gr.Markdown("### Initializing...")
            with gr.Row():
                sam3_display = gr.Image(label="\U0001f4e1 SAM3 sensor view (left)", type="pil", interactive=False)
                gt_display = gr.Image(label="✅ Ground truth (right)", type="pil", interactive=False)
            status_box = gr.Textbox(label="\U0001f4ca Result", interactive=False)
            verbose_box = gr.Textbox(
                label="\U0001f52c Verbose orchestration log (actions · per-pass SAM3 · verifier · VLM prompts · estimates)",
                interactive=False, lines=24, max_lines=24,
            )

        with gr.Column(scale=2):
            split_dropdown = gr.Dropdown(
                choices=["train", "val", "test"], value="train", label="\U0001f4c2 Dataset split")

            with gr.Row():
                policy_dropdown = gr.Dropdown(
                    choices=["heuristic", "vlm"], value="heuristic", label="\U0001f9e0 Policy")
                verifier_radio = gr.Radio(
                    choices=["off", "ioc", "vip"], value="ioc", label="✅ Verification")

            with gr.Row():
                overlap_radio = gr.Radio(
                    choices=["box", "mask"], value="box",
                    label="\U0001f4d0 Overlap metric (geometry: boxes vs instance masks)")
                conf_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=Config().conf, step=0.05,
                    label="\U0001f39a️ Detection confidence")

            gate_mode_radio = gr.Radio(
                choices=["dual", "iou_only", "iom_only"], value=Config().gate_mode,
                label="\U0001f3af NMS suppression gate (dual = IoU+IoM default; pick one per dataset)")

            with gr.Row():
                iou_threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=Config().nms_iou_threshold, step=0.05,
                    label="Gate A: IoU threshold")
                iom_threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=Config().nms_iom_threshold, step=0.05,
                    label="Gate B: IoM threshold")

            budget_number = gr.Number(value=Config().budget_max_actions, precision=0,
                                      label="\U0001f501 Max actions (budget)")
            thinking_checkbox = gr.Checkbox(
                value=Config().policy_enable_thinking,
                label="\U0001f9e0 Enable VLM thinking trace (policy loop)")

            with gr.Accordion("FM+V-IP (vip) options", open=False):
                query_file_dropdown = gr.Dropdown(
                    choices=_QUERY_FILES, value=_DEFAULT_QUERY_FILE, label="Query set (vip)")
                mock_checkbox = gr.Checkbox(
                    value=False, label="\U0001f9ea Mock backend (offline: MockOracle + stubbed VLM)")
                mock_class_dropdown = gr.Dropdown(
                    choices=["target", "distractor", "spurious"], value="target",
                    label="MockOracle assumed true class (UI testing only)")

            prompt_input = gr.Textbox(value=DEFAULT_PROMPT, label="✏️ Target concept (drives SAM3 + agent)")

            run_button = gr.Button("▶ Run Orchestration", variant="primary")

            gr.HTML("<hr>")
            with gr.Row():
                btn_prev = gr.Button("⬅️ Prev")
                btn_next = gr.Button("Next ➡️")
            with gr.Row():
                jump_input = gr.Textbox(label="Jump to index", placeholder="e.g. 42", scale=2)
                btn_jump = gr.Button("\U0001f3af Jump", scale=1)

    nav_outputs = [sam3_display, gt_display, banner_md, status_box, idx_state]
    run_inputs = [split_dropdown, idx_state, prompt_input, policy_dropdown, verifier_radio,
                  overlap_radio, gate_mode_radio, iou_threshold_slider, iom_threshold_slider,
                  conf_slider, budget_number, thinking_checkbox, mock_checkbox,
                  query_file_dropdown, mock_class_dropdown]
    run_outputs = [sam3_display, gt_display, status_box, verbose_box]

    app.load(fn=load_sample_view, inputs=[split_dropdown, idx_state], outputs=nav_outputs)
    split_dropdown.change(fn=on_split_change, inputs=[split_dropdown], outputs=nav_outputs)

    run_button.click(fn=run_orchestration, inputs=run_inputs, outputs=run_outputs)
    prompt_input.submit(fn=run_orchestration, inputs=run_inputs, outputs=run_outputs)

    btn_next.click(fn=nav_next, inputs=[split_dropdown, idx_state], outputs=nav_outputs)
    btn_prev.click(fn=nav_prev, inputs=[split_dropdown, idx_state], outputs=nav_outputs)
    btn_jump.click(fn=nav_jump, inputs=[split_dropdown, jump_input], outputs=nav_outputs)


def main():
    logger.info("Launching citrus orchestration sandbox on %s", device)
    app.launch()


if __name__ == "__main__":
    main()

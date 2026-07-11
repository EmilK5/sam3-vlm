"""
citrus_orchestration_app.py

Three-window prompt-refinement experiment (Phase 9). For each image the app shows
THREE panels side by side so the human can eyeball whether the VLM refining the
SAM3 text prompt actually helps:

  1. Ground truth        - GT boxes + count.
  2. Generic SAM3 arm    - a FIXED two-call cascade on ONE prompt (no VLM):
                           canopy ROI + leaf map + global pass @ cfg.seed_conf
                           (seeds confident exemplars) + tiled pass @
                           cfg.refine_conf_default, sharing those exemplars.
  3. Active-refine arm    - the v3 loop (agent/policy_vlm_v3): bootstrap global
                           pass @ cfg.seed_conf on the tree ROI, then the VLM
                           refines the TEXT PROMPT (1-2 adjectives + noun) +
                           threshold each step over the SAME region (the tree ROI,
                           one global pass) until it stops / discovery saturates /
                           budget is hit.

Grounding invariant (CLAUDE.md #3): the VLM only emits text; SAM3 grounds every
box. No sub-ROI zoom -- the region is fixed, which keeps every found exemplar in
frame (SAM3 exemplars are welded to the query frame) and isolates the causal effect
of prompt refinement. This is a debug/intuition tool; the statistical claim
(adaptive vs. equal-budget random/fixed prompt schedule) is a later eval ablation.

Standalone driver: it does NOT touch app.py, pipeline.py, or any validated core
module -- it only *uses* their public entry points (pipeline.execute_pass /
initialize_canopy_roi, agent.runner.run_episode / make_refine_policy,
agent.belief.count_estimates, eval.datasets.load_split, verifier.oracle/queries,
inference.load_sam3_model). Run on the GPU box -- SAM3 (and, for a live run, a
Qwen-3-VL endpoint) must be reachable; use the Mock toggle to dry-run the loop.

Dataset: a local YOLO-format green-fruit/citrus split (images/<split>,
labels/<split>). Root defaults to "datasets/citruses_dataset" (override with
CITRUS_DATASET_ROOT); split falls back to a flat root/images layout.
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
from agent.belief import DiscoveryCurve, count_estimates
from agent.budget import CostMeter
from agent import runner
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
_ROI_COLOR = (255, 215, 0)         # canopy / tree ROI
_GT_COLOR = (0, 229, 255)


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_sam3_view(image_pil, graph, prompt_str, arm_label="SAM3"):
    """A SAM3 arm panel: canopy ROI box + every candidate's box, colored by verifier
    classification, with a translucent mask fill wherever the node carries one
    (overlap_mode='mask', non-tiled passes). The banner names the arm + the (final)
    prompt + the per-class tally."""
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
        draw.text((roi[0] + 4, max(0, roi[1] - 16)), "tree ROI", fill=_ROI_COLOR, font=font)

    tally = {}
    for node in graph.nodes.values():
        x1, y1, x2, y2 = node.box
        rgb = _CLASS_COLORS.get(node.classification, (255, 255, 255))
        tally[node.classification] = tally.get(node.classification, 0) + 1
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
        tag = f"P{node.found_in_pass} {node.classification[:1]}:{node.scores['detection_confidence']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=rgb, font=font)

    banner = f"{arm_label} | prompt='{prompt_str}' | {tally or 'no candidates'}"
    draw.text((8, 8), banner, fill=(255, 255, 255), font=font)
    return canvas


def draw_gt_view(image_pil, gt_boxes, gt_count):
    """Ground-truth panel: GT boxes + count."""
    canvas = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for box in gt_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=_GT_COLOR, width=3)
    draw.text((8, 8), f"GROUND TRUTH | count={gt_count}", fill=(255, 255, 255), font=font)
    return canvas


# ==========================================
# 3. Offline mock client (dry-run the refine loop without a live Qwen endpoint)
# ==========================================
class _Resp:
    """Minimal stand-in for an OpenAI chat completion response."""
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _StubRefineClient:
    """Deterministic offline stand-in for the Qwen endpoint driving the v3 refine
    policy: emits ONE refined prompt ("round <target-noun>") then "stop". So a mock
    episode runs the genuine bootstrap -> refine -> stop loop without a network,
    exercising the real policy_vlm_v3.choose validation path (not a silent
    heuristic fallback). Used only when the Mock toggle is on."""
    def __init__(self, cfg):
        self.cfg = cfg
        self._calls = 0
        self.chat = self
        self.completions = self

    def create(self, model=None, temperature=None, messages=None, **kwargs):
        self._calls += 1
        if self._calls >= 2:
            action = {"action": "stop", "args": {"estimate_name": "N_obs"}}
        else:
            noun = (getattr(self.cfg, "target_prompt", "fruit").split() or ["fruit"])[-1]
            action = {"action": "refine",
                      "args": {"prompt": f"round {noun}",
                               "threshold": self.cfg.refine_conf_default}}
        return _Resp(json.dumps(action))


# ==========================================
# 4. Experiment driver
# ==========================================
class _ListLogHandler(logging.Handler):
    """Captures formatted log records emitted during one experiment run."""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


def _build_cfg(prompt, verifier, overlap_mode, gate_mode, iou_threshold, iom_threshold,
               dedup_metric, dedup_threshold, seed_conf, budget, query_file, enable_thinking):
    """Both arms share one cfg. conf is set to seed_conf so the arm-3 bootstrap
    (runner uses cfg.conf) seeds confident exemplars; refine passes carry their own
    VLM-chosen threshold, and the generic arm passes seed_conf/refine_conf_default
    explicitly."""
    base = Config()
    return dataclasses.replace(
        base,
        verifier_mode=verifier,
        overlap_mode=overlap_mode,
        gate_mode=gate_mode,
        nms_iou_threshold=float(iou_threshold),
        nms_iom_threshold=float(iom_threshold),
        cross_pass_dedup_metric=dedup_metric,
        cross_pass_dedup_threshold=float(dedup_threshold),
        conf=float(seed_conf),
        seed_conf=float(seed_conf),
        target_prompt=prompt,
        budget_max_actions=int(budget),
        vip_query_file=query_file or base.vip_query_file,
        policy_enable_thinking=bool(enable_thinking),
    )


def _build_oracle(cfg, verifier, use_mock, mock_true_class, oracle_kind):
    """Return (oracle, query_set) for the run. Both None unless verifier='vip'."""
    if verifier != "vip":
        return None, None
    from verifier.queries import load_query_set
    query_set = load_query_set(cfg.vip_query_file)
    if use_mock:
        from verifier.oracle import MockOracle
        oracle = MockOracle(true_class=mock_true_class)
    elif oracle_kind == "router":
        from verifier.oracle import RouterOracle, QwenOracle
        oracle = RouterOracle(cfg, processor=PROCESSOR, vlm_oracle=QwenOracle(cfg))
    else:
        from verifier.oracle import QwenOracle
        oracle = QwenOracle(cfg)
    return oracle, query_set


def _pass_kwargs(cfg, oracle, query_set, graph):
    return dict(
        processor=PROCESSOR, graph=graph, cfg=cfg, oracle=oracle, query_set=query_set,
        nms_mode=getattr(cfg, "nms_mode", "dualgate"), gate_mode=cfg.gate_mode,
        use_canopy_roi=cfg.use_canopy_roi, nms_iou_threshold=cfg.nms_iou_threshold,
        nms_iom_threshold=cfg.nms_iom_threshold,
        cross_pass_dedup_metric=cfg.cross_pass_dedup_metric,
        cross_pass_dedup_threshold=cfg.cross_pass_dedup_threshold,
    )


def run_generic_arm(image_pil, cfg, oracle, query_set):
    """Arm 2: fixed two-call cascade on ONE prompt, no VLM. Global @ seed_conf
    (seeds confident exemplars) then tiled @ refine_conf_default (shares them)."""
    from pipeline import execute_pass
    graph = OrchardGraph()
    cost = CostMeter()
    kw = _pass_kwargs(cfg, oracle, query_set, graph)
    s1 = execute_pass(image_pil=image_pil, conf=cfg.seed_conf, clahe=False, tiling=False,
                      pass_number=1, prompt=cfg.target_prompt, **kw)
    s2 = execute_pass(image_pil=image_pil, conf=cfg.refine_conf_default, clahe=False, tiling=True,
                      pass_number=2, prompt=cfg.target_prompt, **kw)
    for s in (s1, s2):
        cost.n_sam += int(getattr(s, "n_sam_calls", 0))
        cost.n_tile += int(getattr(s, "n_tiles", 0))
        cost.n_verify += int(getattr(s, "n_verify_calls", 0))
    return {"graph": graph, "counts": count_estimates(graph, cfg), "cost": cost, "stats": [s1, s2]}


def run_active_arm(image_pil, cfg, oracle, query_set, vlm_client=None):
    """Arm 3: the v3 refine loop. Bootstrap global pass @ seed_conf on the tree ROI,
    then make_refine_policy drives prompt+threshold refinement over the same region
    with the hybrid stop (VLM stop / saturation / budget)."""
    from pipeline import initialize_canopy_roi
    graph = OrchardGraph()
    cost = CostMeter()
    roi = initialize_canopy_roi(PROCESSOR, np.array(image_pil), graph)
    cost.n_sam += 1  # the canopy sweep is a real global SAM3 call
    ctx = ActionContext(
        processor=PROCESSOR, image_pil=image_pil, graph=graph, cfg=cfg,
        oracle=oracle, query_set=query_set, discovery=DiscoveryCurve(),
        partition=[tuple(int(v) for v in roi)], cost=cost,
    )
    pol = runner.make_refine_policy(ctx, vlm_client=vlm_client)
    result = runner.run_episode(image_pil, ctx, pol, max_actions=cfg.budget_max_actions,
                                bootstrap_global_pass=True, bootstrap_tiled_pass=True,
                                auto_stop=True)
    result["ctx"] = ctx
    return result


def _final_prompt(history, default):
    """The last text prompt the active arm actually sensed (bootstrap or a refine)."""
    last = default
    for rec in history:
        p = rec.get("x", {}).get("prompt")
        if p:
            last = p
    return last


def _tally(graph):
    t = {}
    for node in graph.nodes.values():
        t[node.classification] = t.get(node.classification, 0) + 1
    return t


def run_experiment(split, idx, prompt, verifier, overlap_mode, gate_mode, iou_threshold,
                   iom_threshold, dedup_metric, dedup_threshold, seed_conf, budget,
                   enable_thinking, use_mock, query_file, mock_true_class, oracle_kind):
    """Run both arms on one dataset image. Returns (gt_view, generic_view,
    active_view, status, verbose)."""
    idx = int(idx)
    samples = get_samples(split)
    if not samples:
        ph = Image.new("RGB", (512, 400), "#c0392b")
        return (ph, ph, ph, "Dataset unavailable.",
                f"ERROR: no samples for split '{split}' from root '{DATASET_ROOT}'. "
                "Check CITRUS_DATASET_ROOT / the images+labels layout.")

    sample = samples[idx]
    image_pil = Image.open(sample["image_path"]).convert("RGB")
    gt_boxes, gt_count = sample["gt_boxes"], sample["count"]
    gt_view = draw_gt_view(image_pil, gt_boxes, gt_count)

    if PROCESSOR is None:
        ph = Image.new("RGB", (512, 400), "#2c3e50")
        return (gt_view, ph, ph, "SAM3 model not loaded on this machine.",
                f"[FATAL] SAM3 weights did not load:\n{_LOAD_ERROR}\n\n"
                "Run this app on the GPU box with the SAM3 repo present.")

    target = (prompt or "").strip() or DEFAULT_PROMPT
    cfg = _build_cfg(target, verifier, overlap_mode, gate_mode, iou_threshold, iom_threshold,
                     dedup_metric, dedup_threshold, seed_conf, budget, query_file, enable_thinking)

    try:
        oracle, query_set = _build_oracle(cfg, verifier, use_mock, mock_true_class, oracle_kind)
    except Exception as exc:
        return gt_view, image_pil, image_pil, f"ERROR building verifier: {exc}", str(exc)

    handler = _ListLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    oracle_desc = "mock" if use_mock else (oracle_kind if verifier == "vip" else "n/a")
    header = [
        "=" * 84,
        f"EXPERIMENT  dataset=citrus split={split} idx={idx}  verifier={verifier}"
        f"  oracle={oracle_desc}  seed_conf={cfg.seed_conf:.2f}  refine_conf={cfg.refine_conf_default:.2f}"
        f"  budget={cfg.budget_max_actions}  thinking={'on' if enable_thinking else 'off'}"
        f"  mock={'on' if use_mock else 'off'}",
        f"INITIAL CONCEPT: '{target}'   IMAGE: {os.path.basename(sample['image_path'])}   GT={gt_count}",
        "=" * 84,
    ]

    try:
        generic = run_generic_arm(image_pil, cfg, oracle, query_set)
        vlm_client = _StubRefineClient(cfg) if use_mock else None
        active = run_active_arm(image_pil, cfg, oracle, query_set, vlm_client=vlm_client)
    except Exception as exc:
        root.removeHandler(handler)
        root.setLevel(prev_level)
        detail = "\n".join(header + ["", "[PIPELINE EXCEPTION]", str(exc), "",
                                     "--- captured log ---"] + handler.records)
        return gt_view, image_pil, image_pil, f"Pipeline error: {exc}", detail
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    g_graph, g_counts = generic["graph"], generic["counts"]
    a_graph, a_counts = active["graph"], active["counts"]
    a_history = active.get("history", [])
    final_prompt = _final_prompt(a_history, target)

    generic_view = draw_sam3_view(image_pil, g_graph, target, arm_label="GENERIC (fixed cascade)")
    active_view = draw_sam3_view(image_pil, a_graph, final_prompt, arm_label="ACTIVE (VLM refine)")

    # Refinement trajectory (x_t -> y_t) the VLM produced -- the story of the run.
    traj = ["", "--- ACTIVE ARM: prompt refinement trajectory (x_t -> y_t) ---"]
    for rec in a_history:
        x, y = rec["x"], rec["y"]
        p = x.get("prompt")
        label = x["action"] if p is None else f"{x['action']} prompt='{p}' @{x.get('conf'):.2f}"
        traj.append(
            f"  t={rec['t']:>2}  {label:<48}  ->  det={y.get('n_detections', 0)} "
            f"new={y['n_new']} redet={y.get('n_redetected', 0)} totals={y['totals']}")

    footer = [
        "",
        "--- RESULTS (headline = N_obs) ---",
        f"  GROUND TRUTH   N* = {gt_count}",
        f"  GENERIC arm    N_obs={g_counts['N_obs']}  N_supp={g_counts['N_supp']}  N_cons={g_counts['N_cons']}"
        f"   tally={_tally(g_graph)}   cost={generic['cost'].total(cfg):.3f}",
        f"  ACTIVE  arm    N_obs={a_counts['N_obs']}  N_supp={a_counts['N_supp']}  N_cons={a_counts['N_cons']}"
        f"   tally={_tally(a_graph)}   cost={active['ctx'].cost.total(cfg):.3f}"
        f"   actions={len(active.get('log', []))}",
    ]

    verbose = "\n".join(header + ["", "--- CAPTURED LOG (chronological) ---"]
                        + handler.records + traj + footer)

    def _delta(n):
        return f"{n - gt_count:+d}"
    status = (f"GT={gt_count}  |  GENERIC N_obs={g_counts['N_obs']} ({_delta(g_counts['N_obs'])})  "
              f"|  ACTIVE N_obs={a_counts['N_obs']} ({_delta(a_counts['N_obs'])})  |  final prompt='{final_prompt}'")
    return gt_view, generic_view, active_view, status, verbose


# ==========================================
# 5. Navigation (load the image + GT only; running the experiment is on demand)
# ==========================================
def load_sample_view(split, idx):
    samples = get_samples(split)
    n = len(samples)
    if n == 0:
        ph = Image.new("RGB", (512, 400), "#c0392b")
        return (ph, None, None, f"### split '{split}' empty/unavailable",
                f"Dataset unavailable: check CITRUS_DATASET_ROOT='{DATASET_ROOT}' and its "
                "images/<split> + labels/<split> layout.", 0)
    idx = max(0, min(int(idx), n - 1))
    sample = samples[idx]
    image_pil = Image.open(sample["image_path"]).convert("RGB")
    gt_view = draw_gt_view(image_pil, sample["gt_boxes"], sample["count"])
    banner = (f"### [{split}] index {idx} / {n - 1} | "
              f"{os.path.basename(sample['image_path'])} | GT count: {sample['count']}")
    return gt_view, None, None, banner, "Loaded. Press ▶ Run Experiment.", idx


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

with gr.Blocks(theme=gr.themes.Soft(), title="Citrus Prompt-Refinement Experiment") as app:
    idx_state = gr.State(value=0)

    gr.Markdown("# \U0001f34a Citrus Prompt-Refinement Experiment")
    gr.Markdown(
        "Three windows per image: **Ground truth** | **Generic** (fixed cascade, one "
        "prompt) | **Active** (the VLM refines the SAM3 text prompt + threshold over "
        "the tree ROI, `agent/policy_vlm_v3`). The VLM only writes text; SAM3 grounds "
        "every box; the region never changes. Navigation only loads the image; press "
        "**Run Experiment** to execute both arms."
    )
    if _LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **SAM3 not loaded:** `{_LOAD_ERROR}` -- runs will report this.")

    with gr.Row():
        with gr.Column(scale=5):
            banner_md = gr.Markdown("### Initializing...")
            with gr.Row():
                gt_display = gr.Image(label="✅ Ground truth", type="pil", interactive=False)
                generic_display = gr.Image(label="\U0001f4e6 Generic (fixed cascade)", type="pil", interactive=False)
                active_display = gr.Image(label="\U0001f9e0 Active (VLM refine)", type="pil", interactive=False)
            status_box = gr.Textbox(label="\U0001f4ca Result", interactive=False)
            verbose_box = gr.Textbox(
                label="\U0001f52c Verbose log (both arms · refinement trajectory · counts)",
                interactive=False, lines=24, max_lines=24,
            )

        with gr.Column(scale=2):
            split_dropdown = gr.Dropdown(
                choices=["train", "val", "test"], value="train", label="\U0001f4c2 Dataset split")

            prompt_input = gr.Textbox(value=DEFAULT_PROMPT, label="✏️ Initial concept (seed prompt for both arms)")

            with gr.Row():
                seed_conf_slider = gr.Slider(
                    minimum=0.30, maximum=0.90, value=Config().seed_conf, step=0.05,
                    label="\U0001f39a️ Seed confidence (global/bootstrap pass)")
                budget_number = gr.Number(value=Config().budget_max_actions, precision=0,
                                          label="\U0001f501 Active budget (max actions)")

            verifier_radio = gr.Radio(
                choices=["off", "ioc", "vip"], value="ioc", label="✅ Verification (both arms)")

            with gr.Row():
                overlap_radio = gr.Radio(
                    choices=["box", "mask"], value="box", label="\U0001f4d0 Overlap metric")
                gate_mode_radio = gr.Radio(
                    choices=["dual", "iou_only", "iom_only"], value=Config().gate_mode,
                    label="\U0001f3af NMS gate")

            with gr.Row():
                iou_threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=Config().nms_iou_threshold, step=0.05,
                    label="Gate A: IoU")
                iom_threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=Config().nms_iom_threshold, step=0.05,
                    label="Gate B: IoM")

            with gr.Row():
                dedup_metric_radio = gr.Radio(
                    choices=["iou", "iom"], value=Config().cross_pass_dedup_metric,
                    label="Cross-pass dedup metric")
                dedup_threshold_slider = gr.Slider(
                    minimum=0.10, maximum=0.95, value=Config().cross_pass_dedup_threshold, step=0.05,
                    label="Cross-pass dedup threshold")

            thinking_checkbox = gr.Checkbox(
                value=Config().policy_enable_thinking, label="\U0001f9e0 VLM thinking trace (policy loop)")

            with gr.Accordion("FM+V-IP (vip) options", open=False):
                query_file_dropdown = gr.Dropdown(
                    choices=_QUERY_FILES, value=_DEFAULT_QUERY_FILE, label="Query set (vip)")
                oracle_dropdown = gr.Dropdown(
                    choices=["qwen", "router"], value=Config().oracle_kind, label="\U0001f5c2️ Oracle (non-mock)")
                mock_checkbox = gr.Checkbox(
                    value=False, label="\U0001f9ea Mock backend (offline: MockOracle + stubbed refine VLM)")
                mock_class_dropdown = gr.Dropdown(
                    choices=["target", "distractor", "spurious"], value="target",
                    label="MockOracle assumed true class (UI testing only)")

            run_button = gr.Button("▶ Run Experiment", variant="primary")

            gr.HTML("<hr>")
            with gr.Row():
                btn_prev = gr.Button("⬅️ Prev")
                btn_next = gr.Button("Next ➡️")
            with gr.Row():
                jump_input = gr.Textbox(label="Jump to index", placeholder="e.g. 42", scale=2)
                btn_jump = gr.Button("\U0001f3af Jump", scale=1)

    nav_outputs = [gt_display, generic_display, active_display, banner_md, status_box, idx_state]
    run_inputs = [split_dropdown, idx_state, prompt_input, verifier_radio, overlap_radio,
                  gate_mode_radio, iou_threshold_slider, iom_threshold_slider, dedup_metric_radio,
                  dedup_threshold_slider, seed_conf_slider, budget_number, thinking_checkbox,
                  mock_checkbox, query_file_dropdown, mock_class_dropdown, oracle_dropdown]
    run_outputs = [gt_display, generic_display, active_display, status_box, verbose_box]

    app.load(fn=load_sample_view, inputs=[split_dropdown, idx_state], outputs=nav_outputs)
    split_dropdown.change(fn=on_split_change, inputs=[split_dropdown], outputs=nav_outputs)

    run_button.click(fn=run_experiment, inputs=run_inputs, outputs=run_outputs)
    prompt_input.submit(fn=run_experiment, inputs=run_inputs, outputs=run_outputs)

    btn_next.click(fn=nav_next, inputs=[split_dropdown, idx_state], outputs=nav_outputs)
    btn_prev.click(fn=nav_prev, inputs=[split_dropdown, idx_state], outputs=nav_outputs)
    btn_jump.click(fn=nav_jump, inputs=[split_dropdown, jump_input], outputs=nav_outputs)


def main():
    logger.info("Launching citrus prompt-refinement experiment on %s", device)
    app.launch()


if __name__ == "__main__":
    main()

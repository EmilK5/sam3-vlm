"""
orchestration_app.py

Interactive Gradio dashboard that runs the FULL active-perception orchestration
pipeline of this repo (agent episode: policy -> action -> belief update) on a
chosen benchmark image, with a verbose log window exposing every action, per-pass
SAM3 diagnostic, verifier call, and count estimate.

This is a standalone driver (it does NOT touch app.py, pipeline.py, or any of the
validated core modules). It only *uses* their public entry points:

    agent.runner.run_episode            - the orchestration loop
    agent.policy_heuristic / policy_vlm - the two policies
    agent.actions.ActionContext         - the episode sensing state
    agent.belief.count_estimates        - N_obs / N_supp / N_cons
    pipeline.initialize_canopy_roi      - canopy ROI for the initial partition
    verifier.queries.load_query_set     - FM+V-IP query set (vip mode)
    verifier.oracle.MockOracle/QwenOracle
    inference.load_sam3_model

Datasets (access layer shared with eval/run_eval.py, see eval/datasets.py's
count-only section; keyed off a dropdown here):
    countbench  - HF nielsr/countbench (train split)      [needs `datasets`]
    pixmo       - HF allenai/pixmo-count (test split)      [needs `datasets`, network]
    carpk       - local CARPK_devkit (Images/Annotations)  [needs CARPK_BASE_DIR]

NOTE ON DEPENDENCIES: CountBench and PixMo-Count require the HuggingFace
`datasets` library, which is not in the repo's core allowed-deps set. It is
imported lazily (only when one of those datasets is selected), so CARPK and the
rest of the app work without it. Run this on the GPU box (SAM3 weights load at
startup); it is not runnable on a CPU-only dev box.
"""

import os
import sys
import json
import logging

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 0. Paths (machine-specific; mirror app.py / sandbox.py)
# ==========================================
SAM3_REPO_ROOT = "/home/ekielar/sam3"
BPE_PATH = f"{SAM3_REPO_ROOT}/assets/bpe_simple_vocab_16e6.txt.gz"

if SAM3_REPO_ROOT not in sys.path:
    sys.path.append(SAM3_REPO_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Per-dataset UI starting points (overlap "mask" is orchard-specific -> keep "box").
#
# Two SEPARATE IoU/IoM axes -- easy to conflate:
#   - gate_mode/nms_iou_threshold/nms_iom_threshold: INTRA-pass NMS, deduping
#     multiple detections from the SAME SAM3 call. The validated reference
#     (sandbox.py's inference.apply_nms) always ORs both gates at high, barely-
#     engaging thresholds (0.95/0.95) -- so this stays "dual" here, not
#     iou_only/iom_only, matching that reference exactly.
#   - cross_pass_dedup_metric/cross_pass_dedup_threshold: INTER-pass dedup,
#     deciding whether a box just found is the same object as one already
#     registered from an earlier pass/tile. This is the knob that actually
#     mattered in the validated reference: sparse/varied scenes (pixmo/
#     countbench) use plain IoU at a moderate threshold; CARPK's dense
#     uniform-size grids use IoM (Intersection over Minimum), which correctly
#     merges a tight vs. loose detection of the same car where IoU alone
#     would under-merge.
DATASET_UI_DEFAULTS = {
    "countbench": dict(overlap_mode="box", conf=0.35, gate_mode="dual",
                       nms_iou_threshold=0.95, nms_iom_threshold=0.95,
                       cross_pass_dedup_metric="iou", cross_pass_dedup_threshold=0.60),
    "pixmo": dict(overlap_mode="box", conf=0.35, gate_mode="dual",
                 nms_iou_threshold=0.95, nms_iom_threshold=0.95,
                 cross_pass_dedup_metric="iou", cross_pass_dedup_threshold=0.65),
    "carpk": dict(overlap_mode="box", conf=0.45, gate_mode="dual",
                 nms_iou_threshold=0.95, nms_iom_threshold=0.95,
                 cross_pass_dedup_metric="iom", cross_pass_dedup_threshold=0.85),
}

# Local imports that pull torch/transformers/pipeline. Kept after sys.path setup.
from config import Config
from graph import OrchardGraph
from agent.actions import ActionContext
from agent.belief import DiscoveryCurve
from agent import runner, policy_heuristic
import inference
# Dataset access layer shared with eval/run_eval.py (aliased so call sites below
# are unchanged): get_count_dataset->get_dataset, count_dataset_size->total_items,
# get_count_sample->get_sample.
from eval.datasets import (
    get_count_dataset as get_dataset,
    count_dataset_size as total_items,
    get_prompt_map,
    default_prompt_for,
    get_count_sample as get_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# Load SAM3 once. If weights are unavailable (e.g. CPU dev box), keep the UI alive
# and surface the failure at run time instead of crashing on import.
try:
    MODEL, PROCESSOR = inference.load_sam3_model(BPE_PATH, 0.35, device=device)
    _LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - depends on machine weights
    MODEL, PROCESSOR = None, None
    _LOAD_ERROR = str(exc)
    logger.warning("SAM3 model failed to load: %s", exc)


# ==========================================
# 2. Rendering
# ==========================================
_CLASS_COLORS = {
    "fruit": "#39FF14",       # verified target
    "leaf": "#FF3B3B",        # verified distractor
    "spurious": "#9AA0A6",    # rejected
    "unresolved": "#00E5FF",  # unverified candidate (still counted toward N_obs)
}


def draw_boxes(image_pil, graph, prompt_str, gt, pred):
    """Overlay every candidate box coloured by classification, with a status banner."""
    canvas = image_pil.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    match = (gt is not None and pred == gt)
    banner = (f"PROMPT: '{prompt_str}' | GT: {gt} | PRED(N_obs): {pred} | "
              f"{'MATCH' if match else 'MISMATCH'}")
    draw.text((8, 8), banner, fill="#2ECC71" if match else "#E74C3C", font=font)

    for node in graph.nodes.values():
        x1, y1, x2, y2 = node.box
        color = _CLASS_COLORS.get(node.classification, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"P{node.found_in_pass} {node.classification[:1]}:{node.scores['detection_confidence']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=color, font=font)
    return canvas


# ==========================================
# 3. Offline mock client (Mock toggle for vip verifier / VLM policy)
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
# 4. Orchestration driver (mirrors eval.run_eval agent wiring)
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
               dedup_metric, dedup_threshold, conf, budget, query_file):
    import dataclasses
    return dataclasses.replace(
        Config(),
        verifier_mode=verifier,
        overlap_mode=overlap_mode,
        gate_mode=gate_mode,
        nms_iou_threshold=float(iou_threshold),
        nms_iom_threshold=float(iom_threshold),
        cross_pass_dedup_metric=dedup_metric,
        cross_pass_dedup_threshold=float(dedup_threshold),
        conf=float(conf),
        target_prompt=prompt,
        budget_max_actions=int(budget),
        vip_query_file=query_file or Config().vip_query_file,
        # None of countbench/pixmo/carpk have a canopy -- the "tree canopy" SAM3
        # sweep would only ever return a spurious/empty match here, so it's
        # always off for this app (unlike citrus_orchestration_app.py, whose
        # single dataset is a real orchard and keeps the default True).
        use_canopy_roi=False,
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


def _forced_tile_pass(ctx, cfg):
    """Unconditionally run one global SAM3 pass over the anchor region before
    the policy loop starts (v2: the action space has no tiled action, so the
    old forced TileQueryA became a global QueryA over partition[0]). Uses only
    the public agent.actions API (same call runner.run_episode makes internally),
    so no core file is touched."""
    from agent.actions import QueryA, execute as agent_execute
    prompt = getattr(cfg, "target_prompt", "green fruit")
    conf = getattr(cfg, "conf", 0.3)
    region = tuple(ctx.partition[0]) if ctx.partition else (0, 0, 0, 0)
    n_new = int(agent_execute(QueryA(region=region, prompt=prompt, conf=conf), ctx))
    ctx.discovery.append(n_new)
    logging.getLogger("agent.runner").info(json.dumps({
        "t": 0, "action": "QueryA(forced-initial-global)", "n_new": n_new,
        "cost_so_far": round(ctx.cost.total(cfg), 3),
    }))


def _run_episode(image_pil, cfg, policy, oracle, query_set, use_mock, force_tile=False):
    """Build the ActionContext, run the episode, return (result, ctx)."""
    from agent.budget import CostMeter

    graph = OrchardGraph()
    cost = CostMeter()

    if PROCESSOR is not None:
        from pipeline import initialize_canopy_roi
        use_canopy_roi = getattr(cfg, "use_canopy_roi", True)
        if use_canopy_roi:
            cost.n_sam += 1
        roi = initialize_canopy_roi(PROCESSOR, np.array(image_pil), graph, use_canopy=use_canopy_roi)
        partition = [tuple(int(v) for v in roi)]
    else:
        w, h = image_pil.size
        partition = [(0, 0, w, h)]

    ctx = ActionContext(
        processor=PROCESSOR, image_pil=image_pil, graph=graph, cfg=cfg,
        oracle=oracle, query_set=query_set, discovery=DiscoveryCurve(),
        partition=partition, cost=cost,
    )

    if force_tile:
        _forced_tile_pass(ctx, cfg)

    if policy == "heuristic":
        pol = policy_heuristic.choose
    elif use_mock:
        # v2: make_vlm_policy takes only vlm_client (scene-inspection retired in 8.3).
        stub = _StubChatClient(cfg)
        pol = runner.make_vlm_policy(ctx, vlm_client=stub)
    else:
        pol = runner.make_vlm_policy(ctx)

    # v2: VLM episodes ALWAYS open with the mandatory bootstrap (canopy_roi /
    # leaf_map / global_pass records) and auto-stop on saturation, mirroring
    # eval.run_eval._run_agent_policy; the heuristic baseline is untouched.
    is_vlm = policy != "heuristic"
    remaining_budget = max(0, cfg.budget_max_actions - (1 if force_tile else 0))
    result = runner.run_episode(image_pil, ctx, pol, max_actions=remaining_budget,
                                bootstrap_global_pass=is_vlm, auto_stop=is_vlm)
    return result, ctx


def _classification_tally(graph):
    tally = {}
    for node in graph.nodes.values():
        tally[node.classification] = tally.get(node.classification, 0) + 1
    return tally


def run_orchestration(dataset, idx, prompt, policy, verifier, overlap_mode, gate_mode,
                      iou_threshold, iom_threshold, dedup_metric, dedup_threshold, conf,
                      budget, force_tile, use_mock, query_file, mock_true_class):
    """Full-pipeline execution on one image. Returns (image, banner_md, status, verbose)."""
    idx = int(idx)
    if PROCESSOR is None:
        placeholder = Image.new("RGB", (640, 400), "#2c3e50")
        return (placeholder, "### SAM3 unavailable",
                "SAM3 model not loaded on this machine.",
                f"[FATAL] SAM3 weights did not load:\n{_LOAD_ERROR}\n\n"
                "Run this app on the GPU box with the SAM3 repo present.")

    image_pil, raw_prompt, gt, err = get_sample(dataset, idx)
    banner = f"### [{dataset}] index {idx} | Ground Truth: {gt}"
    if err is not None:
        return Image.new("RGB", (640, 400), "#c0392b"), banner, f"ERROR: {err}", err

    target = (prompt or "").strip() or default_prompt_for(dataset, raw_prompt)
    cfg = _build_cfg(target, verifier, overlap_mode, gate_mode, iou_threshold, iom_threshold,
                     dedup_metric, dedup_threshold, conf, budget, query_file)

    try:
        oracle, query_set = _build_oracle(cfg, verifier, use_mock, mock_true_class)
    except Exception as exc:
        return image_pil, banner, f"ERROR building verifier: {exc}", str(exc)

    # Capture everything logged during the episode into the verbose window.
    handler = _ListLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    header = [
        "=" * 78,
        f"RUN  dataset={dataset} idx={idx}  policy={policy}  verifier={verifier}"
        f"  overlap={overlap_mode}  nms_gate={gate_mode} (iou_t={cfg.nms_iou_threshold:.2f} "
        f"iom_t={cfg.nms_iom_threshold:.2f})  dedup={cfg.cross_pass_dedup_metric}@"
        f"{cfg.cross_pass_dedup_threshold:.2f}  conf={cfg.conf:.2f}  budget={cfg.budget_max_actions}"
        f"  force_tile={'on' if force_tile else 'off'}  mock={'on' if use_mock else 'off'}",
        f"TARGET CONCEPT: '{target}'   (raw label: '{raw_prompt}')",
        "=" * 78,
    ]
    try:
        result, ctx = _run_episode(image_pil, cfg, policy, oracle, query_set, use_mock, force_tile)
        graph = result["graph"]
        counts = result["counts"]
    except Exception as exc:
        root.removeHandler(handler)
        root.setLevel(prev_level)
        detail = "\n".join(header + ["", "[PIPELINE EXCEPTION]", str(exc), "",
                                     "--- captured log ---"] + handler.records)
        return image_pil, banner, f"Pipeline error: {exc}", detail
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    pred = counts["N_obs"]

    # Per-step action log (structured, from the episode).
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
        f"   (headline = N_obs)   GT={gt}",
        f"  classification tally: {_classification_tally(graph)}",
        f"  actions used: {len(result['log'])}   normalized cost: {ctx.cost.total(cfg):.3f}",
        f"  cost breakdown: {ctx.cost.as_dict()}",
    ]

    verbose = "\n".join(header + ["", "--- CAPTURED LOG (chronological) ---"]
                        + handler.records + step_lines + footer)

    canvas = draw_boxes(image_pil, graph, target, gt, pred)
    if gt is not None and pred == gt:
        status = f"EXACT MATCH  |  N_obs={pred} == GT={gt}"
    else:
        direction = "under" if (gt is not None and pred < gt) else "over"
        status = f"MISMATCH ({direction})  |  N_obs={pred} vs GT={gt}   (N_supp={counts['N_supp']}, N_cons={counts['N_cons']})"
    return canvas, banner, status, verbose


# ==========================================
# 5. Navigation
# ==========================================
def load_sample_view(dataset, idx):
    """Load an image + GT without running the pipeline (pipeline is on-demand)."""
    n = total_items(dataset)
    if n == 0:
        placeholder = Image.new("RGB", (640, 400), "#c0392b")
        return (placeholder, f"### [{dataset}] empty / unavailable", "", "",
                "Dataset unavailable (check paths / network / `datasets` install).", 0)
    idx = max(0, min(int(idx), n - 1))
    image_pil, raw_prompt, gt, err = get_sample(dataset, idx)
    banner = f"### [{dataset}] index {idx} / {n - 1} | Ground Truth: {gt}"
    default_prompt = default_prompt_for(dataset, raw_prompt)
    if err is not None:
        return (Image.new("RGB", (640, 400), "#c0392b"), banner, raw_prompt or "",
                default_prompt, f"skip: {err}", idx)
    return image_pil, banner, raw_prompt or "", default_prompt, "Loaded. Press ▶ Run Orchestration.", idx


def nav_next(dataset, idx):
    return load_sample_view(dataset, int(idx) + 1)


def nav_prev(dataset, idx):
    return load_sample_view(dataset, int(idx) - 1)


def nav_jump(dataset, idx_text):
    idx = int(idx_text) if str(idx_text).strip().lstrip("-").isdigit() else 0
    return load_sample_view(dataset, idx)


def on_dataset_change(dataset):
    d = DATASET_UI_DEFAULTS[dataset]
    image, banner, raw, prompt, status, idx = load_sample_view(dataset, 0)
    return (image, banner, raw, prompt, status, idx,
            gr.update(value=d["conf"]), gr.update(value=d["overlap_mode"]),
            gr.update(value=d["gate_mode"]), gr.update(value=d["nms_iou_threshold"]),
            gr.update(value=d["nms_iom_threshold"]), gr.update(value=d["cross_pass_dedup_metric"]),
            gr.update(value=d["cross_pass_dedup_threshold"]))


# ==========================================
# 6. Gradio interface
# ==========================================
_QUERY_FILES = sorted(
    os.path.join("queries", f) for f in os.listdir("queries")
    if f.endswith(".json")
) if os.path.isdir("queries") else [Config().vip_query_file]

with gr.Blocks(theme=gr.themes.Soft(), title="SAM3 Orchestration Dashboard") as app:
    idx_state = gr.State(value=0)

    gr.Markdown("# 🎛️ SAM3 Active-Perception Orchestration Dashboard")
    gr.Markdown(
        "Runs the **full agent orchestration** (policy → action → belief update) on a "
        "benchmark image, with every action, per-pass SAM3 stat, verifier call, and "
        "count estimate streamed to the verbose window. Navigation only loads the "
        "image; press **Run Orchestration** to execute the pipeline."
    )
    if _LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **SAM3 not loaded:** `{_LOAD_ERROR}` — runs will report this.")

    with gr.Row():
        with gr.Column(scale=3):
            image_display = gr.Image(label="Pipeline prediction view", type="pil", interactive=False)
            verbose_box = gr.Textbox(
                label="🔬 Verbose orchestration log (actions · per-pass SAM3 · verifier · estimates)",
                interactive=False, lines=26, max_lines=26,
            )

        with gr.Column(scale=2):
            banner_md = gr.Markdown("### Initializing…")
            status_box = gr.Textbox(label="📊 Result", interactive=False)

            dataset_dropdown = gr.Dropdown(
                choices=["countbench", "carpk", "pixmo"], value="countbench", label="📂 Dataset")

            with gr.Row():
                policy_dropdown = gr.Dropdown(
                    choices=["heuristic", "vlm"], value="heuristic", label="🧠 Policy")
                verifier_radio = gr.Radio(
                    choices=["off", "ioc", "vip"], value="ioc", label="✅ Verification")

            with gr.Row():
                overlap_radio = gr.Radio(
                    choices=["box", "mask"], value="box",
                    label="📐 Overlap metric (geometry: boxes vs instance masks)")
                conf_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.35, step=0.05, label="🎚️ Detection confidence")

            gr.Markdown("**Intra-pass NMS** (dedupes multiple detections from *one* SAM3 call)")
            gate_mode_radio = gr.Radio(
                choices=["dual", "iou_only", "iom_only"], value="dual",
                label="🎯 NMS suppression gate (dual = IoU+IoM, matches the validated reference)")

            with gr.Row():
                iou_threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.95, step=0.05,
                    label="Gate A: IoU threshold")
                iom_threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.95, step=0.05,
                    label="Gate B: IoM threshold")

            gr.Markdown("**Cross-pass dedup** (is this box the same object as one already "
                       "registered from an earlier pass/tile? the more consequential knob)")
            with gr.Row():
                dedup_metric_radio = gr.Radio(
                    choices=["iou", "iom"], value="iou",
                    label="Metric (iom = validated CARPK choice for dense uniform-size grids)")
                dedup_threshold_slider = gr.Slider(
                    minimum=0.10, maximum=0.95, value=0.60, step=0.05,
                    label="Threshold")

            budget_number = gr.Number(value=12, precision=0, label="🔁 Max actions (budget)")
            force_tile_checkbox = gr.Checkbox(
                value=False,
                label="🔲 Force at least one tile pass (quadrant split, before the policy loop)")

            with gr.Accordion("FM+V-IP (vip) options", open=False):
                query_file_dropdown = gr.Dropdown(
                    choices=_QUERY_FILES, value=_QUERY_FILES[0], label="Query set (vip)")
                mock_checkbox = gr.Checkbox(
                    value=False, label="🧪 Mock backend (offline: MockOracle + stubbed VLM)")
                mock_class_dropdown = gr.Dropdown(
                    choices=["target", "distractor", "spurious"], value="target",
                    label="MockOracle assumed true class (UI testing only)")

            raw_prompt_display = gr.Textbox(label="Raw dataset label / caption", interactive=False, lines=2)
            prompt_input = gr.Textbox(label="✏️ Target concept (drives SAM3 + agent)",
                                      placeholder="e.g. green fruit, car, apple…")

            run_button = gr.Button("▶ Run Orchestration", variant="primary")

            gr.HTML("<hr>")
            with gr.Row():
                btn_prev = gr.Button("⬅️ Prev")
                btn_next = gr.Button("Next ➡️")
            with gr.Row():
                jump_input = gr.Textbox(label="Jump to index", placeholder="e.g. 42", scale=2)
                btn_jump = gr.Button("🎯 Jump", scale=1)

    nav_outputs = [image_display, banner_md, raw_prompt_display, prompt_input, status_box, idx_state]
    run_inputs = [dataset_dropdown, idx_state, prompt_input, policy_dropdown, verifier_radio,
                  overlap_radio, gate_mode_radio, iou_threshold_slider, iom_threshold_slider,
                  dedup_metric_radio, dedup_threshold_slider, conf_slider, budget_number,
                  force_tile_checkbox, mock_checkbox, query_file_dropdown, mock_class_dropdown]
    run_outputs = [image_display, banner_md, status_box, verbose_box]

    app.load(fn=load_sample_view, inputs=[dataset_dropdown, idx_state], outputs=nav_outputs)
    dataset_dropdown.change(fn=on_dataset_change, inputs=[dataset_dropdown],
                            outputs=nav_outputs + [conf_slider, overlap_radio, gate_mode_radio,
                                                   iou_threshold_slider, iom_threshold_slider,
                                                   dedup_metric_radio, dedup_threshold_slider])

    run_button.click(fn=run_orchestration, inputs=run_inputs, outputs=run_outputs)
    prompt_input.submit(fn=run_orchestration, inputs=run_inputs, outputs=run_outputs)

    btn_next.click(fn=nav_next, inputs=[dataset_dropdown, idx_state], outputs=nav_outputs)
    btn_prev.click(fn=nav_prev, inputs=[dataset_dropdown, idx_state], outputs=nav_outputs)
    btn_jump.click(fn=nav_jump, inputs=[dataset_dropdown, jump_input], outputs=nav_outputs)


def main():
    logger.info("Launching orchestration dashboard on %s", device)
    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()

"""
config.py

Single source of truth for tunables. No magic numbers elsewhere in the
codebase -- new parameters are added here, grouped by concern, and read
from a Config instance passed down through the call stack.
"""

import dataclasses
import json
import os


@dataclasses.dataclass
class Config:
    # --- paths ---
    sam3_repo: str = "/home/ekielar/sam3"
    queries_dir: str = "queries"
    output_dir: str = "out"

    # --- detection ---
    conf: float = 0.35
    nms_mode: str = "dualgate"
    gate_mode: str = "dual"  # "dual" (default, unchanged) | "iou_only" | "iom_only": which
    #     apply_nms_dualgate suppression gate(s) apply. "iou_only" is pure lateral-duplicate
    #     IoU suppression (no containment check) -- e.g. countbench-style scenes. "iom_only"
    #     is pure containment suppression with NO size-ratio guard (no lateral-IoU check
    #     either) -- e.g. CARPK's dense grids of uniform-size objects, where IoU can
    #     over-suppress adjacent-but-distinct boxes, and a fully-contained box is always a
    #     duplicate regardless of its size relative to the container. Only takes effect
    #     when nms_mode == "dualgate".
    nms_iou_threshold: float = 0.40  # Gate A (IoU) threshold for apply_nms_dualgate.
    #     Named with an "nms_" prefix to stay unambiguous from the unrelated cross-pass
    #     dedup iou_threshold in pipeline.register_and_verify_candidates (still
    #     hardcoded at 0.40, a different mechanism). Ignored when nms_mode=="iou".
    nms_iom_threshold: float = 0.90  # Gate B (IoM containment) threshold for
    #     apply_nms_dualgate. Ignored when nms_mode=="iou".
    use_canopy_roi: bool = True  # False skips the "tree canopy" SAM3 sweep in
    #     pipeline.initialize_canopy_roi and anchors the ROI to the full frame instead --
    #     for datasets with no canopy concept (CARPK, CountBench, PixMo), where that sweep
    #     wastes a SAM3 call on a prompt that can never legitimately match.
    target_prompt: str = "green fruit"  # concept string for agent Query/TileQuery actions
    overlap_mode: str = "box"  # "box" (default) | "mask": NMS IoU/IoM + cross-pass dedup
    #     measured on instance masks instead of boxes (global passes only; tiled
    #     passes fall back to box overlap with a warning).

    # --- verifier ---
    verifier_mode: str = "ioc"  # "ioc" (default, unchanged) | "vip" (opt-in) | "off" (disabled)
    vip_query_file: str = "queries/green_citrus.json"
    vip_epsilon: float = None  # None -> use the query set's epsilon; a float overrides it
    vip_stop: float = 0.10
    vip_max_queries: int = 10
    crop_scale: float = 1.4
    crop_size: int = 256
    answer_mode: str = "batched"  # "batched" | "sequential"

    # --- oracle (Qwen-3-VL via OpenAI-compatible endpoint) ---
    oracle_base_url: str = dataclasses.field(
        default_factory=lambda: os.environ.get("QWEN_BASE_URL", "")
    )
    oracle_model_name: str = dataclasses.field(
        default_factory=lambda: os.environ.get("QWEN_MODEL", "")
    )
    oracle_temperature: float = 0.0
    oracle_max_retries: int = 2

    # --- agent ---
    window_m: int = 3
    delta_disc: float = 1.0
    delta_U: float = 0.5
    lambdas: dict = dataclasses.field(
        default_factory=lambda: {
            "lambda_D": 1.0,
            "lambda_S": 1.0,
            "lambda_k": 1.0,
            "lambda_delta": 1.0,
            "lambda_A": 1.0,
            "alpha_delta": 1.0,
            "alpha_s": 1.0,
            "lambda_C": 1.0,
            "lambda_U": 1.0,
        }
    )
    budget_max_actions: int = 12
    c0: float = 1.0            # base orchestration cost in the VoI-per-cost ratio
    small_area: float = 1024.0 # median candidate area (px^2) below which tiling is boosted
    tau_w: float = 0.5         # support-score threshold: low-w verification / N_supp
    k_min: int = 2             # min support k for the N_cons estimator
    tau_high: float = 0.5      # high-confidence s_bar threshold for N_cons
    area_min: float = 0.0      # plausible candidate area bounds (px^2) for the
    area_max: float = float("inf")  # lambda_A term of support_score; defaults are inert

    # --- guided-ROI policy (phase 7) ---
    roi_margin: float = 0.10      # fraction of a proposed ROI's size added on each
                                  # side before sensing (agent LookROIA)
    roi_min_size: float = 32.0    # min ROI side (px) worth sensing; smaller -> no-op
    roi_max_depth: int = 2        # max quadrant-equiv zoom levels; the ROI area floor
                                  # is frame_area / 4**roi_max_depth (16-quadrant cap)
    roi_dup_iou: float = 0.7      # a proposed ROI overlapping a sensed one above this
                                  # IoU is a no-op (avoids re-sensing)
    policy_enable_thinking: bool = False  # qwen3-vl "thinking" on the policy-loop
                                  # (look/tile/verify/stop) call; OFF by default since
                                  # those decisions are structured. inspect + the verify
                                  # oracle keep thinking on (their default call). One
                                  # model id throughout -- never switch models.

    # --- costs (normalized relative to one global SAM3 call) ---
    c_sam: float = 1.0
    c_tile: float = 0.25
    c_verify: float = 0.5
    c_inspect: float = 0.5
    c_orch: float = 0.05

    @classmethod
    def load(cls, path: str) -> "Config":
        """Build a Config from defaults overridden by a JSON file of field values."""
        with open(path, "r") as f:
            overrides = json.load(f)
        return dataclasses.replace(cls(), **overrides)


def thinking_call_kwargs(enable_thinking: bool) -> dict:
    """Extra kwargs for an OpenAI-compatible chat.completions.create() call that
    toggle qwen3-vl "thinking" WITHOUT switching models.

        enable_thinking=True  -> {} (the model's default; thinking on)
        enable_thinking=False -> the disable-thinking argument

    NOTE: the exact key is serving-stack specific. This uses the vLLM/Qwen
    chat-template convention (extra_body.chat_template_kwargs.enable_thinking).
    VERIFY it against the live Ollama qwen3-vl endpoint -- Ollama may instead want
    a top-level {"think": false}. If so, change only this function.
    """
    if enable_thinking:
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

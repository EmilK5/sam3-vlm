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

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

    # --- verifier ---
    verifier_mode: str = "ioc"  # "ioc" (default, unchanged) | "vip" (opt-in)
    vip_query_file: str = "queries/green_citrus.json"
    vip_epsilon: float = 0.15
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

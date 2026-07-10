"""
scripts/demo_verify.py

Human-run demo of the FM+V-IP verifier: verify one or more hand-picked boxes on
an image and pretty-print each interpretable query->answer chain and verdict.

This is the GO/NO-GO tool: with --oracle qwen on real crops, the printed chains
should read like sensible visual reasoning. Tweak queries/epsilon in the JSON
and re-run until they do.
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from config import Config
from verifier.queries import load_query_set
from verifier.oracle import MockOracle, QwenOracle, RouterOracle
from verifier.verify import verify_candidate

logger = logging.getLogger(__name__)


def parse_box(text: str):
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"box must be 'x1,y1,x2,y2', got {text!r}")
    return parts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo the FM+V-IP verifier on hand-picked boxes.")
    parser.add_argument("--image", required=True, help="Path to the image.")
    parser.add_argument("--boxes", required=True, action="append", type=parse_box,
                        metavar="x1,y1,x2,y2", help="A box (repeatable).")
    parser.add_argument("--oracle", choices=["mock", "qwen", "router"], default="qwen",
                        help="Which oracle answers the queries. 'router' answers "
                             "cv-routed queries locally and sends only the residual "
                             "to Qwen (<=1 Qwen call per candidate); SAM3 queries "
                             "get 0 here since no processor is loaded in this demo.")
    parser.add_argument("--mock-class", default="target",
                        help="true_class for --oracle mock (offline demo).")
    parser.add_argument("--query-file", default=None,
                        help="Query set JSON (defaults to cfg.vip_query_file).")
    return parser.parse_args(argv)


def format_chain(result) -> str:
    cells = "".join(f"[{c['q'][:28]} {c['a']}]" for c in result["chain"])
    # show the winning class's posterior mass (verdict == argmax of posterior)
    p_verdict = max(result["posterior"])
    return f"{cells} -> {result['verdict']} ({p_verdict:.2f})"


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    cfg = Config()
    query_file = args.query_file or cfg.vip_query_file
    query_set = load_query_set(query_file)

    if args.oracle == "mock":
        oracle = MockOracle(args.mock_class)
    elif args.oracle == "router":
        # No SAM3 processor is loaded in the demo, so sam3-routed queries answer 0.
        oracle = RouterOracle(cfg, processor=None, vlm_oracle=QwenOracle(cfg))
    else:
        oracle = QwenOracle(cfg)

    image_np = np.array(Image.open(args.image).convert("RGB"))

    for box in args.boxes:
        result = verify_candidate(image_np, box, oracle, query_set, cfg)
        print(f"box {tuple(box)}  (calls={result['n_oracle_calls']})")
        print("  " + format_chain(result))
        for cls, p in zip(query_set.classes, result["posterior"]):
            print(f"    P({cls}) = {p:.3f}")
        if isinstance(oracle, RouterOracle):
            print(f"    router: {oracle.n_vlm_calls} Qwen call(s), "
                  f"{oracle.n_sam_calls} SAM3 call(s) on the last crop")


if __name__ == "__main__":
    main()

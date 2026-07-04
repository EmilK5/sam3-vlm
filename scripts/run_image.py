"""
scripts/run_image.py

CLI: run N passes of the existing SAM3 cascade on one image, reusing a
single OrchardGraph across passes, and dump the overlay image and the
graph JSON. Makes no changes to pipeline.execute_pass.
"""

import argparse
import dataclasses
import json
import logging
import os
import sys

# Make the repo root importable when run as `python scripts/run_image.py`
# (running a script file puts scripts/ on sys.path, not the repo root).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

_cfg = Config()
if _cfg.sam3_repo not in sys.path:
    sys.path.append(_cfg.sam3_repo)

from verifier.queries import load_query_set
from verifier.oracle import MockOracle, QwenOracle

from PIL import Image
import torch

import graph as graph_module
import inference
import pipeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run N passes of the SAM3 orchard cascade on one image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="Text prompt, e.g. 'green fruit'.")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes to run.")
    parser.add_argument("--tiling", action="store_true", help="Enable tiled inference.")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE enhancement.")
    parser.add_argument("--conf", type=float, default=_cfg.conf, help="Confidence threshold.")
    parser.add_argument("--verifier", choices=["ioc", "vip", "off"], default=_cfg.verifier_mode,
                        help="Candidate verifier: ioc (default), vip (FM+V-IP), or off (disabled).")
    parser.add_argument("--oracle", choices=["mock", "qwen"], default="qwen",
                        help="Oracle for --verifier vip (ignored otherwise).")
    parser.add_argument("--query-file", default=None,
                        help="Query set JSON for --verifier vip (defaults to cfg.vip_query_file).")
    parser.add_argument("--mock-class", default="target",
                        help="true_class for --oracle mock (offline vip runs).")
    return parser.parse_args(argv)


def build_verifier(args):
    """Return (cfg, oracle, query_set) for the requested verifier mode.

    oracle/query_set are None unless --verifier vip is selected.
    """
    cfg = dataclasses.replace(_cfg, verifier_mode=args.verifier)
    if args.verifier != "vip":
        return cfg, None, None

    query_set = load_query_set(args.query_file or cfg.vip_query_file)
    oracle = MockOracle(args.mock_class) if args.oracle == "mock" else QwenOracle(cfg)
    return cfg, oracle, query_set


def build_output_paths(image_path: str, output_dir: str, suffix: str = "") -> tuple:
    """Returns (overlay_path, graph_path) for a given input image.

    `suffix` (e.g. the verifier mode) is inserted into the filename so runs in
    different modes on the same image don't overwrite each other.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    tag = f"_{suffix}" if suffix else ""
    overlay_path = os.path.join(output_dir, f"{stem}{tag}_overlay.jpg")
    graph_path = os.path.join(output_dir, f"{stem}{tag}_graph.json")
    return overlay_path, graph_path


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bpe_path = f"{_cfg.sam3_repo}/assets/bpe_simple_vocab_16e6.txt.gz"
    _, processor = inference.load_sam3_model(bpe_path, args.conf, device=device)

    cfg, oracle, query_set = build_verifier(args)
    logging.info(f"Verifier mode: {cfg.verifier_mode}")

    image_pil = Image.open(args.image).convert("RGB")
    orchard_graph = graph_module.OrchardGraph()

    for pass_number in range(1, args.passes + 1):
        stats = pipeline.execute_pass(
            processor=processor,
            image_pil=image_pil,
            graph=orchard_graph,
            conf=args.conf,
            clahe=args.clahe,
            tiling=args.tiling,
            pass_number=pass_number,
            prompt=args.prompt,
            cfg=cfg,
            oracle=oracle,
            query_set=query_set,
        )
        logging.info(f"Pass {pass_number}: {stats.as_row()}")

    os.makedirs(_cfg.output_dir, exist_ok=True)
    overlay_path, graph_path = build_output_paths(args.image, _cfg.output_dir, suffix=cfg.verifier_mode)

    inference.plot_graph_scene(image_pil, orchard_graph, output_path=overlay_path)
    with open(graph_path, "w") as f:
        json.dump(orchard_graph.to_dict(), f, indent=2)

    logging.info(f"Saved overlay to {overlay_path} and graph to {graph_path}")


if __name__ == "__main__":
    main()

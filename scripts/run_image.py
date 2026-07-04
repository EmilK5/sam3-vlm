"""
scripts/run_image.py

CLI: run N passes of the existing SAM3 cascade on one image, reusing a
single OrchardGraph across passes, and dump the overlay image and the
graph JSON. Makes no changes to pipeline.execute_pass.
"""

import argparse
import json
import logging
import os
import sys

from config import Config

_cfg = Config()
if _cfg.sam3_repo not in sys.path:
    sys.path.append(_cfg.sam3_repo)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    return parser.parse_args(argv)


def build_output_paths(image_path: str, output_dir: str) -> tuple:
    """Returns (overlay_path, graph_path) for a given input image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.jpg")
    graph_path = os.path.join(output_dir, f"{stem}_graph.json")
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
        )
        logging.info(f"Pass {pass_number}: {stats.as_row()}")

    os.makedirs(_cfg.output_dir, exist_ok=True)
    overlay_path, graph_path = build_output_paths(args.image, _cfg.output_dir)

    inference.plot_graph_scene(image_pil, orchard_graph, output_path=overlay_path)
    with open(graph_path, "w") as f:
        json.dump(orchard_graph.to_dict(), f, indent=2)

    logging.info(f"Saved overlay to {overlay_path} and graph to {graph_path}")


if __name__ == "__main__":
    main()

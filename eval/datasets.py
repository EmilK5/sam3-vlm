"""
eval/datasets.py

Ground-truth dataset loaders for evaluation. No labeled data is used before
evaluation; these loaders exist only to score the zero-shot system.

load_split(root, fmt, split) -> list of samples, each a dict:
    {"image_path": str, "gt_boxes": np.ndarray (N,4) float xyxy, "count": int}

Coordinate frame: gt_boxes are global-frame xyxy pixels.

Expected dataset layout (a per-split subdirectory under each kind):
    root/
      images/<split>/*.png|jpg
      labels/<split>/*.txt         (yolo)
      masks/<split>/*.png          (minneapple)
where <split> is one of train / val / test. If the <split> subdirectory is
absent, the loader falls back to the flat root/images (+ labels/masks) layout.

Supported formats:
    "yolo"       - per-image <stem>.txt with rows "cls xc yc w h" (normalized),
                   converted with inference.yolo_to_xyxy.
    "minneapple" - per-image instance mask PNG where each object has a distinct
                   pixel value/color; one box per instance id.
"""

import argparse
import os

import numpy as np
from PIL import Image

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------- YOLO -----------------------

def parse_yolo_labels(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Parse one YOLO label file into an (N,4) float xyxy array (pixels).

    Missing file or empty file -> shape (0,4). Uses inference.yolo_to_xyxy
    (imported lazily so this module stays importable without torch).
    """
    from inference import yolo_to_xyxy  # lazy: inference imports torch/transformers

    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _cls, xc, yc, w, h = parts[:5]
                boxes.append(yolo_to_xyxy(float(xc), float(yc), float(w), float(h), img_w, img_h))
    return np.array(boxes, dtype=float).reshape(-1, 4)


def _split_dir(root: str, kind: str, split: str) -> str:
    """Locate root/<kind>/<split>, falling back to the flat root/<kind>.

    Raises FileNotFoundError (with a helpful message) if neither exists.
    """
    with_split = os.path.join(root, kind, split)
    if os.path.isdir(with_split):
        return with_split
    flat = os.path.join(root, kind)
    if os.path.isdir(flat):
        return flat
    raise FileNotFoundError(
        f"Could not find '{kind}' for split '{split}': tried {with_split} and {flat}. "
        f"Expected layout root/{kind}/<split>/."
    )


def _list_images(image_dir: str):
    names = sorted(os.listdir(image_dir))
    return [n for n in names if os.path.splitext(n)[1].lower() in IMAGE_EXTS]


def _load_yolo(root: str, split: str) -> list:
    image_dir = _split_dir(root, "images", split)
    label_dir = _split_dir(root, "labels", split)
    samples = []
    for fname in _list_images(image_dir):
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        img_w, img_h = Image.open(image_path).size
        boxes = parse_yolo_labels(os.path.join(label_dir, stem + ".txt"), img_w, img_h)
        samples.append({"image_path": image_path, "gt_boxes": boxes, "count": int(len(boxes))})
    return samples


# ----------------------- MinneApple (instance masks) -----------------------

def masks_to_boxes(mask_path: str) -> np.ndarray:
    """Bounding box (xyxy pixels, half-open max) of each instance in a mask PNG.

    Grayscale mask: each nonzero pixel value is one instance id.
    RGB mask: each distinct non-black color is one instance.
    """
    mask = np.array(Image.open(mask_path))
    boxes = []

    if mask.ndim == 2:
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            ys, xs = np.where(mask == inst_id)
            if xs.size:
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    else:  # color-encoded instances
        flat = mask.reshape(-1, mask.shape[2])
        for color in np.unique(flat, axis=0):
            if np.all(color == 0):
                continue
            member = np.all(mask == color, axis=2)
            ys, xs = np.where(member)
            if xs.size:
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

    return np.array(boxes, dtype=float).reshape(-1, 4)


def _load_minneapple(root: str, split: str) -> list:
    image_dir = _split_dir(root, "images", split)
    mask_dir = _split_dir(root, "masks", split)
    samples = []
    for fname in _list_images(image_dir):
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, stem + ".png")
        boxes = masks_to_boxes(mask_path) if os.path.exists(mask_path) else np.empty((0, 4))
        samples.append({"image_path": image_path, "gt_boxes": boxes, "count": int(len(boxes))})
    return samples


# ----------------------- public API -----------------------

def load_split(root: str, fmt: str, split: str = "train") -> list:
    """Load a dataset split. fmt in {"yolo", "minneapple"}; split in train/val/test."""
    if fmt == "yolo":
        return _load_yolo(root, split)
    if fmt == "minneapple":
        return _load_minneapple(root, split)
    raise ValueError(f"Unknown dataset fmt {fmt!r}; expected 'yolo' or 'minneapple'.")


def draw_gt_overlay(image_pil, gt_boxes, output_path: str):
    """Draw ground-truth boxes (cyan) on the image and save to output_path.

    Uses a headless Agg figure so no display / global backend is required.
    """
    fig = Figure(figsize=(12, 12))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.imshow(image_pil)
    for x1, y1, x2, y2 in gt_boxes:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor="cyan", facecolor="none"))
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", dpi=150)


# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect a ground-truth dataset split.")
    parser.add_argument("--root", required=True, help="Dataset root (contains images/ and labels/).")
    parser.add_argument("--fmt", choices=["yolo", "minneapple"], required=True)
    parser.add_argument("--split", default="train", help="Split subdir: train / val / test.")
    parser.add_argument("--out", default="out/gt_overlay.jpg", help="Where to save one GT overlay.")
    args = parser.parse_args()

    samples = load_split(args.root, args.fmt, args.split)
    print(f"Dataset '{args.split}' size: {len(samples)} images")
    for s in samples[:5]:
        print(f"  {os.path.basename(s['image_path'])}: {s['count']} objects")

    if samples:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        first = samples[0]
        draw_gt_overlay(Image.open(first["image_path"]).convert("RGB"), first["gt_boxes"], args.out)
        print(f"Saved GT overlay for {os.path.basename(first['image_path'])} to {args.out}")


if __name__ == "__main__":
    main()

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

Also provides a second, unrelated loader family for count-only benchmark
datasets (pixmo / countbench / carpk) that have no box-level ground truth, only
a scalar count per image -- see get_count_sample() below.
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


# ----------------------- count-only benchmark datasets -----------------------
#
# pixmo / countbench / carpk have no box-level ground truth, only a scalar count
# per image (unlike load_split's yolo/minneapple, which have real boxes). Ported
# from orchestration_app.py's dataset access layer so the dashboard and eval CLI
# share one implementation. get_count_sample() is the uniform accessor:
#     get_count_sample(name, idx) -> (image_pil_or_None, raw_prompt, gt_count, error_or_None)
# Network/optional-dependency imports (datasets, requests) stay lazy so this
# module keeps importing cleanly without them installed.

COUNT_DATASETS = ("pixmo", "countbench", "carpk")

# Local CARPK devkit root (override via env var to match wherever it lives).
CARPK_BASE_DIR = os.environ.get("CARPK_BASE_DIR", "./datasets/CARPK_devkit/data")
CARPK_IMAGE_DIR = os.path.join(CARPK_BASE_DIR, "Images")
CARPK_ANNO_DIR = os.path.join(CARPK_BASE_DIR, "Annotations")
CARPK_SPLIT_FILE = os.path.join(CARPK_BASE_DIR, "ImageSets", "test.txt")

# Optional label->noun-phrase maps. Absent => raw label.
PROMPT_MAP_FILES = {
    "pixmo": "./pixmo_count_generic_map.json",
    "countbench": "./countbench_generic_map.json",
    "carpk": None,
}

# CountBench remote-payload / ground-truth corrections (from the reference eval).
PURGED_IMAGE_IDS = {"pixmo": set(), "countbench": {126, 406, 174, 281}, "carpk": set()}
GROUND_TRUTH_OVERRIDES = {
    "pixmo": {},
    "countbench": {134: 9, 271: 8, 331: 1, 194: 25},
    "carpk": {},
}

_count_dataset_cache = {}
_prompt_maps = {}


def _parse_carpk():
    import logging
    if not os.path.exists(CARPK_SPLIT_FILE):
        logging.getLogger(__name__).error("CARPK split file missing: %s", CARPK_SPLIT_FILE)
        return []
    with open(CARPK_SPLIT_FILE, "r", encoding="utf-8") as f:
        stems = [line.strip() for line in f if line.strip()]
    parsed = []
    for stem in stems:
        anno_path = os.path.join(CARPK_ANNO_DIR, f"{stem}.txt")
        gt = 0
        if os.path.exists(anno_path):
            with open(anno_path, "r", encoding="utf-8") as af:
                gt = sum(1 for line in af if line.strip())
        parsed.append({"image_path": os.path.join(CARPK_IMAGE_DIR, f"{stem}.png"), "count": gt})
    return parsed


def get_count_dataset(name):
    """Load (and cache) the raw dataset object/list for a count-only dataset."""
    if name not in _count_dataset_cache:
        if name == "countbench":
            from datasets import load_dataset  # lazy: optional dependency
            _count_dataset_cache[name] = load_dataset("nielsr/countbench", split="train")
        elif name == "pixmo":
            from datasets import load_dataset  # lazy: optional dependency
            _count_dataset_cache[name] = load_dataset("allenai/pixmo-count", split="test")
        elif name == "carpk":
            _count_dataset_cache[name] = _parse_carpk()
        else:
            raise ValueError(f"Unknown count dataset {name!r}; expected one of {COUNT_DATASETS}.")
    return _count_dataset_cache[name]


def count_dataset_size(name):
    import logging
    try:
        return len(get_count_dataset(name))
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to load dataset %s: %s", name, exc)
        return 0


def get_prompt_map(name):
    if name not in _prompt_maps:
        path = PROMPT_MAP_FILES.get(name)
        if path and os.path.exists(path):
            import json
            with open(path, "r", encoding="utf-8") as f:
                _prompt_maps[name] = json.load(f)
        else:
            _prompt_maps[name] = {}
    return _prompt_maps[name]


def default_prompt_for(name, raw_prompt):
    if name == "carpk":
        return "car"
    pm = get_prompt_map(name)
    if name == "pixmo":
        return pm.get(raw_prompt, (raw_prompt or "").lower())
    return pm.get(raw_prompt, raw_prompt or "")  # countbench keeps native casing


def get_count_sample(name, idx):
    """Uniform accessor -> (image_pil_or_None, raw_prompt, gt_count, error_or_None).

    Applies the dataset's ground-truth overrides; purged ids are surfaced as errors.
    """
    data = get_count_dataset(name)
    if idx in PURGED_IMAGE_IDS.get(name, set()):
        return None, "", None, f"index {idx} is purged (dead payload) for {name}"

    override = GROUND_TRUTH_OVERRIDES.get(name, {}).get(idx)
    sample = data[idx]

    if name == "countbench":
        raw_prompt = sample["text"]
        gt = override if override is not None else int(sample["number"])
        if sample.get("image") is None:
            return None, raw_prompt, gt, "image link expired/corrupted"
        return sample["image"].convert("RGB"), raw_prompt, gt, None

    if name == "pixmo":
        import io
        import requests  # lazy: only pixmo fetches by URL
        raw_prompt = sample["label"]
        gt = override if override is not None else int(sample["count"])
        url = sample.get("image_url")
        if not url:
            return None, raw_prompt, gt, "missing image_url"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return None, raw_prompt, gt, f"HTTP status {r.status_code}"
            return Image.open(io.BytesIO(r.content)).convert("RGB"), raw_prompt, gt, None
        except Exception as exc:
            return None, raw_prompt, gt, f"link timeout/unresponsive ({exc})"

    if name == "carpk":
        gt = override if override is not None else int(sample["count"])
        path = sample["image_path"]
        if not os.path.exists(path):
            return None, "car", gt, f"local file path missing: '{path}'"
        return Image.open(path).convert("RGB"), "car", gt, None

    raise ValueError(f"Unknown count dataset {name!r}; expected one of {COUNT_DATASETS}.")


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

"""
Tests for eval/datasets.py.

The YOLO path calls inference.yolo_to_xyxy; inference imports torch/transformers
which aren't installed here, so those are stubbed in sys.modules so the REAL
yolo_to_xyxy is exercised. The MinneApple and overlay paths need no stubbing.
"""

import sys
import types

import numpy as np
import pytest
from PIL import Image

from eval import datasets


@pytest.fixture
def stub_heavy_deps():
    saved = {n: sys.modules.get(n) for n in ("torch", "transformers", "inference")}
    sys.modules["torch"] = types.ModuleType("torch")
    tf = types.ModuleType("transformers")
    tf.Sam3Model = object
    tf.Sam3Processor = object
    sys.modules["transformers"] = tf
    sys.modules.pop("inference", None)  # force a fresh import under the stubs
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _make_yolo_split(root, split="train", flat=False):
    """Two 100x100 images + YOLO labels in root/images/<split> + root/labels/<split>.

    a: 2 boxes, b: 1 box. If flat=True, use root/images + root/labels (no split subdir).
    """
    if flat:
        img_dir = root / "images"
        lbl_dir = root / "labels"
    else:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (100, 100)).save(img_dir / "a.png")
    Image.new("RGB", (100, 100)).save(img_dir / "b.png")
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.25 0.25 0.1 0.1\n")
    (lbl_dir / "b.txt").write_text("0 0.5 0.5 0.4 0.4\n")


def test_load_yolo_counts_and_coords(tmp_path, stub_heavy_deps):
    _make_yolo_split(tmp_path, split="train")
    samples = datasets.load_split(str(tmp_path), "yolo", split="train")

    assert [s["count"] for s in samples] == [2, 1]  # sorted a, b
    a = samples[0]
    assert a["gt_boxes"].shape == (2, 4)
    # center (50,50), wh (20,20) -> [40,40,60,60]
    assert np.allclose(a["gt_boxes"][0], [40, 40, 60, 60])
    assert np.allclose(a["gt_boxes"][1], [20, 20, 30, 30])
    assert np.allclose(samples[1]["gt_boxes"][0], [30, 30, 70, 70])


def test_load_yolo_selects_requested_split(tmp_path, stub_heavy_deps):
    _make_yolo_split(tmp_path, split="train")          # 2 images
    val_img = tmp_path / "images" / "val"
    val_lbl = tmp_path / "labels" / "val"
    val_img.mkdir(parents=True)
    val_lbl.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(val_img / "v.png")
    (val_lbl / "v.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    assert len(datasets.load_split(str(tmp_path), "yolo", split="train")) == 2
    val = datasets.load_split(str(tmp_path), "yolo", split="val")
    assert len(val) == 1 and val[0]["count"] == 1


def test_load_yolo_flat_layout_fallback(tmp_path, stub_heavy_deps):
    _make_yolo_split(tmp_path, flat=True)
    samples = datasets.load_split(str(tmp_path), "yolo", split="train")  # falls back to flat
    assert [s["count"] for s in samples] == [2, 1]


def test_load_yolo_missing_dir_raises(tmp_path, stub_heavy_deps):
    with pytest.raises(FileNotFoundError):
        datasets.load_split(str(tmp_path), "yolo", split="train")  # nothing created


def test_load_yolo_missing_label_is_empty(tmp_path, stub_heavy_deps):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (50, 50)).save(tmp_path / "images" / "train" / "c.png")  # no c.txt
    samples = datasets.load_split(str(tmp_path), "yolo", split="train")
    assert samples[0]["count"] == 0
    assert samples[0]["gt_boxes"].shape == (0, 4)


def test_load_minneapple_instance_masks(tmp_path):
    img_dir = tmp_path / "images" / "train"
    mask_dir = tmp_path / "masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(img_dir / "x.png")

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 30:40] = 1   # instance 1 -> xs[30,39], ys[10,19]
    mask[50:55, 60:70] = 2   # instance 2 -> xs[60,69], ys[50,54]
    Image.fromarray(mask, mode="L").save(mask_dir / "x.png")

    samples = datasets.load_split(str(tmp_path), "minneapple", split="train")
    assert samples[0]["count"] == 2
    boxes = samples[0]["gt_boxes"]
    # half-open max: [xmin, ymin, xmax+1, ymax+1]
    assert np.allclose(boxes[0], [30, 10, 40, 20])
    assert np.allclose(boxes[1], [60, 50, 70, 55])


def test_load_split_rejects_unknown_fmt(tmp_path):
    with pytest.raises(ValueError):
        datasets.load_split(str(tmp_path), "coco")


def test_draw_gt_overlay_writes_file(tmp_path):
    out = tmp_path / "gt.jpg"
    img = Image.new("RGB", (64, 64))
    datasets.draw_gt_overlay(img, np.array([[5, 5, 25, 25], [30, 30, 60, 60]], dtype=float), str(out))
    assert out.exists() and out.stat().st_size > 0

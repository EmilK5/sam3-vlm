"""Deterministic visualizations for saved experiment runs."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from eval.dataset_adapters import CanonicalSample
from provenance.contracts import GraphNodeSnapshotRecord, PassRecord


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _node_label(node, target_class: str) -> tuple[str, str]:
    declaration = getattr(node, "final_declaration", None)
    if declaration is None:
        declaration = getattr(node, "temporary_map_class", None)
    if declaration is None:
        declaration = getattr(node, "classification", "unresolved")
    posterior = getattr(node, "posterior", None)
    probability = None if posterior is None else posterior.get(target_class)
    label = str(declaration or "unresolved")
    if probability is not None:
        label += f" {float(probability):.2f}"
    color = "#20c55a" if declaration == target_class else "#e5484d"
    if declaration in {None, "unresolved"}:
        color = "#00a8e8"
    return label, color


def render_final_overlay(
    image: Image.Image,
    *,
    sample: CanonicalSample,
    nodes: Iterable,
    target_class: str,
    predictions: Mapping[str, object],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for box in sample.ground_truth_boxes:
        draw.rectangle(tuple(float(value) for value in box), outline="#f5c542", width=2)
    for node in nodes:
        box = tuple(float(value) for value in node.box)
        label, color = _node_label(node, target_class)
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 2, max(0, box[1] - 12)), label, fill=color, font=font)
    gt = sample.ground_truth_count
    hard = predictions.get("hard_count")
    soft = predictions.get("soft_count")
    banner = f"{sample.dataset_name}/{sample.split}/{sample.sample_key} | GT={gt} | hard={hard} | soft={soft}"
    draw.rectangle((0, 0, min(canvas.width, max(420, len(banner) * 7)), 22), fill="#111827")
    draw.text((6, 5), banner, fill="white", font=font)
    return canvas


def render_pass_timeline(
    image: Image.Image,
    passes: Sequence[PassRecord],
    *,
    target_class: str,
    max_columns: int = 4,
) -> Image.Image:
    if not passes:
        return image.convert("RGB").copy()
    panels = [_render_pass_panel(image, record, target_class) for record in passes]
    width, height = panels[0].size
    columns = max(1, min(max_columns, len(panels)))
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
    return sheet


def _render_pass_panel(image: Image.Image, record: PassRecord, target_class: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for node in record.graph_after:
        label, color = _node_label(node, target_class)
        box = tuple(float(value) for value in node.box)
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 2, max(0, box[1] - 11)), label, fill=color, font=font)
    text = f"pass {record.pass_index} | action={record.selected_action_id or '-'} | nodes={len(record.graph_after)}"
    draw.rectangle((0, 0, min(canvas.width, max(360, len(text) * 7)), 22), fill="#111827")
    draw.text((5, 5), text, fill="white", font=font)
    return canvas


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_run_image(run_directory: str | Path, relative_path: str) -> Image.Image:
    path = Path(run_directory) / relative_path
    with Image.open(path) as image:
        return image.convert("RGB")


__all__ = [
    "image_to_png_bytes",
    "load_run_image",
    "render_final_overlay",
    "render_pass_timeline",
]

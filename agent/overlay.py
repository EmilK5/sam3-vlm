"""
agent/overlay.py

render_overlay: draw the current belief state onto a copy of the frame so the v2
policy can SEE what has already been found and sensed. PIL ImageDraw only -- no
matplotlib, no model, no torch. All boxes are global-frame xyxy pixels.

Candidate boxes are colored by their graph classification; already-sensed ROIs
and the tree region-of-interest are drawn in their own distinct styles so the
VLM can tell "objects", "regions I already looked at", and "the region I'm
allowed to look inside" apart at a glance.
"""

from PIL import ImageDraw

# Candidate-box outline color per classification (RGB).
_CLASS_COLORS = {
    "fruit": (60, 200, 60),        # green
    "leaf": (210, 140, 40),        # orange
    "spurious": (220, 50, 50),     # red
    "unresolved": (170, 170, 170),  # gray
}
_SENSED_ROI_COLOR = (60, 120, 240)  # blue -- regions already sensed
_TREE_ROI_COLOR = (240, 60, 220)    # magenta -- the allowed sensing region


def _rect(draw, box, color, width):
    x1, y1, x2, y2 = (float(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def render_overlay(image_pil, graph, sensed_rois=None, tree_roi=None):
    """Return a copy of image_pil with the belief state drawn on it.

    graph        : OrchardGraph whose nodes' boxes are drawn, colored by class.
    sensed_rois  : iterable of xyxy ROIs already sensed (drawn blue). None -> none.
    tree_roi     : the xyxy tree region-of-interest (drawn magenta). None -> not drawn.

    Draw order is tree ROI, then sensed ROIs, then candidate boxes on top, so the
    detections stay the most legible layer.
    """
    canvas = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    if tree_roi is not None:
        _rect(draw, tree_roi, _TREE_ROI_COLOR, 3)

    for roi in (sensed_rois or []):
        _rect(draw, roi, _SENSED_ROI_COLOR, 2)

    for node in graph.nodes.values():
        color = _CLASS_COLORS.get(node.classification, _CLASS_COLORS["unresolved"])
        _rect(draw, node.box, color, 2)

    return canvas

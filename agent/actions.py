"""
agent/actions.py

The sensing action layer (v2, active-perception formulation): plain dataclasses
for the allowed actions and a single execute() dispatcher.

    QueryA(region, prompt, conf)   - SAM3 on one region (roi_override); the
                                     bootstrap/global sensing primitive
    LookROIA(region)               - ONE untiled SAM3 query on a policy-proposed
                                     ROI (sensing target only; +margin, guarded,
                                     never a node)
    StopA(estimate_name)           - terminate; names the count estimator to report

execute(action, ctx) -> int  (number of new candidate tracks the action created;
0 for Stop, and 0 for a guarded/no-op LookROIA). ctx (ActionContext) bundles the
sensing state. Verification is not an action: it runs automatically inside
execute_pass (cfg.verifier_mode).

Grounding invariant: no VLM-supplied box ever becomes a candidate. A LookROIA
region is a SENSING TARGET ONLY -- it is passed to execute_pass as roi_override to
restrict where SAM3 looks, and is never added to the graph. Candidates originate
only from SAM3 (via execute_pass); regions/ROIs only steer the sensor.
"""

import dataclasses
import logging

import numpy as np

from agent.budget import CostMeter


# ----------------------- action dataclasses -----------------------

@dataclasses.dataclass
class QueryA:
    region: tuple            # xyxy pixel region to query
    prompt: str
    conf: float


@dataclasses.dataclass
class LookROIA:
    region: tuple            # xyxy ROI the policy proposes to sense. A SENSING
                             # TARGET ONLY: expanded + sensed once (untiled) by
                             # execute(), never added to the graph as a candidate
                             # (grounding invariant).


@dataclasses.dataclass
class StopA:
    estimate_name: str       # which count estimator to report (e.g. "N_obs")


@dataclasses.dataclass
class ActionContext:
    """Bundle passed to execute(); the mutable episode sensing state."""
    processor: object = None
    image_pil: object = None
    graph: object = None
    cfg: object = None
    oracle: object = None
    query_set: object = None
    discovery: object = None
    partition: list = dataclasses.field(default_factory=list)
    cost: CostMeter = dataclasses.field(default_factory=CostMeter)
    n_passes: int = 0        # sensing passes executed (Query/Look only), so a
                             # Stop never inflates pass numbers
    sensed_rois: list = dataclasses.field(default_factory=list)  # expanded xyxy ROIs
                             # already sensed by LookROIA, for de-duplicating repeats


# ----------------------- helpers -----------------------

def _expand_and_clamp(region, margin, img_w, img_h) -> tuple:
    """Grow an xyxy region outward by `margin` (a fraction of its OWN width/height)
    on every side, then clamp to the image [0, 0, img_w, img_h]. Returns an int
    xyxy tuple in global-frame pixels."""
    x1, y1, x2, y2 = region
    w = x2 - x1
    h = y2 - y1
    ex1, ey1 = x1 - margin * w, y1 - margin * h
    ex2, ey2 = x2 + margin * w, y2 + margin * h
    cx1 = max(0, min(int(round(ex1)), img_w))
    cy1 = max(0, min(int(round(ey1)), img_h))
    cx2 = max(0, min(int(round(ex2)), img_w))
    cy2 = max(0, min(int(round(ey2)), img_h))
    return (cx1, cy1, cx2, cy2)


def _roi_iou(a, b) -> float:
    """Plain xyxy IoU (used only to de-duplicate sensed ROIs; not on any SAM3 path)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _next_pass_number(ctx) -> int:
    """execute_pass pass_number for the next sensing action in the episode.

    Counts only sensing passes (ctx.n_passes), not all episode actions, so a
    non-sensing action between queries does not skew pass-dependent behavior in
    the pipeline (leaf-map caching, exemplar feedback, the pass>=3 IoC guard).
    """
    return int(getattr(ctx, "n_passes", 0)) + 1


def _execute_query(action, ctx, tiling, roi_override) -> int:
    from pipeline import execute_pass  # lazy: pipeline imports torch/inference
    pass_number = _next_pass_number(ctx)
    # nms_mode/gate_mode/use_canopy_roi/nms thresholds read via getattr with
    # execute_pass's own defaults, so an episode whose cfg doesn't set them (or a
    # bare Config()) is unaffected.
    nms_mode = getattr(ctx.cfg, "nms_mode", "dualgate")
    gate_mode = getattr(ctx.cfg, "gate_mode", "dual")
    use_canopy_roi = getattr(ctx.cfg, "use_canopy_roi", True)
    nms_iou_threshold = getattr(ctx.cfg, "nms_iou_threshold", 0.40)
    nms_iom_threshold = getattr(ctx.cfg, "nms_iom_threshold", 0.90)
    cross_pass_dedup_metric = getattr(ctx.cfg, "cross_pass_dedup_metric", "iou")
    cross_pass_dedup_threshold = getattr(ctx.cfg, "cross_pass_dedup_threshold", 0.40)
    stats = execute_pass(
        processor=ctx.processor, image_pil=ctx.image_pil, graph=ctx.graph,
        conf=action.conf, clahe=False, tiling=tiling,
        pass_number=pass_number, prompt=action.prompt,
        cfg=ctx.cfg, oracle=ctx.oracle, query_set=ctx.query_set,
        roi_override=roi_override, nms_mode=nms_mode, gate_mode=gate_mode,
        use_canopy_roi=use_canopy_roi, nms_iou_threshold=nms_iou_threshold,
        nms_iom_threshold=nms_iom_threshold,
        cross_pass_dedup_metric=cross_pass_dedup_metric,
        cross_pass_dedup_threshold=cross_pass_dedup_threshold,
    )
    ctx.n_passes = pass_number
    _meter_query(ctx, stats)
    return int(stats)


def _meter_query(ctx, stats):
    """Meter a Query/Look from the pass's actual call counts (PassStats):
    global SAM3 calls (canopy if run + leaf map if generated + global proposal),
    tile calls, and any inline FM+V-IP oracle calls made under verifier='vip'."""
    if ctx.cost is None:
        return
    ctx.cost.n_sam += int(getattr(stats, "n_sam_calls", 0))
    ctx.cost.n_tile += int(getattr(stats, "n_tiles", 0))
    ctx.cost.n_verify += int(getattr(stats, "n_verify_calls", 0))


def _execute_look(action, ctx) -> int:
    """Sense a policy-proposed ROI (docs/active_perception_formulation.md §6).

    Expand the proposed region by cfg.roi_margin, clamp to the image, then run
    ONE untiled SAM3 query restricted to it -- selecting an ROI already is the
    tiling, so the crop is sensed in a single global inference (v2 decision).
    The ROI is a SENSING TARGET ONLY: it is forwarded to execute_pass as
    roi_override and is never inserted into the graph; every candidate still
    originates from SAM3 inside the ROI. Exemplars are picked up automatically
    by execute_pass (the graph feedback path), so a Look run after the global
    bootstrap pass is exemplar-primed.

    Returns the number of new candidate tracks, or 0 (no model call) when the ROI
    is below cfg.roi_min_size, zoomed past cfg.roi_max_depth quadrant levels, or
    duplicates an already-sensed ROI.
    """
    cfg = ctx.cfg
    margin = getattr(cfg, "roi_margin", 0.10)
    min_size = getattr(cfg, "roi_min_size", 32.0)
    max_depth = getattr(cfg, "roi_max_depth", 2)
    dup_iou = getattr(cfg, "roi_dup_iou", 0.7)

    img_np = np.asarray(ctx.image_pil)
    img_h, img_w = img_np.shape[:2]
    roi = _expand_and_clamp(action.region, margin, img_w, img_h)
    rw, rh = roi[2] - roi[0], roi[3] - roi[1]

    # Guard 1: degenerate / too small to tile meaningfully.
    if rw < min_size or rh < min_size:
        logging.info("LookROIA: ROI %s below min size %.0f px; no-op.", roi, min_size)
        return 0

    # Guard 2: zoomed finer than roi_max_depth quadrant levels. Each quadrant level
    # quarters the area, so a level-d region has area >= frame/4^d; anything below
    # that floor is deeper than the allowed 2 splits (the user's 16-quadrant cap).
    min_area = (img_w * img_h) / float(4 ** max_depth)
    if rw * rh < min_area:
        logging.info("LookROIA: ROI %s past max depth %d (area floor %.0f); no-op.",
                     roi, max_depth, min_area)
        return 0

    # Guard 3: substantially overlaps a previously-sensed ROI (avoid re-sensing).
    for prev in ctx.sensed_rois:
        if _roi_iou(roi, prev) > dup_iou:
            logging.info("LookROIA: ROI %s duplicates sensed ROI %s; no-op.", roi, prev)
            return 0

    # Sense: reuse the Query executor (metering + pass-number bookkeeping) with a
    # single untiled, ROI-restricted pass. The VLM chose only WHERE to look; the
    # concept prompt and confidence are system-level (never model-supplied).
    prompt = getattr(cfg, "target_prompt", "green fruit")
    conf = getattr(cfg, "conf", 0.3)
    inner = QueryA(region=roi, prompt=prompt, conf=conf)
    n_new = _execute_query(inner, ctx, tiling=False, roi_override=roi)
    ctx.sensed_rois.append(roi)
    return n_new


def execute(action, ctx) -> int:
    """Dispatch an action, returning the number of new candidate tracks created."""
    if ctx.cost is not None:
        ctx.cost.n_orch += 1  # one orchestration decision per executed action
    if isinstance(action, QueryA):
        return _execute_query(action, ctx, tiling=False, roi_override=action.region)
    if isinstance(action, LookROIA):
        return _execute_look(action, ctx)
    if isinstance(action, StopA):
        return 0
    raise TypeError(f"Unknown action type: {type(action).__name__}")

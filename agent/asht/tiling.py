"""SAM3Count-style density-aware ROI tiling with complete provenance.

The implementation is dependency-light and works through the staged SAM3
executor.  It follows the released SAM3Count image pipeline's main ideas:
full-image density assessment, LARGE/MEDIUM/SMALL adaptive grids, an ROI formed
from stage-1 detections, overlapping tile inference, and cross-tile NMS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline_stages import DetectionBatch, Sam3QuerySpec, StageContext, execute_sam3_request
from provenance.contracts import DedupComparisonRecord, TileRecord, TilingDecisionRecord
from provenance.ids import EntityKind
from provenance.schema import CoordinateSpace, DedupDecision, TilingMode

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TileRule:
    name: str
    target_columns: int
    target_rows: int
    overlap_ratio: float
    description: str

    def __post_init__(self) -> None:
        if self.target_columns < 1 or self.target_rows < 1:
            raise ValueError("tile grid dimensions must be positive")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must lie in [0, 1)")


DEFAULT_TILE_RULES: Mapping[str, TileRule] = {
    "LARGE": TileRule("LARGE", 2, 1, 0.35, "coarse grid for many larger objects"),
    "MEDIUM": TileRule("MEDIUM", 4, 2, 0.30, "balanced grid for mid-size dense scenes"),
    "SMALL": TileRule("SMALL", 6, 4, 0.25, "fine grid for many small objects"),
}


@dataclass(frozen=True)
class AdaptiveTilingConfig:
    min_tile_size: int = 97
    max_tile_size: int = 1024
    count_normalizer: float = 50.0
    many_object_threshold: int = 90
    high_coverage_threshold: float = 0.40
    high_coverage_max_size_ratio: float = 0.02
    small_object_density_threshold: float = 0.70
    small_object_max_size_ratio: float = 0.01
    coverage_weight: float = 0.30
    count_weight: float = 0.50
    size_weight: float = 0.20
    roi_padding_ratio: float = 0.04
    roi_min_padding_pixels: int = 16
    roi_tile_intersection_ratio: float = 0.05
    cross_tile_iou_threshold: float = 0.50
    max_tiles_per_action: int = 64
    force_minimum_2x2: bool = True
    rules: Mapping[str, TileRule] = field(default_factory=lambda: dict(DEFAULT_TILE_RULES))

    def __post_init__(self) -> None:
        if self.min_tile_size < 1 or self.max_tile_size < self.min_tile_size:
            raise ValueError("invalid tile-size bounds")
        if self.count_normalizer <= 0 or self.many_object_threshold < 1:
            raise ValueError("invalid density count parameters")
        if self.max_tiles_per_action < 1:
            raise ValueError("max_tiles_per_action must be positive")
        probabilities = (
            self.high_coverage_threshold,
            self.high_coverage_max_size_ratio,
            self.small_object_density_threshold,
            self.small_object_max_size_ratio,
            self.roi_padding_ratio,
            self.roi_tile_intersection_ratio,
            self.cross_tile_iou_threshold,
        )
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in probabilities):
            raise ValueError("tiling thresholds must be finite and lie in [0, 1]")
        weights = self.coverage_weight + self.count_weight + self.size_weight
        if not math.isclose(weights, 1.0, abs_tol=1e-6):
            raise ValueError("density weights must sum to one")


@dataclass(frozen=True)
class DensityAssessment:
    density_score: float
    use_tiling: bool
    tiling_rule: str
    reason: str
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class TileSpec:
    index: int
    row: int
    column: int
    local_box: Box
    global_box: Box
    overlap_fraction: float


@dataclass(frozen=True)
class TiledExecutionResult:
    decision: TilingDecisionRecord
    tiles: tuple[TileRecord, ...]
    raw: DetectionBatch
    merged: DetectionBatch
    comparisons: tuple[DedupComparisonRecord, ...]
    sam3_call_count: int


def assess_density(
    boxes: Sequence[Sequence[float]] | np.ndarray,
    *,
    image_width: int,
    image_height: int,
    config: AdaptiveTilingConfig | None = None,
) -> DensityAssessment:
    """Estimate density using SAM3Count's coverage/count/size formulation."""

    cfg = config or AdaptiveTilingConfig()
    arr = np.asarray(boxes, dtype=float).reshape((-1, 4))
    image_area = max(1.0, float(image_width * image_height))
    if len(arr) == 0:
        return DensityAssessment(
            density_score=0.0,
            use_tiling=False,
            tiling_rule="NONE",
            reason="no objects detected",
            metrics={
                "coverage": 0.0,
                "object_count": 0,
                "avg_object_size_ratio": 0.0,
                "density_score": 0.0,
                "object_count_score": 0.0,
                "size_score": 0.0,
            },
        )

    widths = np.maximum(0.0, arr[:, 2] - arr[:, 0])
    heights = np.maximum(0.0, arr[:, 3] - arr[:, 1])
    areas = widths * heights
    area_sum = float(np.sum(areas))
    object_count = int(len(arr))
    coverage = area_sum / image_area
    count_score = min(object_count / cfg.count_normalizer, 1.0)
    average_size_ratio = float(np.mean(areas) / image_area)
    size_score = 1.0 - min(average_size_ratio / 0.1, 1.0)
    density_score = (
        cfg.coverage_weight * coverage
        + cfg.count_weight * count_score
        + cfg.size_weight * size_score
    )

    if object_count > cfg.many_object_threshold:
        use_tiling = True
        rule = "LARGE"
        reason = f"many objects ({object_count} > {cfg.many_object_threshold})"
    elif (
        coverage > cfg.high_coverage_threshold
        and average_size_ratio < cfg.high_coverage_max_size_ratio
    ):
        use_tiling = True
        rule = "MEDIUM"
        reason = "high coverage with small average objects"
    elif (
        density_score > cfg.small_object_density_threshold
        and average_size_ratio < cfg.small_object_max_size_ratio
    ):
        use_tiling = True
        rule = "SMALL"
        reason = "high density score with very small objects"
    else:
        use_tiling = False
        rule = "NONE"
        reason = "scene does not meet adaptive tiling thresholds"

    return DensityAssessment(
        density_score=float(density_score),
        use_tiling=use_tiling,
        tiling_rule=rule,
        reason=reason,
        metrics={
            "coverage": float(coverage),
            "object_count": object_count,
            "avg_object_size_ratio": average_size_ratio,
            "density_score": float(density_score),
            "object_count_score": float(count_score),
            "size_score": float(size_score),
        },
    )


def calculate_tile_parameters(
    width: int,
    height: int,
    rule_name: str,
    *,
    config: AdaptiveTilingConfig | None = None,
    tile_scale: float | None = None,
) -> tuple[int, int, int, TileRule]:
    cfg = config or AdaptiveTilingConfig()
    rule = cfg.rules.get(rule_name, cfg.rules["MEDIUM"])
    target_w = width / rule.target_columns
    target_h = height / rule.target_rows
    tile_size = int(max(target_w, target_h))
    if tile_scale is not None:
        if not math.isfinite(tile_scale) or tile_scale <= 0:
            raise ValueError("tile_scale must be finite and positive")
        tile_size = int(round(tile_size / tile_scale))
    tile_size = max(cfg.min_tile_size, min(tile_size, cfg.max_tile_size))
    overlap = int(tile_size * rule.overlap_ratio)
    stride = max(1, tile_size - overlap)
    return tile_size, overlap, stride, rule


def build_union_roi(
    boxes: Sequence[Sequence[float]] | np.ndarray,
    *,
    image_width: int,
    image_height: int,
    config: AdaptiveTilingConfig | None = None,
) -> Box | None:
    cfg = config or AdaptiveTilingConfig()
    arr = np.asarray(boxes, dtype=float).reshape((-1, 4))
    if len(arr) == 0:
        return None
    x1, y1 = float(np.min(arr[:, 0])), float(np.min(arr[:, 1]))
    x2, y2 = float(np.max(arr[:, 2])), float(np.max(arr[:, 3]))
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = max(float(cfg.roi_min_padding_pixels), cfg.roi_padding_ratio * width)
    pad_y = max(float(cfg.roi_min_padding_pixels), cfg.roi_padding_ratio * height)
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width), x2 + pad_x),
        min(float(image_height), y2 + pad_y),
    )


def generate_tiles(
    *,
    source_region: Box,
    source_width: int,
    source_height: int,
    tile_size: int,
    overlap: int,
    roi_global: Box | None,
    config: AdaptiveTilingConfig | None = None,
) -> tuple[TileSpec, ...]:
    cfg = config or AdaptiveTilingConfig()
    stride = max(1, tile_size - overlap)
    rows = max(1, int(math.ceil(source_height / stride)))
    columns = max(1, int(math.ceil(source_width / stride)))
    if cfg.force_minimum_2x2 and rows == 1 and columns == 1:
        if source_height >= tile_size // 2 and source_width >= tile_size // 2:
            rows = columns = 2

    x_offset, y_offset = source_region[0], source_region[1]
    seen: set[tuple[int, int, int, int]] = set()
    specs: list[TileSpec] = []
    for row in range(rows):
        for column in range(columns):
            y1 = min(row * stride, max(0, source_height - tile_size))
            x1 = min(column * stride, max(0, source_width - tile_size))
            y2 = min(y1 + tile_size, source_height)
            x2 = min(x1 + tile_size, source_width)
            key = (int(x1), int(y1), int(x2), int(y2))
            if key in seen:
                continue
            seen.add(key)
            local_box = tuple(float(value) for value in key)
            global_box = (
                float(x_offset + x1),
                float(y_offset + y1),
                float(x_offset + x2),
                float(y_offset + y2),
            )
            if roi_global is not None:
                intersection = _intersection_area(global_box, roi_global)
                tile_area = max(1.0, (global_box[2] - global_box[0]) * (global_box[3] - global_box[1]))
                if intersection <= 0.0:
                    continue
                if intersection / tile_area < cfg.roi_tile_intersection_ratio:
                    # SAM3Count keeps any intersecting tile; retain that behavior
                    # while recording the exact ratio for later analysis.
                    pass
            specs.append(
                TileSpec(
                    index=len(specs),
                    row=row,
                    column=column,
                    local_box=local_box,
                    global_box=global_box,
                    overlap_fraction=float(overlap / max(tile_size, 1)),
                )
            )
    if not specs and roi_global is not None:
        return generate_tiles(
            source_region=source_region,
            source_width=source_width,
            source_height=source_height,
            tile_size=tile_size,
            overlap=overlap,
            roi_global=None,
            config=cfg,
        )
    return tuple(specs)


def execute_adaptive_tiling(
    *,
    backend: Any,
    processor: Any,
    image_np: np.ndarray,
    base_query: Sam3QuerySpec,
    stage1_batch: DetectionBatch,
    mode: TilingMode,
    context: StageContext,
    config: AdaptiveTilingConfig | None = None,
    tile_scale: float | None = None,
    agent_requested: bool = False,
    max_additional_calls: int | None = None,
) -> TiledExecutionResult:
    """Decide, execute, and merge tiled SAM3 requests."""

    cfg = config or AdaptiveTilingConfig()
    height, width = int(image_np.shape[0]), int(image_np.shape[1])
    assessment = assess_density(
        stage1_batch.boxes_local,
        image_width=width,
        image_height=height,
        config=cfg,
    )
    if mode is TilingMode.OFF:
        triggered = False
        trigger_source = "off"
        rule_name = "NONE"
        reason = "tiling disabled"
    elif mode is TilingMode.ALWAYS:
        triggered = True
        trigger_source = "always"
        rule_name = assessment.tiling_rule if assessment.tiling_rule != "NONE" else "MEDIUM"
        reason = "tiling forced"
    elif mode is TilingMode.DENSITY_ADAPTIVE:
        triggered = assessment.use_tiling
        trigger_source = "density"
        rule_name = assessment.tiling_rule
        reason = assessment.reason
    elif mode is TilingMode.AGENT_CONTROLLED:
        triggered = bool(agent_requested)
        trigger_source = "agent"
        rule_name = assessment.tiling_rule if assessment.tiling_rule != "NONE" else "MEDIUM"
        reason = "agent requested tiled sensing" if triggered else "agent did not request tiling"
    else:
        raise ValueError(f"unsupported tiling mode: {mode}")

    roi = build_union_roi(
        stage1_batch.boxes_global,
        image_width=int(base_query.region[2]),
        image_height=int(base_query.region[3]),
        config=cfg,
    )
    tile_size = overlap = stride = 0
    rule_description = None
    specs: tuple[TileSpec, ...] = ()
    if triggered:
        tile_size, overlap, stride, rule = calculate_tile_parameters(
            width,
            height,
            rule_name,
            config=cfg,
            tile_scale=tile_scale,
        )
        rule_description = rule.description
        base_specs = generate_tiles(
            source_region=base_query.region,
            source_width=width,
            source_height=height,
            tile_size=tile_size,
            overlap=overlap,
            roi_global=None,
            config=cfg,
        )
        specs = _filter_tile_specs_by_roi(
            base_specs, roi, min_intersection_ratio=cfg.roi_tile_intersection_ratio
        )
        if roi is not None and not specs:
            specs = base_specs
    else:
        base_specs = ()
    base_tile_count = len(base_specs)
    tile_limit = cfg.max_tiles_per_action
    if max_additional_calls is not None:
        tile_limit = min(tile_limit, max(0, int(max_additional_calls)))
    specs = specs[:tile_limit]
    if triggered and not specs:
        triggered = False
        reason = "tiling selected but no tile-call budget remained"

    decision = TilingDecisionRecord(
        tiling_decision_id=context.new_id(EntityKind.TILING_DECISION),
        pass_id=context.pass_id,
        mode=mode,
        triggered=triggered,
        trigger_source=trigger_source,
        tiling_rule=rule_name,
        reason=reason,
        source_detection_ids=stage1_batch.detection_ids,
        source_roi=base_query.region,
        density_metrics=dict(assessment.metrics),
        reliability_metrics={},
        tile_size=tile_size if triggered else None,
        overlap_pixels=overlap if triggered else None,
        stride_pixels=stride if triggered else None,
        base_tile_count=base_tile_count,
        selected_tile_count=len(specs),
        thresholds={
            "many_object_threshold": cfg.many_object_threshold,
            "high_coverage_threshold": cfg.high_coverage_threshold,
            "high_coverage_max_size_ratio": cfg.high_coverage_max_size_ratio,
            "small_object_density_threshold": cfg.small_object_density_threshold,
            "small_object_max_size_ratio": cfg.small_object_max_size_ratio,
            "roi_tile_intersection_ratio": cfg.roi_tile_intersection_ratio,
            "cross_tile_iou_threshold": cfg.cross_tile_iou_threshold,
        },
        selected_roi=roi,
        metadata={"rule_description": rule_description, "tile_scale": tile_scale},
    )
    context.emit(decision)

    if not triggered:
        empty = _empty_batch(base_query)
        return TiledExecutionResult(decision, (), empty, empty, (), 0)

    batches: list[DetectionBatch] = []
    tile_records: list[TileRecord] = []
    source_x, source_y = base_query.region[0], base_query.region[1]
    for spec in specs:
        x1, y1, x2, y2 = (int(value) for value in spec.local_box)
        tile_image = image_np[y1:y2, x1:x2]
        tile_id = context.new_id(EntityKind.TILE)
        positive_boxes = _boxes_for_tile(base_query.positive_boxes, spec.local_box)
        negative_boxes = _boxes_for_tile(base_query.negative_boxes, spec.local_box)
        tile_query = Sam3QuerySpec(
            prompt=base_query.prompt,
            threshold=base_query.threshold,
            region=spec.global_box,
            positive_boxes=positive_boxes,
            negative_boxes=negative_boxes,
            positive_exemplar_node_ids=base_query.positive_exemplar_node_ids,
            negative_exemplar_node_ids=base_query.negative_exemplar_node_ids,
            coordinate_space=CoordinateSpace.TILE_LOCAL,
            disable_size_filter=base_query.disable_size_filter,
            return_masks=base_query.return_masks,
            preprocessing={**dict(base_query.preprocessing), "tiled": True},
            decoder_settings={
                **dict(base_query.decoder_settings),
                "tile_index": spec.index,
                "source_region_origin": [source_x, source_y],
            },
            tile_id=tile_id,
        )
        batch = execute_sam3_request(
            backend,
            processor,
            tile_image,
            tile_query,
            context=context,
        )
        if batch.masks:
            expanded_masks = []
            for mask in batch.masks:
                expanded = np.zeros((height, width), dtype=bool)
                mask_arr = np.asarray(mask, dtype=bool)
                target_h, target_w = y2 - y1, x2 - x1
                if mask_arr.shape != (target_h, target_w):
                    raise ValueError(
                        "tile mask shape does not match the tile image dimensions"
                    )
                expanded[y1:y2, x1:x2] = mask_arr
                expanded_masks.append(expanded)
            batch = DetectionBatch(
                boxes_local=batch.boxes_local,
                boxes_global=batch.boxes_global,
                scores=batch.scores,
                masks=tuple(expanded_masks),
                detection_ids=batch.detection_ids,
                coordinate_space=batch.coordinate_space,
                source_region=batch.source_region,
                prompt=batch.prompt,
                sam3_call_id=batch.sam3_call_id,
                tile_id=batch.tile_id,
                source_indices=batch.source_indices,
            )
        batches.append(batch)
        call_ids = (batch.sam3_call_id,) if batch.sam3_call_id else ()
        record = TileRecord(
            tile_id=tile_id,
            pass_id=context.pass_id,
            tile_index=spec.index,
            grid_row=spec.row,
            grid_column=spec.column,
            local_box=spec.local_box,
            global_box=spec.global_box,
            overlap_fraction=spec.overlap_fraction,
            source_roi=base_query.region,
            sam3_call_ids=call_ids,
            raw_detection_ids=batch.detection_ids,
            transform={
                "translation": [spec.global_box[0], spec.global_box[1]],
                "tile_width": x2 - x1,
                "tile_height": y2 - y1,
            },
        )
        context.emit(record)
        tile_records.append(record)

    raw = concatenate_batches(batches, source_region=base_query.region, prompt=base_query.prompt)
    merged, comparisons = suppress_cross_tile_duplicates(
        raw,
        pass_id=context.pass_id,
        id_source=context,
        iou_threshold=cfg.cross_tile_iou_threshold,
    )
    for comparison in comparisons:
        context.emit(comparison)
    return TiledExecutionResult(
        decision=decision,
        tiles=tuple(tile_records),
        raw=raw,
        merged=merged,
        comparisons=comparisons,
        sam3_call_count=len(specs),
    )


def concatenate_batches(
    batches: Sequence[DetectionBatch],
    *,
    source_region: Box,
    prompt: str,
) -> DetectionBatch:
    if not batches or sum(len(batch) for batch in batches) == 0:
        return DetectionBatch(
            boxes_local=np.empty((0, 4), dtype=float),
            boxes_global=np.empty((0, 4), dtype=float),
            scores=np.empty((0,), dtype=float),
            source_region=source_region,
            prompt=prompt,
            coordinate_space=CoordinateSpace.IMAGE_GLOBAL,
        )
    boxes_global = np.concatenate([batch.boxes_global for batch in batches], axis=0)
    # At this point the merged working coordinate system is global.
    masks = tuple(mask for batch in batches for mask in batch.masks)
    ids = tuple(value for batch in batches for value in batch.detection_ids)
    return DetectionBatch(
        boxes_local=boxes_global.copy(),
        boxes_global=boxes_global,
        scores=np.concatenate([batch.scores for batch in batches], axis=0),
        masks=masks,
        detection_ids=ids,
        coordinate_space=CoordinateSpace.IMAGE_GLOBAL,
        source_region=source_region,
        prompt=prompt,
        source_indices=tuple(index for batch in batches for index in batch.source_indices),
    )


def suppress_cross_tile_duplicates(
    batch: DetectionBatch,
    *,
    pass_id: str,
    id_source,
    iou_threshold: float,
) -> tuple[DetectionBatch, tuple[DedupComparisonRecord, ...]]:
    if len(batch) == 0:
        return batch, ()
    order = np.argsort(batch.scores)[::-1]
    kept: list[int] = []
    suppressed: set[int] = set()
    comparisons: list[DedupComparisonRecord] = []
    for position, i_value in enumerate(order):
        i = int(i_value)
        if i in suppressed:
            continue
        kept.append(i)
        for j_value in order[position + 1 :]:
            j = int(j_value)
            if j in suppressed:
                continue
            iou = _iou(batch.boxes_global[i], batch.boxes_global[j])
            should_suppress = iou > iou_threshold
            comparisons.append(
                DedupComparisonRecord(
                    dedup_decision_id=id_source.new_id(EntityKind.DEDUP_DECISION),
                    pass_id=pass_id,
                    new_detection_id=batch.detection_ids[j],
                    existing_detection_id=batch.detection_ids[i],
                    existing_node_id=None,
                    stage="cross_tile_nms",
                    metrics={"iou": float(iou)},
                    thresholds={"iou": float(iou_threshold)},
                    decision=(
                        DedupDecision.SUPPRESS_NEW
                        if should_suppress
                        else DedupDecision.KEEP_DISTINCT
                    ),
                    selected_survivor_id=batch.detection_ids[i] if should_suppress else None,
                    reason="higher-score tile detection retained" if should_suppress else "below threshold",
                )
            )
            if should_suppress:
                suppressed.add(j)
    kept_sorted = sorted(kept)
    return batch.subset(kept_sorted), tuple(comparisons)


def _filter_tile_specs_by_roi(
    specs: Sequence[TileSpec],
    roi: Box | None,
    *,
    min_intersection_ratio: float,
) -> tuple[TileSpec, ...]:
    if roi is None:
        return tuple(specs)
    kept: list[TileSpec] = []
    for spec in specs:
        intersection = _intersection_area(spec.global_box, roi)
        area = max(
            1.0,
            (spec.global_box[2] - spec.global_box[0])
            * (spec.global_box[3] - spec.global_box[1]),
        )
        # The released SAM3Count code keeps any intersecting tile, while also
        # exposing a nominal intersection threshold.  Preserve that behavior.
        if intersection / area >= min_intersection_ratio or intersection > 0.0:
            kept.append(spec)
    return tuple(kept)


def _boxes_for_tile(boxes: Sequence[Box], tile_local: Box) -> tuple[Box, ...]:
    tx1, ty1, tx2, ty2 = tile_local
    result: list[Box] = []
    for box in boxes:
        x1 = max(tx1, float(box[0]))
        y1 = max(ty1, float(box[1]))
        x2 = min(tx2, float(box[2]))
        y2 = min(ty2, float(box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        result.append((x1 - tx1, y1 - ty1, x2 - tx1, y2 - ty1))
    return tuple(result)


def _empty_batch(query: Sam3QuerySpec) -> DetectionBatch:
    return DetectionBatch(
        boxes_local=np.empty((0, 4), dtype=float),
        boxes_global=np.empty((0, 4), dtype=float),
        scores=np.empty((0,), dtype=float),
        source_region=query.region,
        prompt=query.prompt,
        coordinate_space=CoordinateSpace.IMAGE_GLOBAL,
    )


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = _intersection_area(a, b)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


__all__ = [
    "AdaptiveTilingConfig",
    "DEFAULT_TILE_RULES",
    "DensityAssessment",
    "TileRule",
    "TileSpec",
    "TiledExecutionResult",
    "assess_density",
    "build_union_roi",
    "calculate_tile_parameters",
    "concatenate_batches",
    "execute_adaptive_tiling",
    "generate_tiles",
    "suppress_cross_tile_duplicates",
]

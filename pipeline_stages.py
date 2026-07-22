"""Explicit, provenance-aware stages for SAM3 sensing and registration.

This module is deliberately independent of the legacy ``execute_pass`` control
flow.  It provides the typed stage boundaries that the legacy wrapper and the
future ASHT runner can adopt incrementally:

request -> SAM3 execution -> normalization -> coordinate translation ->
intra-pass suppression -> cross-pass registration.

No stage silently mutates the graph except ``register_detection_batch``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from provenance.contracts import (
    DedupComparisonRecord,
    GraphNodeSnapshotRecord,
    RawDetectionRecord,
    RegistrationDecisionRecord,
    Sam3CallRecord,
)
from provenance.ids import EntityKind
from provenance.run_store import ReportingLevel
from provenance.schema import (
    CoordinateSpace,
    DedupDecision,
    NodeStatus,
    RegistrationOutcome,
)

Box = tuple[float, float, float, float]


class StageError(ValueError):
    """Raised when a stage receives inconsistent or invalid data."""


class ProvenanceSink(Protocol):
    """Small subset of ``RunStore`` used by pipeline stages."""

    def new_id(self, kind: EntityKind) -> str:
        ...

    def append(
        self,
        payload: Any,
        *,
        event_kind: Any | None = None,
        pass_id: str | None = None,
        minimum_level: ReportingLevel = ReportingLevel.STANDARD,
    ) -> Any:
        ...


@dataclass(frozen=True)
class StageContext:
    """Provenance context shared by all stages in one pipeline pass."""

    pass_id: str
    model_id: str = "sam3"
    sink: ProvenanceSink | None = None

    def new_id(self, kind: EntityKind) -> str | None:
        return self.sink.new_id(kind) if self.sink is not None else None

    def emit(
        self,
        payload: Any,
        *,
        minimum_level: ReportingLevel = ReportingLevel.STANDARD,
    ) -> None:
        if self.sink is not None:
            self.sink.append(
                payload,
                pass_id=self.pass_id,
                minimum_level=minimum_level,
            )


@dataclass(frozen=True)
class Sam3QuerySpec:
    """Complete, normalized request for one SAM3 invocation."""

    prompt: str
    threshold: float
    region: Box
    positive_boxes: tuple[Box, ...] = ()
    negative_boxes: tuple[Box, ...] = ()
    positive_exemplar_node_ids: tuple[str, ...] = ()
    negative_exemplar_node_ids: tuple[str, ...] = ()
    coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_GLOBAL
    disable_size_filter: bool = False
    return_masks: bool = False
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    decoder_settings: Mapping[str, Any] = field(default_factory=dict)
    tile_id: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise StageError("prompt must be non-empty")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise StageError("threshold must be finite and in [0, 1]")
        _validate_box(self.region, "region")
        for index, box in enumerate(self.positive_boxes):
            _validate_box(box, f"positive_boxes[{index}]")
        for index, box in enumerate(self.negative_boxes):
            _validate_box(box, f"negative_boxes[{index}]")


@dataclass
class DetectionBatch:
    """Index-aligned detections moving between explicit pipeline stages."""

    boxes_local: np.ndarray
    boxes_global: np.ndarray
    scores: np.ndarray
    masks: tuple[np.ndarray, ...] = ()
    detection_ids: tuple[str, ...] = ()
    coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_GLOBAL
    source_region: Box = (0.0, 0.0, 0.0, 0.0)
    prompt: str = ""
    sam3_call_id: str | None = None
    tile_id: str | None = None
    source_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.boxes_local = np.asarray(self.boxes_local, dtype=float).reshape((-1, 4))
        self.boxes_global = np.asarray(self.boxes_global, dtype=float).reshape((-1, 4))
        self.scores = np.asarray(self.scores, dtype=float).reshape((-1,))
        n = len(self.scores)
        if len(self.boxes_local) != n or len(self.boxes_global) != n:
            raise StageError("boxes and scores must have the same length")
        if self.masks and len(self.masks) != n:
            raise StageError("masks must be index-aligned with detections")
        if self.detection_ids and len(self.detection_ids) != n:
            raise StageError("detection_ids must be index-aligned with detections")
        if self.source_indices and len(self.source_indices) != n:
            raise StageError("source_indices must be index-aligned with detections")
        if np.any(~np.isfinite(self.boxes_local)) or np.any(~np.isfinite(self.boxes_global)):
            raise StageError("detection boxes must be finite")
        if np.any(~np.isfinite(self.scores)) or np.any(self.scores < 0.0) or np.any(self.scores > 1.0):
            raise StageError("detection scores must be finite and in [0, 1]")
        for index, box in enumerate(self.boxes_local):
            _validate_box(tuple(float(v) for v in box), f"boxes_local[{index}]")
        for index, box in enumerate(self.boxes_global):
            _validate_box(tuple(float(v) for v in box), f"boxes_global[{index}]")
        if not self.source_indices:
            self.source_indices = tuple(range(n))

    def __len__(self) -> int:
        return len(self.scores)

    def subset(self, indices: Sequence[int]) -> "DetectionBatch":
        idx = np.asarray(indices, dtype=int)
        return DetectionBatch(
            boxes_local=self.boxes_local[idx],
            boxes_global=self.boxes_global[idx],
            scores=self.scores[idx],
            masks=tuple(self.masks[int(i)] for i in idx) if self.masks else (),
            detection_ids=tuple(self.detection_ids[int(i)] for i in idx) if self.detection_ids else (),
            coordinate_space=self.coordinate_space,
            source_region=self.source_region,
            prompt=self.prompt,
            sam3_call_id=self.sam3_call_id,
            tile_id=self.tile_id,
            source_indices=tuple(self.source_indices[int(i)] for i in idx),
        )


@dataclass(frozen=True)
class SuppressionResult:
    kept: DetectionBatch
    removed_detection_ids: tuple[str, ...]
    comparisons: tuple[DedupComparisonRecord, ...] = ()


@dataclass(frozen=True)
class RegistrationResult:
    created_node_ids: tuple[str, ...]
    updated_node_ids: tuple[str, ...]
    rejected_detection_ids: tuple[str, ...]
    comparisons: tuple[DedupComparisonRecord, ...] = ()
    decisions: tuple[RegistrationDecisionRecord, ...] = ()



def execute_sam3_request(
    backend: Any,
    processor: Any,
    image_np: np.ndarray,
    query: Sam3QuerySpec,
    *,
    context: StageContext | None = None,
) -> DetectionBatch:
    """Execute exactly one SAM3 call and normalize its output.

    ``backend`` must expose ``run_raw_inference`` with the existing repository
    signature.  The function is injectable so tests do not import torch.
    """

    pos = _boxes_to_array(query.positive_boxes)
    neg = _boxes_to_array(query.negative_boxes)
    started = time.perf_counter()
    raw = backend.run_raw_inference(
        processor,
        image_np,
        query.threshold,
        prompt=query.prompt,
        pos_boxes=pos,
        neg_boxes=neg,
        disable_size_filter=query.disable_size_filter,
        return_masks=query.return_masks,
    )
    latency = time.perf_counter() - started

    if query.return_masks:
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise StageError("SAM3 must return (boxes, scores, masks) when return_masks=True")
        boxes, scores, masks = raw
    else:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise StageError("SAM3 must return (boxes, scores) when return_masks=False")
        boxes, scores = raw
        masks = ()

    sam3_call_id = context.new_id(EntityKind.SAM3_CALL) if context else None
    detection_ids: tuple[str, ...] = ()
    if context and context.sink is not None:
        detection_ids = tuple(
            context.new_id(EntityKind.RAW_DETECTION) for _ in range(len(scores))
        )  # type: ignore[arg-type]

    batch = normalize_sam3_output(
        boxes,
        scores,
        masks=masks,
        query=query,
        detection_ids=detection_ids,
        sam3_call_id=sam3_call_id,
    )

    if context and context.sink is not None and sam3_call_id is not None:
        call_record = Sam3CallRecord(
            sam3_call_id=sam3_call_id,
            pass_id=context.pass_id,
            model_id=context.model_id,
            prompt=query.prompt,
            region=query.region,
            threshold=query.threshold,
            positive_exemplar_node_ids=query.positive_exemplar_node_ids,
            negative_exemplar_node_ids=query.negative_exemplar_node_ids,
            preprocessing=dict(query.preprocessing),
            decoder_settings=dict(query.decoder_settings),
            raw_output_artifact_ids=(),
            raw_detection_ids=batch.detection_ids,
            tile_id=query.tile_id,
            latency_seconds=float(latency),
            coordinate_space=query.coordinate_space,
        )
        context.emit(call_record)
        for index in range(len(batch)):
            context.emit(
                RawDetectionRecord(
                    raw_detection_id=batch.detection_ids[index],
                    sam3_call_id=sam3_call_id,
                    pass_id=context.pass_id,
                    box_local=_box_tuple(batch.boxes_local[index]),
                    box_global=_box_tuple(batch.boxes_global[index]),
                    score=float(batch.scores[index]),
                    prompt=query.prompt,
                    coordinate_space=query.coordinate_space,
                    tile_id=query.tile_id,
                    geometry={"source_region": list(query.region)},
                    raw_index=index,
                )
            )
    return batch



def normalize_sam3_output(
    boxes: Any,
    scores: Any,
    *,
    masks: Iterable[Any] = (),
    query: Sam3QuerySpec,
    detection_ids: tuple[str, ...] = (),
    sam3_call_id: str | None = None,
) -> DetectionBatch:
    """Validate and normalize raw SAM3 arrays without changing detections."""

    boxes_arr = np.asarray(boxes, dtype=float)
    if boxes_arr.size == 0:
        boxes_arr = np.empty((0, 4), dtype=float)
    boxes_arr = boxes_arr.reshape((-1, 4))
    scores_arr = np.asarray(scores, dtype=float).reshape((-1,))
    if len(boxes_arr) != len(scores_arr):
        raise StageError("SAM3 returned different numbers of boxes and scores")
    masks_tuple = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if masks_tuple and len(masks_tuple) != len(scores_arr):
        raise StageError("SAM3 returned different numbers of masks and scores")

    boxes_global = _translate_boxes(boxes_arr, query.region, query.coordinate_space)
    return DetectionBatch(
        boxes_local=boxes_arr,
        boxes_global=boxes_global,
        scores=scores_arr,
        masks=masks_tuple,
        detection_ids=detection_ids,
        coordinate_space=query.coordinate_space,
        source_region=query.region,
        prompt=query.prompt,
        sam3_call_id=sam3_call_id,
        tile_id=query.tile_id,
    )



def translate_detection_batch(
    batch: DetectionBatch,
    region: Box,
    *,
    source_space: CoordinateSpace | None = None,
) -> DetectionBatch:
    """Translate local boxes into image-global coordinates, preserving lineage."""

    space = source_space or batch.coordinate_space
    global_boxes = _translate_boxes(batch.boxes_local, region, space)
    return DetectionBatch(
        boxes_local=batch.boxes_local.copy(),
        boxes_global=global_boxes,
        scores=batch.scores.copy(),
        masks=tuple(mask.copy() for mask in batch.masks),
        detection_ids=batch.detection_ids,
        coordinate_space=CoordinateSpace.IMAGE_GLOBAL,
        source_region=region,
        prompt=batch.prompt,
        sam3_call_id=batch.sam3_call_id,
        tile_id=batch.tile_id,
        source_indices=batch.source_indices,
    )



def suppress_detection_batch(
    batch: DetectionBatch,
    backend: Any,
    *,
    confidence: float,
    mode: str = "dualgate",
    gate_mode: str = "dual",
    iou_threshold: float = 0.40,
    iom_threshold: float = 0.90,
    use_concentric: bool = False,
    context: StageContext | None = None,
) -> SuppressionResult:
    """Run intra-pass NMS while retaining source IDs and audit decisions."""

    if len(batch) == 0:
        return SuppressionResult(kept=batch, removed_detection_ids=())

    if mode == "dualgate":
        output = backend.apply_nms_dualgate(
            batch.boxes_local,
            batch.scores,
            confidence,
            use_concentric=use_concentric,
            masks=list(batch.masks) if batch.masks else None,
            return_indices=True,
            gate_mode=gate_mode,
            iou_threshold=iou_threshold,
            iom_threshold=iom_threshold,
        )
    elif mode == "iou":
        output = backend.apply_nms(
            batch.boxes_local,
            batch.scores,
            confidence,
            iou_threshold=iou_threshold,
            return_indices=True,
        )
    else:
        raise StageError(f"Unsupported suppression mode: {mode!r}")

    if not isinstance(output, tuple) or len(output) != 3:
        raise StageError("NMS backend must return boxes, scores, and source indices")
    _, _, keep_indices_raw = output
    keep_indices = tuple(int(i) for i in np.asarray(keep_indices_raw).reshape((-1,)))
    kept = batch.subset(keep_indices)
    keep_set = set(keep_indices)
    removed_indices = tuple(index for index in range(len(batch)) if index not in keep_set)
    removed_ids = tuple(
        batch.detection_ids[index] for index in removed_indices
    ) if batch.detection_ids else ()

    comparisons: list[DedupComparisonRecord] = []
    if context and context.sink is not None and batch.detection_ids:
        for removed_index in removed_indices:
            survivor_index, metrics = _best_survivor(
                batch, removed_index, keep_indices
            )
            survivor_id = (
                batch.detection_ids[survivor_index]
                if survivor_index is not None
                else None
            )
            record = DedupComparisonRecord(
                dedup_decision_id=context.new_id(EntityKind.DEDUP_DECISION),  # type: ignore[arg-type]
                pass_id=context.pass_id,
                new_detection_id=batch.detection_ids[removed_index],
                existing_detection_id=survivor_id,
                existing_node_id=None,
                stage="intra_pass_suppression",
                metrics=metrics,
                thresholds={
                    "confidence": float(confidence),
                    "iou": float(iou_threshold),
                    "iom": float(iom_threshold),
                },
                decision=DedupDecision.SUPPRESS_NEW,
                reason=f"removed by {mode}:{gate_mode}",
                selected_survivor_id=survivor_id,
            )
            comparisons.append(record)
            context.emit(record)

    return SuppressionResult(
        kept=kept,
        removed_detection_ids=removed_ids,
        comparisons=tuple(comparisons),
    )



def register_detection_batch(
    batch: DetectionBatch,
    graph: Any,
    *,
    pass_number: int,
    dedup_metric: str = "iou",
    dedup_threshold: float = 0.40,
    match_classes: tuple[str, ...] = ("fruit", "unresolved"),
    signature: str | None = None,
    class_names: tuple[str, ...] = ("fruit", "leaf", "background"),
    context: StageContext | None = None,
) -> RegistrationResult:
    """Associate detections with graph nodes and log only final assignments.

    Earlier versions emitted one record for every detection/node comparison and
    embedded full node snapshots in each registration.  That made a single pass
    grow quadratically with the number of objects.  The compact contract records
    only the comparison that actually caused a merge, plus one final registration
    outcome for every detection.
    """

    del class_names  # retained in the public signature for compatibility
    if dedup_metric not in {"iou", "iom", "mask_iou"}:
        raise StageError(f"Unsupported cross-pass metric: {dedup_metric!r}")

    created: list[str] = []
    updated: list[str] = []
    rejected: list[str] = []
    decisive_comparisons: list[DedupComparisonRecord] = []
    decisions: list[RegistrationDecisionRecord] = []

    for index in range(len(batch)):
        box = batch.boxes_global[index]
        mask = batch.masks[index] if batch.masks else None
        detection_id = batch.detection_ids[index] if batch.detection_ids else None
        matched_node = None
        matched_metrics: dict[str, float] | None = None
        matched_rank: int | None = None

        for rank, node in enumerate(graph.nodes.values()):
            if getattr(node, "classification", "unresolved") not in match_classes:
                continue
            node_box = getattr(node, "bbox", getattr(node, "box", None))
            if node_box is None:
                continue
            node_mask = getattr(node, "mask", None)
            metrics = _pair_metrics(box, mask, node_box, node_mask)
            metric_name = (
                "mask_iou"
                if dedup_metric == "mask_iou"
                and mask is not None
                and node_mask is not None
                else dedup_metric
            )
            if metrics[metric_name] > dedup_threshold:
                matched_node = node
                matched_metrics = metrics
                matched_rank = rank
                break

        selected_dedup_id = None
        if (
            matched_node is not None
            and context is not None
            and context.sink is not None
            and detection_id is not None
        ):
            selected_dedup_id = context.new_id(EntityKind.DEDUP_DECISION)
            comparison = DedupComparisonRecord(
                dedup_decision_id=selected_dedup_id,  # type: ignore[arg-type]
                pass_id=context.pass_id,
                new_detection_id=detection_id,
                existing_detection_id=None,
                existing_node_id=matched_node.id,
                stage="cross_pass_registration",
                metrics=dict(matched_metrics or {}),
                thresholds={dedup_metric: float(dedup_threshold)},
                decision=DedupDecision.MERGE,
                reason="selected cross-pass match",
                selected_survivor_id=matched_node.id,
                comparison_rank=matched_rank,
            )
            decisive_comparisons.append(comparison)
            context.emit(comparison)

        if matched_node is not None:
            registration_id = (
                context.new_id(EntityKind.REGISTRATION)
                if context and context.sink is not None and detection_id is not None
                else None
            )
            matched_node.reinforce(
                box,
                signature,
                source_detection_id=detection_id,
                found_in_pass=pass_number,
                dedup_decision_id=selected_dedup_id,
                registration_id=registration_id,
            )
            updated.append(matched_node.id)
            if detection_id is not None:
                rejected.append(detection_id)
            if registration_id is not None and detection_id is not None:
                decision = RegistrationDecisionRecord(
                    registration_id=registration_id,
                    pass_id=context.pass_id,
                    raw_detection_id=detection_id,
                    outcome=RegistrationOutcome.UPDATED_NODE,
                    graph_node_id=matched_node.id,
                    candidate_node_ids=(matched_node.id,),
                    selected_dedup_decision_id=selected_dedup_id,
                    reason="assigned to existing graph object",
                    node_before=None,
                    node_after=None,
                )
                decisions.append(decision)
                context.emit(decision)
            continue

        node_id = (
            context.new_id(EntityKind.GRAPH_NODE)
            if context and context.sink is not None
            else None
        )
        registration_id = (
            context.new_id(EntityKind.REGISTRATION)
            if context and context.sink is not None and detection_id is not None
            else None
        )
        node_id = graph.add_candidate(
            box,
            float(batch.scores[index]),
            found_in_pass=pass_number,
            node_id=node_id,
            source_detection_id=detection_id,
        )
        node = graph.nodes[node_id]
        if signature is not None:
            node.signatures.add(signature)
        if registration_id is not None:
            node.registration_ids.append(registration_id)
        if mask is not None:
            node.mask = mask
        created.append(node_id)
        if registration_id is not None and detection_id is not None:
            decision = RegistrationDecisionRecord(
                registration_id=registration_id,
                pass_id=context.pass_id,
                raw_detection_id=detection_id,
                outcome=RegistrationOutcome.CREATED_NODE,
                graph_node_id=node_id,
                candidate_node_ids=(),
                selected_dedup_decision_id=None,
                reason="created a new graph object",
                node_before=None,
                node_after=None,
            )
            decisions.append(decision)
            context.emit(decision)

    return RegistrationResult(
        created_node_ids=tuple(created),
        updated_node_ids=tuple(updated),
        rejected_detection_ids=tuple(rejected),
        comparisons=tuple(decisive_comparisons),
        decisions=tuple(decisions),
    )


def _node_snapshot(node: Any, *, pass_id: str, class_names: tuple[str, ...]) -> GraphNodeSnapshotRecord:
    belief = getattr(node, "belief", None)
    if belief is not None:
        prior_map = belief.prior_mapping()
        posterior_map = belief.as_mapping()
        status = NodeStatus(belief.status)
        final = belief.decision
        temporary = belief.map_class
        query_count = belief.query_count
    else:
        value = 1.0 / len(class_names)
        prior_map = {name: value for name in class_names}
        posterior_map = dict(prior_map)
        status = NodeStatus.UNRESOLVED
        final = None
        temporary = max(posterior_map, key=posterior_map.get)
        query_count = 0
    found = getattr(node, "found_in_passes", None)
    if found is None:
        found = (int(getattr(node, "found_in_pass", 0)),)
    return GraphNodeSnapshotRecord(
        graph_node_id=str(node.id),
        pass_id=pass_id,
        box=_box_tuple(getattr(node, "bbox", getattr(node, "box"))),
        status=status,
        source_detection_ids=tuple(getattr(node, "source_detection_ids", ())),
        found_in_passes=tuple(int(v) for v in found),
        prior=prior_map,
        posterior=posterior_map,
        temporary_map_class=temporary,
        final_declaration=final,
        query_count=query_count,
        exemplar_eligible_positive=(
            belief is not None
            and belief.status == "stopped"
            and belief.decision == class_names[0]
            and posterior_map.get(class_names[0], 0.0) >= 0.90
        ),
        exemplar_eligible_negative=(
            belief is not None
            and len(class_names) > 1
            and belief.status == "stopped"
            and belief.decision == class_names[1]
            and posterior_map.get(class_names[1], 0.0) >= 0.90
        ),
        legacy_scores={str(k): float(v) for k, v in getattr(node, "scores", {}).items()},
        metadata={
            "legacy_belief_placeholder": belief is None,
            "legacy_classification": getattr(node, "classification", "unresolved"),
            "dedup_decision_ids": list(getattr(node, "dedup_decision_ids", ())),
            "registration_ids": list(getattr(node, "registration_ids", ())),
            "verification_action_ids": list(getattr(node, "verification_action_ids", ())),
            "observation_ids": list(getattr(node, "observation_ids", ())),
            "belief_update_ids": list(getattr(node, "belief_update_ids", ())),
        },
    )



def _best_survivor(
    batch: DetectionBatch,
    removed_index: int,
    survivor_indices: Sequence[int],
) -> tuple[int | None, dict[str, float]]:
    best_index = None
    best_metrics = {"iou": 0.0, "iom": 0.0, "mask_iou": 0.0}
    best_score = -1.0
    for survivor in survivor_indices:
        metrics = _pair_metrics(
            batch.boxes_local[removed_index],
            batch.masks[removed_index] if batch.masks else None,
            batch.boxes_local[survivor],
            batch.masks[survivor] if batch.masks else None,
        )
        score = max(metrics.values())
        if score > best_score:
            best_score = score
            best_index = int(survivor)
            best_metrics = metrics
    return best_index, best_metrics



def _pair_metrics(box_a: Any, mask_a: Any, box_b: Any, mask_b: Any) -> dict[str, float]:
    iou, iom = _box_overlap(box_a, box_b)
    mask_iou = _mask_iou(box_a, mask_a, box_b, mask_b) if mask_a is not None and mask_b is not None else 0.0
    return {"iou": iou, "iom": iom, "mask_iou": mask_iou}



def _box_overlap(box_a: Any, box_b: Any) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    min_area = min(area_a, area_b)
    iom = inter / min_area if min_area > 0 else 0.0
    return float(iou), float(iom)



def _mask_iou(box_a: Any, mask_a: np.ndarray, box_b: Any, mask_b: np.ndarray) -> float:
    ax1, ay1 = int(round(float(box_a[0]))), int(round(float(box_a[1])))
    bx1, by1 = int(round(float(box_b[0]))), int(round(float(box_b[1])))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2 = min(ax1 + mask_a.shape[1], bx1 + mask_b.shape[1])
    iy2 = min(ay1 + mask_a.shape[0], by1 + mask_b.shape[0])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    sub_a = mask_a[iy1 - ay1 : iy2 - ay1, ix1 - ax1 : ix2 - ax1]
    sub_b = mask_b[iy1 - by1 : iy2 - by1, ix1 - bx1 : ix2 - bx1]
    inter = float(np.count_nonzero(np.logical_and(sub_a, sub_b)))
    union = float(np.count_nonzero(mask_a)) + float(np.count_nonzero(mask_b)) - inter
    return inter / union if union > 0 else 0.0



def _translate_boxes(boxes: np.ndarray, region: Box, space: CoordinateSpace) -> np.ndarray:
    result = np.asarray(boxes, dtype=float).reshape((-1, 4)).copy()
    if space in {CoordinateSpace.ROI_LOCAL, CoordinateSpace.TILE_LOCAL} and len(result):
        result[:, [0, 2]] += float(region[0])
        result[:, [1, 3]] += float(region[1])
    return result



def _boxes_to_array(boxes: tuple[Box, ...]) -> np.ndarray | None:
    if not boxes:
        return None
    return np.asarray(boxes, dtype=float).reshape((-1, 4))



def _validate_box(box: Box, name: str) -> None:
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise StageError(f"{name} must contain four finite coordinates")
    if float(box[2]) < float(box[0]) or float(box[3]) < float(box[1]):
        raise StageError(f"{name} must satisfy x2 >= x1 and y2 >= y1")



def _box_tuple(box: Any) -> Box:
    return tuple(float(v) for v in box)  # type: ignore[return-value]


__all__ = [
    "DetectionBatch",
    "RegistrationResult",
    "Sam3QuerySpec",
    "StageContext",
    "StageError",
    "SuppressionResult",
    "execute_sam3_request",
    "normalize_sam3_output",
    "register_detection_batch",
    "suppress_detection_batch",
    "translate_detection_batch",
]

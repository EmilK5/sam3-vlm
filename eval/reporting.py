"""Per-run evaluation, visual artifacts, CSV summaries, and aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from eval.dataset_adapters import CanonicalSample
from eval.visualization import image_to_png_bytes, render_final_overlay, render_pass_timeline
from graph import OrchardGraph
from provenance.contracts import EvaluationResultRecord, RunRecord
from provenance.ids import EntityKind
from provenance.io import atomic_write_json, strict_json_load
from provenance.run_store import ReportingLevel, RunStore
from provenance.schema import ArtifactKind, EventKind, NodeStatus, RunStatus


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = True
    iou_threshold: float = 0.50
    save_final_overlay: bool = True
    save_pass_timeline: bool = True
    summary_csv_name: str = "runs.csv"
    evaluator_name: str = "canonical_detection_count"
    evaluator_version: str = "1.0"

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must lie in [0, 1]")
        if not self.summary_csv_name.strip():
            raise ValueError("summary_csv_name must be non-empty")


@dataclass(frozen=True)
class EvaluationOutputs:
    record: EvaluationResultRecord | None
    overlay_artifact_id: str | None = None
    timeline_artifact_id: str | None = None


def evaluate_and_record(
    *,
    store: RunStore,
    sample: CanonicalSample,
    graph: OrchardGraph,
    predictions: Mapping[str, Any],
    image: Image.Image,
    config: EvaluationConfig,
) -> EvaluationOutputs:
    if not config.enabled:
        return EvaluationOutputs(record=None)
    snapshot = store.current_snapshot(final_predictions=predictions)
    predicted = _prediction_nodes(graph, sample.target_class)
    matches, unmatched_pred, unmatched_gt = _greedy_match(
        predicted,
        sample.ground_truth_boxes,
        config.iou_threshold,
    )
    pool_matches, _, _ = _greedy_match(
        list(graph.nodes.values()),
        sample.ground_truth_boxes,
        config.iou_threshold,
    )
    hard_count = int(predictions.get("hard_count", len(predicted)))
    soft_count = float(predictions.get("soft_count", hard_count))
    gt_count = sample.ground_truth_count
    metrics = _metrics(
        gt_count=gt_count,
        hard_count=hard_count,
        soft_count=soft_count,
        prediction_count=len(predicted),
        ground_truth_box_count=len(sample.ground_truth_boxes),
        match_count=len(matches),
        pool_match_count=len(pool_matches),
        graph=graph,
        target_class=sample.target_class,
        matched_prediction_indices={item[0] for item in matches},
        passes=snapshot.passes,
    )
    per_pass_pool_recall = _per_pass_pool_recall(
        snapshot.passes,
        sample.ground_truth_boxes,
        config.iou_threshold,
    )
    counts = {
        "ground_truth": -1 if gt_count is None else int(gt_count),
        "hard_prediction": hard_count,
        "soft_prediction": soft_count,
        "graph_nodes": len(graph.nodes),
        "matched_predictions": len(matches),
    }
    record = EvaluationResultRecord(
        evaluation_id=store.new_id(EntityKind.EVALUATION),
        run_id=store.run_id,
        image_id=snapshot.dataset_sample.image_id,
        evaluator_name=config.evaluator_name,
        evaluator_version=config.evaluator_version,
        metrics=metrics,
        counts=counts,
        matched_pairs=tuple(
            (predicted[pred_index].id, f"gt_{gt_index:05d}")
            for pred_index, gt_index, _ in matches
        ),
        unmatched_prediction_ids=tuple(predicted[index].id for index in unmatched_pred),
        unmatched_ground_truth_ids=tuple(f"gt_{index:05d}" for index in unmatched_gt),
        metadata={
            "iou_threshold": config.iou_threshold,
            "target_class": sample.target_class,
            "box_metrics_available": bool(sample.ground_truth_boxes),
            "count_metric_available": gt_count is not None,
            "per_pass_pool_recall": per_pass_pool_recall,
            "stopping_reason_counts": _stopping_reason_counts(snapshot.passes),
        },
    )
    store.append(record, event_kind=EventKind.EVALUATION, minimum_level=ReportingLevel.MINIMAL)

    overlay_id = None
    if config.save_final_overlay:
        overlay = render_final_overlay(
            image,
            sample=sample,
            nodes=graph.nodes.values(),
            target_class=sample.target_class,
            predictions=predictions,
        )
        artifact = store.write_artifact_bytes(
            image_to_png_bytes(overlay),
            kind=ArtifactKind.OVERLAY,
            relative_path="visualizations/final_overlay.png",
            media_type="image/png",
            width=overlay.width,
            height=overlay.height,
            metadata={"role": "final_overlay", "evaluator": config.evaluator_name},
            minimum_level=ReportingLevel.MINIMAL,
        )
        overlay_id = artifact.artifact_id

    timeline_id = None
    if config.save_pass_timeline and snapshot.passes:
        timeline = render_pass_timeline(image, snapshot.passes, target_class=sample.target_class)
        artifact = store.write_artifact_bytes(
            image_to_png_bytes(timeline),
            kind=ArtifactKind.OVERLAY,
            relative_path="visualizations/pass_timeline.png",
            media_type="image/png",
            width=timeline.width,
            height=timeline.height,
            metadata={"role": "pass_timeline", "pass_count": len(snapshot.passes)},
            minimum_level=ReportingLevel.STANDARD,
        )
        timeline_id = artifact.artifact_id
    return EvaluationOutputs(record, overlay_id, timeline_id)


def _prediction_nodes(graph: OrchardGraph, target_class: str) -> list:
    selected = []
    for node in graph.nodes.values():
        declaration = node.final_declaration
        if declaration is None:
            declaration = node.temporary_map_class
        if declaration is None:
            declaration = node.classification
        if declaration == target_class or declaration == "unresolved":
            selected.append(node)
    return selected


def _metrics(
    *,
    gt_count: int | None,
    hard_count: int,
    soft_count: float,
    prediction_count: int,
    ground_truth_box_count: int,
    match_count: int,
    pool_match_count: int,
    graph: OrchardGraph,
    target_class: str,
    matched_prediction_indices: set[int],
    passes: Sequence,
) -> dict[str, float | int | None]:
    precision = recall = f1 = None
    if ground_truth_box_count > 0:
        precision = match_count / prediction_count if prediction_count else 0.0
        recall = match_count / ground_truth_box_count
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    absolute_error = squared_error = normalized_absolute_error = exact = None
    soft_absolute_error = None
    if gt_count is not None:
        error = hard_count - gt_count
        absolute_error = abs(error)
        squared_error = error * error
        normalized_absolute_error = abs(error) / max(1, gt_count)
        exact = int(error == 0)
        soft_absolute_error = abs(soft_count - gt_count)
    pool_recall = None
    if ground_truth_box_count > 0:
        pool_recall = pool_match_count / ground_truth_box_count
    probabilities = []
    labels = []
    predicted = _prediction_nodes(graph, target_class)
    if ground_truth_box_count > 0:
        for index, node in enumerate(predicted):
            if node.posterior is None:
                continue
            probabilities.append(float(node.posterior.get(target_class, 0.0)))
            labels.append(1.0 if index in matched_prediction_indices else 0.0)
    brier = None
    nll = None
    if probabilities:
        brier = mean((p - y) ** 2 for p, y in zip(probabilities, labels))
        epsilon = 1e-9
        nll = mean(-y * math.log(max(epsilon, p)) - (1-y) * math.log(max(epsilon, 1-p)) for p, y in zip(probabilities, labels))
    eig = [item.expected_information_gain for record in passes for item in record.information_gain_records if item.selected]
    realized = [record.belief_update.realized_kl for record in passes if record.belief_update is not None]
    last_cost = passes[-1].cost_after if passes else None
    query_counts = [node.query_count for node in graph.nodes.values()]
    unresolved = sum(
        1
        for node in graph.nodes.values()
        if node.belief is None or node.belief.status != "stopped"
    )
    return {
        "count_error": None if gt_count is None else hard_count - gt_count,
        "absolute_count_error": absolute_error,
        "squared_count_error": squared_error,
        "normalized_absolute_count_error": normalized_absolute_error,
        "soft_absolute_count_error": soft_absolute_error,
        "exact_count": exact,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate_pool_recall": pool_recall,
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "pass_count": len(passes),
        "selected_eig_mean": mean(eig) if eig else None,
        "selected_eig_total": sum(eig) if eig else 0.0,
        "realized_information_mean": mean(realized) if realized else None,
        "realized_information_total": sum(realized) if realized else 0.0,
        "average_queries_per_node": mean(query_counts) if query_counts else 0.0,
        "unresolved_fraction": unresolved / len(graph.nodes) if graph.nodes else 0.0,
        "sam3_calls": None if last_cost is None else last_cost.sam3_calls,
        "qwen_calls": None if last_cost is None else last_cost.qwen_calls,
        "tile_calls": None if last_cost is None else last_cost.tile_calls,
        "runtime_seconds": None if last_cost is None else last_cost.runtime_seconds,
        "normalized_cost": None if last_cost is None else last_cost.normalized_cost,
    }


def _per_pass_pool_recall(passes: Sequence, ground_truth: Sequence, threshold: float):
    if not ground_truth:
        return []
    rows = []
    for record in passes:
        matches, _, _ = _greedy_match(record.graph_after, ground_truth, threshold)
        rows.append(
            {
                "pass_index": record.pass_index,
                "recall": len(matches) / len(ground_truth),
                "graph_nodes": len(record.graph_after),
            }
        )
    return rows


def _stopping_reason_counts(passes: Sequence) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in passes:
        if record.stopping_decision is None:
            continue
        key = record.stopping_decision.reason.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _greedy_match(predicted: Sequence, ground_truth: Sequence, threshold: float):
    candidates = []
    for pred_index, node in enumerate(predicted):
        for gt_index, box in enumerate(ground_truth):
            overlap = _iou(tuple(node.box), tuple(box))
            if overlap >= threshold:
                candidates.append((overlap, pred_index, gt_index))
    candidates.sort(reverse=True)
    used_pred, used_gt, matches = set(), set(), []
    for overlap, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, overlap))
    return matches, sorted(set(range(len(predicted))) - used_pred), sorted(set(range(len(ground_truth))) - used_gt)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def summary_row(run: RunRecord, run_directory: str | Path) -> dict[str, Any]:
    evaluation = run.evaluations[-1] if run.evaluations else None
    metrics = {} if evaluation is None else dict(evaluation.metrics)
    return {
        "run_id": run.run_id,
        "run_directory": str(Path(run_directory)),
        "status": run.status.value,
        "dataset": run.dataset_sample.dataset_name,
        "dataset_version": run.dataset_sample.dataset_version,
        "split": run.dataset_sample.split,
        "sample_index": run.dataset_sample.sample_index,
        "sample_key": run.dataset_sample.metadata.get("sample_key"),
        "image_sha256": run.dataset_sample.image_sha256,
        "pipeline_mode": run.metadata.get("pipeline_mode"),
        "config_sha256": run.config_sha256,
        "ground_truth_count": run.dataset_sample.ground_truth_count,
        "hard_count": run.final_predictions.get("hard_count"),
        "soft_count": run.final_predictions.get("soft_count"),
        "pass_count": len(run.passes),
        "sam3_queries": run.final_predictions.get("total_sam3_queries"),
        **metrics,
    }


def append_summary_csv(csv_path: str | Path, run: RunRecord, run_directory: str | Path) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = summary_row(run, run_directory)
    existing = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    by_run = {item.get("run_id"): item for item in existing}
    by_run[run.run_id] = {key: "" if value is None else value for key, value in row.items()}
    fields = list(row)
    for item in existing:
        for key in item:
            if key not in fields:
                fields.append(key)
    _atomic_csv(path, fields, by_run.values())
    return path


def aggregate_runs(run_directories: Iterable[str | Path]) -> dict[str, Any]:
    rows = []
    for directory in run_directories:
        run_path = Path(directory) / "run.json"
        if not run_path.is_file():
            continue
        run = RunRecord.from_dict(strict_json_load(run_path))
        if run.status is not RunStatus.SUCCEEDED:
            continue
        rows.append(summary_row(run, directory))
    numeric = {}
    for key in sorted({key for row in rows for key in row}):
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            numeric[key] = {"mean": mean(values), "min": min(values), "max": max(values), "count": len(values)}
    return {"run_count": len(rows), "rows": rows, "aggregate_metrics": numeric}


def write_aggregate(output_path: str | Path, run_directories: Iterable[str | Path]) -> Path:
    path = Path(output_path)
    atomic_write_json(path, aggregate_runs(run_directories))
    return path


def write_aggregate_csv(output_path: str | Path, run_directories: Iterable[str | Path]) -> Path:
    path = Path(output_path)
    payload = aggregate_runs(run_directories)
    rows = payload["rows"]
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(path, fields, rows)
    return path


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


__all__ = [
    "EvaluationConfig",
    "EvaluationOutputs",
    "aggregate_runs",
    "append_summary_csv",
    "evaluate_and_record",
    "summary_row",
    "write_aggregate",
    "write_aggregate_csv",
]

"""Testable service layer behind the Gradio experiment dashboard."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from eval.dataset_adapters import CanonicalSample, DatasetRegistry
from eval.reporting import summary_row
from eval.run_index import IndexedRun, RunIndex
from eval.visualization import load_run_image
from experiments.config import ExperimentMode, UnifiedExperimentConfig
from experiments.runner import UnifiedExperimentRunner, UnifiedRuntime
from provenance.contracts import RunRecord
from provenance.io import strict_json_load


@dataclass(frozen=True)
class DashboardRunRequest:
    dataset_name: str
    split: str
    sample_index: int
    mode: ExperimentMode
    target_concept: str | None = None
    run_name: str | None = None
    reporting_level: str = "full"
    max_passes: int | None = None
    max_sam3_calls: int | None = None
    max_qwen_calls: int | None = None
    stopping_confidence: float | None = None
    tiling_mode: str | None = None
    expert_overrides: Mapping[str, Any] = field(default_factory=dict)


ConfigFactory = Callable[[DashboardRunRequest, CanonicalSample], UnifiedExperimentConfig]


@dataclass(frozen=True)
class DashboardDependencies:
    datasets: DatasetRegistry
    runtime: UnifiedRuntime
    config_factory: ConfigFactory
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))


@dataclass(frozen=True)
class DashboardView:
    run_directory: Path
    run: RunRecord
    image: Image.Image
    summary: Mapping[str, Any]
    pass_rows: tuple[Mapping[str, Any], ...]
    node_rows: tuple[Mapping[str, Any], ...]
    qwen_rows: tuple[Mapping[str, Any], ...]
    information_rows: tuple[Mapping[str, Any], ...]
    tiling_rows: tuple[Mapping[str, Any], ...]


class DashboardService:
    def __init__(self, dependencies: DashboardDependencies) -> None:
        self.dependencies = dependencies

    def dataset_names(self) -> tuple[str, ...]:
        return self.dependencies.datasets.names()

    def splits(self, dataset_name: str) -> tuple[str, ...]:
        return self.dependencies.datasets.adapter(dataset_name).splits()

    def sample_count(self, dataset_name: str, split: str) -> int:
        return self.dependencies.datasets.adapter(dataset_name).size(split)

    def load_sample(self, dataset_name: str, split: str, index: int) -> tuple[Image.Image, Mapping[str, Any]]:
        sample = self.dependencies.datasets.get(dataset_name, split, int(index))
        return sample.load_image(), _sample_summary(sample)

    def run(self, request: DashboardRunRequest) -> DashboardView:
        sample = self.dependencies.datasets.get(
            request.dataset_name, request.split, int(request.sample_index)
        )
        if request.target_concept and request.target_concept.strip():
            sample = dataclasses.replace(sample, target_concept=request.target_concept.strip())
        config = self.dependencies.config_factory(request, sample)
        result = UnifiedExperimentRunner(
            datasets=self.dependencies.datasets,
            runtime=self.dependencies.runtime,
            config=config,
        ).run_sample(sample)
        return self.replay(result.run_directory)

    def available_runs(self, *, successful_only: bool = False) -> tuple[Path, ...]:
        return RunIndex(self.dependencies.output_root).directories(
            successful_only=successful_only
        )

    def replay(self, run_directory: str | Path) -> DashboardView:
        directory = Path(run_directory)
        run = RunRecord.from_dict(strict_json_load(directory / "run.json"))
        image = load_run_image(directory, run.dataset_sample.image_artifact.relative_path)
        overlay = _artifact_for_role(run, "final_overlay")
        if overlay is not None:
            image = load_run_image(directory, overlay.relative_path)
        return DashboardView(
            run_directory=directory,
            run=run,
            image=image,
            summary=summary_row(run, directory),
            pass_rows=tuple(_pass_row(item) for item in run.passes),
            node_rows=tuple(_node_row(item) for item in run.final_graph),
            qwen_rows=tuple(_qwen_row(record, item) for record in run.passes for item in record.qwen_calls),
            information_rows=tuple(
                _information_row(record, item)
                for record in run.passes
                for item in record.information_gain_records
            ),
            tiling_rows=tuple(
                _tiling_row(record)
                for record in run.passes
                if record.tiling_decision is not None
            ),
        )

    def pass_view(self, run_directory: str | Path, pass_index: int) -> tuple[Image.Image, Mapping[str, Any]]:
        view = self.replay(run_directory)
        if not view.run.passes:
            return view.image, {"message": "run has no passes"}
        index = max(0, min(int(pass_index), len(view.run.passes) - 1))
        record = view.run.passes[index]
        source = load_run_image(
            view.run_directory,
            view.run.dataset_sample.image_artifact.relative_path,
        )
        from eval.visualization import render_pass_timeline

        panel = render_pass_timeline(source, (record,), target_class=view.run.dataset_sample.target_class)
        return panel, _pass_row(record)

    def compare(self, run_directories: list[str | Path]) -> tuple[Mapping[str, Any], ...]:
        rows = []
        for directory in run_directories:
            view = self.replay(directory)
            rows.append(view.summary)
        return tuple(rows)

    def raw_json(self, run_directory: str | Path) -> str:
        return json.dumps(strict_json_load(Path(run_directory) / "run.json"), indent=2, sort_keys=True)


def _artifact_for_role(run: RunRecord, role: str):
    return next((item for item in run.artifacts if item.metadata.get("role") == role), None)


def _sample_summary(sample: CanonicalSample) -> Mapping[str, Any]:
    return {
        "dataset": sample.dataset_name,
        "version": sample.dataset_version,
        "split": sample.split,
        "sample_key": sample.sample_key,
        "sample_index": sample.sample_index,
        "target_concept": sample.target_concept,
        "target_class": sample.target_class,
        "confounders": list(sample.confounder_classes),
        "ground_truth_count": sample.ground_truth_count,
        "ground_truth_boxes": len(sample.ground_truth_boxes),
        "capabilities": sample.capabilities.as_dict(),
        "metadata": dict(sample.metadata),
    }


def _pass_row(record) -> Mapping[str, Any]:
    return {
        "pass_index": record.pass_index,
        "pass_id": record.pass_id,
        "target_node_id": record.target_node_id,
        "selected_action_id": record.selected_action_id,
        "qwen_calls": len(record.qwen_calls),
        "sam3_calls": len(record.sam3_calls),
        "tiles": len(record.tiles),
        "raw_detections": len(record.raw_detections),
        "registrations": len(record.registrations),
        "graph_nodes": len(record.graph_after),
        "continuation_reason": record.continuation_reason,
        "warnings": list(record.warnings),
    }


def _node_row(node) -> Mapping[str, Any]:
    return {
        "node_id": node.graph_node_id,
        "box": list(node.box),
        "status": node.status.value,
        "posterior": dict(node.posterior),
        "temporary_map_class": node.temporary_map_class,
        "final_declaration": node.final_declaration,
        "query_count": node.query_count,
        "source_detection_count": len(node.source_detection_ids),
        "found_in_passes": list(node.found_in_passes),
    }


def _qwen_row(pass_record, call) -> Mapping[str, Any]:
    return {
        "pass_index": pass_record.pass_index,
        "qwen_call_id": call.qwen_call_id,
        "model": call.model_id,
        "latency_seconds": call.latency_seconds,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "validation_errors": list(call.validation_errors),
        "fallback_used": call.fallback_used,
        "reasoning_summary": call.reasoning_summary,
    }


def _information_row(pass_record, record) -> Mapping[str, Any]:
    return {
        "pass_index": pass_record.pass_index,
        "action_id": record.action_id,
        "expected_information_gain": record.expected_information_gain,
        "expected_cost": record.expected_cost,
        "cost_adjusted_score": record.cost_adjusted_score,
        "selected": record.selected,
        "rank": record.rank,
        "rejection_reason": record.rejection_reason,
    }


def _tiling_row(pass_record) -> Mapping[str, Any]:
    record = pass_record.tiling_decision
    return {
        "pass_index": pass_record.pass_index,
        "mode": record.mode.value,
        "triggered": record.triggered,
        "trigger_reason": record.reason,
        "density_metrics": dict(record.density_metrics),
        "tiling_rule": record.tiling_rule,
        "tile_size": record.tile_size,
        "selected_tile_count": record.selected_tile_count,
        "tile_count": len(pass_record.tiles),
    }


__all__ = [
    "DashboardDependencies",
    "DashboardRunRequest",
    "DashboardService",
    "DashboardView",
]

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from dashboard.service import DashboardDependencies, DashboardService
from eval.dataset_adapters import DatasetRegistry, GenericFolderAdapter
from eval.reporting import aggregate_runs, write_aggregate_csv
from eval.run_index import RunIndex
from eval.sweep import run_sweep
from experiments.config import DiscoveryPassSpec, ExperimentMode, UnifiedExperimentConfig
from experiments.runner import UnifiedExperimentRunner, UnifiedRuntime
from provenance.ids import create_run_id
from provenance.schema import RunStatus


class FakeBackend:
    def run_raw_inference(
        self,
        processor,
        image_np,
        confidence,
        *,
        prompt,
        pos_boxes,
        neg_boxes,
        disable_size_filter,
        return_masks,
    ):
        boxes = np.array([[2.0, 2.0, 14.0, 14.0]])
        scores = np.array([0.9])
        if return_masks:
            return boxes, scores, [np.ones(image_np.shape[:2], dtype=bool)]
        return boxes, scores

    def apply_nms_dualgate(self, boxes, scores, confidence, **kwargs):
        keep = np.where(np.asarray(scores) >= confidence)[0]
        return np.asarray(boxes)[keep], np.asarray(scores)[keep], keep

    def apply_nms(self, boxes, scores, confidence, **kwargs):
        keep = np.where(np.asarray(scores) >= confidence)[0]
        return np.asarray(boxes)[keep], np.asarray(scores)[keep], keep


def _setup(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (32, 32), "white").save(data / "sample.png")
    (data / "sample.json").write_text(
        '{"count": 1, "boxes": [[2, 2, 14, 14]]}', encoding="utf-8"
    )
    registry = DatasetRegistry(
        [GenericFolderAdapter(data, target_concept="object", target_class="target")]
    )
    runtime = UnifiedRuntime(
        backend=FakeBackend(),
        processor=object(),
        run_id_factory=lambda: create_run_id(
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
            random_token="phase1112",
        ),
    )
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_SINGLE_PASS,
        output_root=tmp_path / "runs",
        class_names=("target", "non_target", "background"),
        target_class="target",
        discovery_passes=(DiscoveryPassSpec(prompt="object", threshold=0.2),),
    )
    return registry, runtime, config


def test_runner_records_evaluation_visualizations_and_csv(tmp_path):
    registry, runtime, config = _setup(tmp_path)
    result = UnifiedExperimentRunner(
        datasets=registry, runtime=runtime, config=config
    ).run("generic", "all", 0)
    assert result.run.status is RunStatus.SUCCEEDED
    assert len(result.run.evaluations) == 1
    metrics = result.run.evaluations[0].metrics
    assert metrics["absolute_count_error"] == 0
    assert metrics["precision"] == 1.0
    assert (result.run_directory / "artifacts/visualizations/final_overlay.png").is_file()
    assert (result.run_directory / "artifacts/visualizations/pass_timeline.png").is_file()
    assert (config.output_root / "runs.csv").is_file()


def test_run_index_dashboard_replay_and_aggregation(tmp_path):
    registry, runtime, config = _setup(tmp_path)
    result = UnifiedExperimentRunner(
        datasets=registry, runtime=runtime, config=config
    ).run("generic", "all", 0)
    indexed = RunIndex(config.output_root).scan(successful_only=True)
    assert len(indexed) == 1
    sample = registry.get("generic", "all", 0)
    found = RunIndex(config.output_root).find_complete(
        dataset_name="generic",
        split="all",
        sample_key=sample.sample_key,
        image_sha256=sample.image_sha256,
        config_sha256=result.run.config_sha256,
    )
    assert found is not None

    dependencies = DashboardDependencies(
        datasets=registry,
        runtime=runtime,
        config_factory=lambda request, sample: config,
        output_root=config.output_root,
    )
    service = DashboardService(dependencies)
    replay = service.replay(result.run_directory)
    assert replay.summary["hard_count"] == 1
    assert len(replay.pass_rows) == 1
    assert replay.image.size == (32, 32)
    pass_image, pass_row = service.pass_view(result.run_directory, 0)
    assert pass_image.size == (32, 32)
    assert pass_row["pass_index"] == 0
    assert len(service.compare([result.run_directory])) == 1

    aggregate = aggregate_runs([result.run_directory])
    assert aggregate["run_count"] == 1
    output_csv = write_aggregate_csv(tmp_path / "aggregate.csv", [result.run_directory])
    assert output_csv.is_file()


def test_resume_safe_sweep_skips_matching_successful_run(tmp_path):
    registry, runtime, config = _setup(tmp_path)
    runner = UnifiedExperimentRunner(datasets=registry, runtime=runtime, config=config)
    first = run_sweep(
        runner,
        dataset_name="generic",
        split="all",
        indices=[0],
        resume=True,
    )
    assert len(first.completed) == 1
    second = run_sweep(
        runner,
        dataset_name="generic",
        split="all",
        indices=[0],
        resume=True,
    )
    assert len(second.completed) == 0
    assert len(second.skipped_run_directories) == 1

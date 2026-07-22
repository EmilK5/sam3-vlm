from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from eval.dataset_adapters import DatasetRegistry, GenericFolderAdapter
from experiments.config import DiscoveryPassSpec, ExperimentMode, UnifiedExperimentConfig
from experiments.runner import UnifiedExperimentRunner, UnifiedRuntime
from provenance.events import EventLogReader
from provenance.ids import create_run_id
from provenance.run_store import ReportingLevel


class ManyDetectionBackend:
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
        boxes = []
        for row in range(10):
            for column in range(12):
                x1 = float(column * 5)
                y1 = float(row * 5)
                boxes.append([x1, y1, x1 + 3.0, y1 + 3.0])
        scores = np.linspace(0.99, 0.70, len(boxes))
        if return_masks:
            masks = [np.zeros(image_np.shape[:2], dtype=bool) for _ in boxes]
            return np.asarray(boxes), scores, masks
        return np.asarray(boxes), scores

    def apply_nms_dualgate(
        self,
        boxes,
        scores,
        confidence,
        *,
        use_concentric,
        masks,
        return_indices,
        gate_mode,
        iou_threshold,
        iom_threshold,
    ):
        keep = np.arange(len(scores))
        return np.asarray(boxes), np.asarray(scores), keep

    def apply_nms(self, boxes, scores, confidence, *, iou_threshold, return_indices):
        keep = np.arange(len(scores))
        return np.asarray(boxes), np.asarray(scores), keep


def test_single_pass_logging_is_compact_and_lifecycle_focused(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (64, 64), (20, 30, 40)).save(image_root / "sample.png")
    registry = DatasetRegistry(
        [
            GenericFolderAdapter(
                image_root,
                target_concept="object",
                target_class="target",
            )
        ]
    )
    runtime = UnifiedRuntime(
        backend=ManyDetectionBackend(),
        processor=object(),
        run_id_factory=lambda: create_run_id(
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
            random_token="compact",
        ),
    )
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_SINGLE_PASS,
        output_root=tmp_path / "runs",
        reporting_level=ReportingLevel.FULL,
        fsync=False,
        class_names=("target", "non_target", "background"),
        target_class="target",
        discovery_passes=(
            DiscoveryPassSpec(prompt="object", threshold=0.2, return_masks=False),
        ),
    )

    result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=runtime,
        config=config,
    ).run("generic", "all", 0)

    run_path = result.run_directory / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    events = EventLogReader(result.run_directory / "events.jsonl").read_all()
    pass_record = result.run.passes[0]

    assert payload["events"] == []
    assert result.run.events == ()
    assert len(events) < 10
    assert len(pass_record.raw_detections) == 120
    assert len(pass_record.registrations) == 120
    assert pass_record.graph_before == ()
    assert pass_record.graph_after == ()
    assert len(result.run.final_graph) == 120
    assert pass_record.dedup_comparisons == ()
    assert all(item.node_before is None for item in pass_record.registrations)
    assert all(item.node_after is None for item in pass_record.registrations)
    assert all(item.candidate_node_ids == () for item in pass_record.registrations)
    assert run_path.stat().st_size < 1_000_000
    assert len(run_path.read_text(encoding="utf-8").splitlines()) < 15_000

    lifecycle = result.run.final_graph[0].metadata["lifecycle"]
    assert lifecycle["created_in_pass"] == 1
    assert lifecycle["created_from_detection_id"] is not None

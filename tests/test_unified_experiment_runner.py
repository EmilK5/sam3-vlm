from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from agent.asht.tiling import AdaptiveTilingConfig
from eval.dataset_adapters import DatasetRegistry, GenericFolderAdapter
from experiments.config import DiscoveryPassSpec, ExperimentMode, UnifiedExperimentConfig
from experiments.runner import UnifiedExperimentRunner, UnifiedRuntime
from provenance.ids import create_run_id
from provenance.run_store import ReportingLevel
from provenance.schema import RunStatus, TilingMode


class FakeBackend:
    def __init__(self):
        self.calls = 0

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
        self.calls += 1
        height, width = image_np.shape[:2]
        boxes = np.array([[1.0, 1.0, min(12.0, width), min(12.0, height)]])
        scores = np.array([0.9])
        if return_masks:
            masks = [np.ones((height, width), dtype=bool)]
            return boxes, scores, masks
        return boxes, scores

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
        keep = np.where(np.asarray(scores) >= confidence)[0]
        return np.asarray(boxes)[keep], np.asarray(scores)[keep], keep

    def apply_nms(self, boxes, scores, confidence, *, iou_threshold, return_indices):
        keep = np.where(np.asarray(scores) >= confidence)[0]
        return np.asarray(boxes)[keep], np.asarray(scores)[keep], keep


def _registry(tmp_path):
    image_dir = tmp_path / "dataset"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), (30, 50, 70)).save(image_dir / "sample.png")
    adapter = GenericFolderAdapter(
        image_dir,
        target_concept="object",
        target_class="target",
    )
    return DatasetRegistry([adapter])


def _runtime(backend):
    return UnifiedRuntime(
        backend=backend,
        processor=object(),
        run_id_factory=lambda: create_run_id(
            now=datetime(2026, 7, 22, tzinfo=timezone.utc),
            random_token="phase910",
        ),
    )


def test_unified_single_pass_creates_self_contained_run(tmp_path):
    backend = FakeBackend()
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_SINGLE_PASS,
        output_root=tmp_path / "runs",
        reporting_level=ReportingLevel.FULL,
        class_names=("target", "non_target", "background"),
        target_class="target",
        discovery_passes=(DiscoveryPassSpec(prompt="object", threshold=0.2),),
    )
    result = UnifiedExperimentRunner(
        datasets=_registry(tmp_path), runtime=_runtime(backend), config=config
    ).run("generic", "all", 0)
    assert result.run.status is RunStatus.SUCCEEDED
    assert len(result.run.passes) == 1
    assert result.run.final_predictions["hard_count"] == 1
    assert (result.run_directory / "run.json").is_file()
    assert (result.run_directory / "artifacts" / "input" / "sample.png").is_file()
    assert (result.run_directory / "artifacts" / "graph" / "final_graph.json").is_file()
    assert result.run.dataset_sample.metadata["capabilities"]["boxes"] is False


def test_unified_fixed_multipass_uses_one_store_and_deduplicates(tmp_path):
    backend = FakeBackend()
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_FIXED_MULTIPASS,
        output_root=tmp_path / "runs",
        class_names=("target", "non_target", "background"),
        target_class="target",
        discovery_passes=(
            DiscoveryPassSpec(prompt="object", threshold=0.2, signature="one"),
            DiscoveryPassSpec(prompt="round object", threshold=0.2, signature="two"),
        ),
    )
    result = UnifiedExperimentRunner(
        datasets=_registry(tmp_path), runtime=_runtime(backend), config=config
    ).run("generic", "all", 0)
    assert len(result.run.passes) == 2
    assert len(result.graph.nodes) == 1
    node = next(iter(result.graph.nodes.values()))
    assert node.support == 2
    assert result.run.final_predictions["total_sam3_queries"] == 2


def test_unified_adaptive_tiling_records_decision_and_tiles(tmp_path):
    backend = FakeBackend()
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_ADAPTIVE_TILING,
        output_root=tmp_path / "runs",
        class_names=("target", "non_target", "background"),
        target_class="target",
        discovery_passes=(
            DiscoveryPassSpec(
                prompt="object",
                threshold=0.2,
                tiling_mode=TilingMode.ALWAYS,
            ),
        ),
        adaptive_tiling=AdaptiveTilingConfig(
            min_tile_size=16,
            max_tile_size=32,
            max_tiles_per_action=2,
        ),
    )
    result = UnifiedExperimentRunner(
        datasets=_registry(tmp_path), runtime=_runtime(backend), config=config
    ).run("generic", "all", 0)
    pass_record = result.run.passes[0]
    assert pass_record.tiling_decision is not None
    assert pass_record.tiling_decision.triggered is True
    assert len(pass_record.tiles) == 2
    assert result.run.final_predictions["tile_calls"] == 2


def test_unified_static_asht_dispatches_existing_controller(tmp_path):
    from agent.asht.action_bank import StaticActionBank, StaticActionTemplate
    from agent.asht.kernels import SensorProfile
    from agent.asht.runner_static import StaticAshtConfig
    from pipeline_stages import Sam3QuerySpec
    from provenance.schema import ActionFamily

    backend = FakeBackend()
    bank = StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="target",
                family=ActionFamily.TARGET,
                prompt="object",
                semantic_key="object",
                beta_by_class={
                    "target": 0.95,
                    "non_target": 0.10,
                    "background": 0.05,
                },
            ),
        ),
        profile=SensorProfile(
            observation_labels=("not_found", "weak_match", "strong_match"),
            present=(0.02, 0.08, 0.90),
            absent=(0.90, 0.08, 0.02),
            source="test",
        ),
        positive_class="target",
        negative_class="non_target",
    )
    asht = StaticAshtConfig(
        class_names=("target", "non_target", "background"),
        target_class="target",
        prior={"target": 1 / 3, "non_target": 1 / 3, "background": 1 / 3},
        action_bank=bank,
        stopping_threshold=0.75,
        max_total_queries=3,
        max_queries_per_node=2,
        bootstrap_query=Sam3QuerySpec(
            prompt="object",
            threshold=0.2,
            region=(0.0, 0.0, 64.0, 64.0),
        ),
    )
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.ASHT_STATIC,
        output_root=tmp_path / "runs",
        class_names=("target", "non_target", "background"),
        target_class="target",
        asht_config=asht,
    )
    result = UnifiedExperimentRunner(
        datasets=_registry(tmp_path), runtime=_runtime(backend), config=config
    ).run("generic", "all", 0)
    assert result.run.status is RunStatus.SUCCEEDED
    assert len(result.run.passes) >= 2
    assert result.run.final_predictions["soft_count"] > 0.0
    assert result.run.final_predictions["total_sam3_queries"] >= 2


def test_unified_runner_materializes_declared_ground_truth_masks(tmp_path):
    backend = FakeBackend()
    image_dir = tmp_path / "dataset"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(image_dir / "sample.png")
    Image.new("L", (32, 32), 1).save(image_dir / "sample_mask.png")
    (image_dir / "sample.json").write_text(
        '{"ground_truth_mask_paths": ["sample_mask.png"], "count": 1}'
    )
    registry = DatasetRegistry(
        [GenericFolderAdapter(image_dir, target_concept="object", target_class="target")]
    )
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_SINGLE_PASS,
        output_root=tmp_path / "runs",
        class_names=("target", "non_target", "background"),
        target_class="target",
        discovery_passes=(DiscoveryPassSpec(prompt="object", threshold=0.2),),
    )
    result = UnifiedExperimentRunner(
        datasets=registry, runtime=_runtime(backend), config=config
    ).run("generic", "all", 0)
    mask_records = result.run.dataset_sample.ground_truth_masks
    assert len(mask_records) == 1
    assert (result.run_directory / mask_records[0].relative_path).is_file()


def test_environment_capture_excludes_secret_variables(monkeypatch):
    from experiments.runner import detect_environment

    monkeypatch.setenv("QWEN_MODEL", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "do-not-store")
    environment = detect_environment()
    assert environment.environment_variables["QWEN_MODEL"] == "qwen"
    assert "QWEN_API_KEY" not in environment.environment_variables

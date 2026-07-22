from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.asht.action_bank import StaticActionBank, StaticActionTemplate
from agent.asht.kernels import SensorProfile
from agent.asht.qwen_generator import QwenActionGeneratorConfig, QwenCandidateActionGenerator
from agent.asht.runner_static import StaticAshtConfig
from agent.asht.tiling import AdaptiveTilingConfig
from eval.dataset_adapters import (
    CarpkAdapter,
    DatasetRegistry,
    GenericFolderAdapter,
    GreenCitrusAdapter,
)
from experiments.config import DiscoveryPassSpec, ExperimentMode, UnifiedExperimentConfig
from experiments.discovery import DiscoveryExperimentExecutor
from experiments.runner import (
    LegacyExecutionResult,
    UnifiedExperimentRunner,
    UnifiedRuntime,
)
from pipeline_stages import Sam3QuerySpec
from provenance.ids import create_run_id
from provenance.io import strict_json_load
from provenance.schema import ActionFamily, RunStatus, TilingMode
from validation import (
    compare_run_semantics,
    semantic_run_fingerprint,
    validate_run_directory,
    validate_run_tree,
)


class DeterministicBackend:
    def __init__(self) -> None:
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
        x2, y2 = min(12.0, float(width)), min(12.0, float(height))
        boxes = np.asarray([[1.0, 1.0, x2, y2]], dtype=float)
        scores = np.asarray([0.95], dtype=float)
        if return_masks:
            return boxes, scores, [np.ones((height, width), dtype=bool)]
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


class DeterministicQwen:
    def __init__(self, *, tiling: bool = False) -> None:
        self.tiling = tiling

    def generate(self, **kwargs):
        action = {
            "family": "target",
            "prompt": "compact target object",
            "semantic_key": "compact+target",
            "beta_by_class": {
                "target": 0.98,
                "non_target": 0.02,
                "background": 0.01,
            },
            "threshold": 0.20,
            "rationale": "Test a target-specific conjunction.",
        }
        if self.tiling:
            action.update({"tiling_mode": "agent_controlled", "tile_scale": 1.0})
        return {
            "content": json.dumps(
                {
                    "reasoning_summary": "Use one deterministic target action.",
                    "actions": [action],
                }
            ),
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }


def _profile() -> SensorProfile:
    return SensorProfile(
        observation_labels=("not_found", "weak_match", "strong_match"),
        present=(0.02, 0.08, 0.90),
        absent=(0.90, 0.08, 0.02),
        source="phase13-test",
    )


def _bank() -> StaticActionBank:
    return StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="target",
                family=ActionFamily.TARGET,
                prompt="compact target object",
                semantic_key="compact+target",
                beta_by_class={
                    "target": 0.98,
                    "non_target": 0.02,
                    "background": 0.01,
                },
                threshold=0.20,
            ),
        ),
        profile=_profile(),
        positive_class="target",
        negative_class="non_target",
    )


def _asht() -> StaticAshtConfig:
    return StaticAshtConfig(
        class_names=("target", "non_target", "background"),
        target_class="target",
        prior={"target": 1 / 3, "non_target": 1 / 3, "background": 1 / 3},
        action_bank=_bank(),
        stopping_threshold=0.70,
        max_total_queries=3,
        max_queries_per_node=1,
        bootstrap_query=Sam3QuerySpec(
            prompt="target object",
            threshold=0.20,
            region=(0.0, 0.0, 64.0, 64.0),
        ),
        adaptive_tiling=AdaptiveTilingConfig(
            min_tile_size=16,
            max_tile_size=32,
            max_tiles_per_action=1,
        ),
    )


def _generic_registry(root: Path) -> DatasetRegistry:
    images = root / "generic"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (20, 40, 60)).save(images / "sample.png")
    (images / "sample.json").write_text(
        json.dumps({"count": 1, "boxes": [[1, 1, 12, 12]]}),
        encoding="utf-8",
    )
    return DatasetRegistry(
        [GenericFolderAdapter(images, target_concept="target object", target_class="target")]
    )


def _run_id_factory(prefix: str):
    counter = 0

    def factory():
        nonlocal counter
        counter += 1
        return create_run_id(
            now=datetime(2026, 7, 22, 12, 0, counter, tzinfo=timezone.utc),
            random_token=f"{prefix}-{counter}",
        )

    return factory


def _qwen_generator(*, tiling: bool = False) -> QwenCandidateActionGenerator:
    return QwenCandidateActionGenerator(
        client=DeterministicQwen(tiling=tiling),
        profile=_profile(),
        fallback_bank=_bank(),
        config=QwenActionGeneratorConfig(max_retries=0, sampling_parameters={"temperature": 0.0}),
    )


def _legacy_executor(*, sample, image, image_np, graph, store, config, runtime):
    executor = DiscoveryExperimentExecutor(
        backend=runtime.backend,
        processor=runtime.processor,
        image_np=image_np,
        graph=graph,
        store=store,
        class_names=config.class_names,
        model_id=config.model_id,
        adaptive_tiling=config.adaptive_tiling,
        suppression_confidence=config.suppression_confidence,
        suppression_kwargs=dict(config.suppression_kwargs),
        cross_pass_dedup_metric=config.cross_pass_dedup_metric,
        cross_pass_dedup_threshold=config.cross_pass_dedup_threshold,
    )
    result = executor.run((DiscoveryPassSpec(prompt="legacy target", threshold=0.2),))
    return LegacyExecutionResult(
        graph=graph,
        final_predictions={
            "hard_count": len(graph.nodes),
            "soft_count": float(len(graph.nodes)),
            "count_variance": 0.0,
            "unresolved_count": len(graph.nodes),
            "total_sam3_queries": result.sam3_calls,
        },
        metadata={"adapter": "phase13-test"},
    )


def _runtime(prefix: str, *, qwen: bool = False, agent_tiling: bool = False, legacy: bool = False):
    return UnifiedRuntime(
        backend=DeterministicBackend(),
        processor=object(),
        qwen_action_generator=_qwen_generator(tiling=agent_tiling) if qwen else None,
        legacy_executor=_legacy_executor if legacy else None,
        runtime_configuration={"test_runtime": "phase13", "agent_tiling": agent_tiling},
        run_id_factory=_run_id_factory(prefix),
    )


def _config(mode: ExperimentMode, output_root: Path) -> UnifiedExperimentConfig:
    common = dict(
        mode=mode,
        output_root=output_root,
        class_names=("target", "non_target", "background"),
        target_class="target",
        fsync=False,
        adaptive_tiling=AdaptiveTilingConfig(
            min_tile_size=16,
            max_tile_size=32,
            max_tiles_per_action=1,
        ),
    )
    if mode is ExperimentMode.SAM3_SINGLE_PASS:
        return UnifiedExperimentConfig(
            **common,
            discovery_passes=(DiscoveryPassSpec(prompt="target", threshold=0.2),),
        )
    if mode is ExperimentMode.SAM3_FIXED_MULTIPASS:
        return UnifiedExperimentConfig(
            **common,
            discovery_passes=(
                DiscoveryPassSpec(prompt="target", threshold=0.2),
                DiscoveryPassSpec(prompt="compact target", threshold=0.2),
            ),
        )
    if mode is ExperimentMode.SAM3_ADAPTIVE_TILING:
        return UnifiedExperimentConfig(
            **common,
            discovery_passes=(
                DiscoveryPassSpec(
                    prompt="target",
                    threshold=0.2,
                    tiling_mode=TilingMode.ALWAYS,
                ),
            ),
        )
    if mode is ExperimentMode.LEGACY_VLM:
        return UnifiedExperimentConfig(**common)
    return UnifiedExperimentConfig(**common, asht_config=_asht())


@pytest.mark.integration
@pytest.mark.parametrize("mode", tuple(ExperimentMode))
def test_every_unified_mode_completes_and_validates(tmp_path: Path, mode: ExperimentMode):
    registry = _generic_registry(tmp_path / mode.value)
    qwen = mode in {
        ExperimentMode.ASHT_QWEN,
        ExperimentMode.ASHT_ADAPTIVE_TILING,
        ExperimentMode.ASHT_AGENT_TILING,
    }
    result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime(
            mode.value,
            qwen=qwen,
            agent_tiling=mode is ExperimentMode.ASHT_AGENT_TILING,
            legacy=mode is ExperimentMode.LEGACY_VLM,
        ),
        config=_config(mode, tmp_path / "runs" / mode.value),
    ).run("generic", "all", 0)
    assert result.run.status is RunStatus.SUCCEEDED
    report = validate_run_directory(result.run_directory)
    assert report.valid, [issue.__dict__ for issue in report.issues]
    assert report.counts["passes"] >= 1


@pytest.mark.reproducibility
def test_semantic_fingerprint_ignores_ids_times_and_latencies(tmp_path: Path):
    registry = _generic_registry(tmp_path / "data")
    config = _config(ExperimentMode.SAM3_FIXED_MULTIPASS, tmp_path / "runs")
    first = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("repeat-a"),
        config=config,
    ).run("generic", "all", 0)
    second = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("repeat-b"),
        config=config,
    ).run("generic", "all", 0)
    assert first.run.run_id != second.run.run_id
    assert semantic_run_fingerprint(first.run) == semantic_run_fingerprint(second.run)
    assert compare_run_semantics(first.run, second.run).equal


@pytest.mark.reproducibility
def test_semantic_comparison_reports_real_policy_difference(tmp_path: Path):
    registry = _generic_registry(tmp_path / "data")
    first = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("policy-a"),
        config=UnifiedExperimentConfig(
            mode=ExperimentMode.SAM3_SINGLE_PASS,
            output_root=tmp_path / "runs-a",
            fsync=False,
            class_names=("target", "non_target", "background"),
            target_class="target",
            discovery_passes=(DiscoveryPassSpec(prompt="target", threshold=0.2),),
        ),
    ).run("generic", "all", 0)
    second = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("policy-b"),
        config=UnifiedExperimentConfig(
            mode=ExperimentMode.SAM3_SINGLE_PASS,
            output_root=tmp_path / "runs-b",
            fsync=False,
            class_names=("target", "non_target", "background"),
            target_class="target",
            discovery_passes=(DiscoveryPassSpec(prompt="different target", threshold=0.2),),
        ),
    ).run("generic", "all", 0)
    comparison = compare_run_semantics(first.run, second.run)
    assert not comparison.equal
    assert comparison.differences


def test_validator_detects_artifact_tampering(tmp_path: Path):
    registry = _generic_registry(tmp_path / "data")
    result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("artifact"),
        config=_config(ExperimentMode.SAM3_SINGLE_PASS, tmp_path / "runs"),
    ).run("generic", "all", 0)
    artifact = result.run.dataset_sample.image_artifact
    (result.run_directory / artifact.relative_path).write_bytes(b"tampered")
    report = validate_run_directory(result.run_directory)
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {"artifact_hash", "input_image_hash"}


def test_validator_detects_event_log_divergence(tmp_path: Path):
    registry = _generic_registry(tmp_path / "data")
    result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("event"),
        config=_config(ExperimentMode.SAM3_SINGLE_PASS, tmp_path / "runs"),
    ).run("generic", "all", 0)
    path = result.run_directory / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["created_at"] = "2026-07-22T00:00:00+00:00"
    lines[0] = json.dumps(payload, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_run_directory(result.run_directory)
    assert not report.valid
    assert any(issue.code == "event_log_hash" for issue in report.issues)


def test_run_tree_validation_and_graph_replay(tmp_path: Path):
    registry = _generic_registry(tmp_path / "data")
    for token in ("tree-a", "tree-b"):
        UnifiedExperimentRunner(
            datasets=registry,
            runtime=_runtime(token),
            config=_config(ExperimentMode.SAM3_SINGLE_PASS, tmp_path / "runs"),
        ).run("generic", "all", 0)
    reports = validate_run_tree(tmp_path / "runs")
    assert len(reports) == 2
    assert all(report.valid for report in reports)
    assert len({report.semantic_fingerprint for report in reports}) == 1


@pytest.mark.integration
def test_synthetic_citrus_and_carpk_adapters_run_end_to_end(tmp_path: Path):
    citrus = tmp_path / "citrus"
    (citrus / "images" / "test").mkdir(parents=True)
    (citrus / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 90, 0)).save(citrus / "images" / "test" / "tree.png")
    (citrus / "labels" / "test" / "tree.txt").write_text("0 0.1 0.1 0.15 0.15\n", encoding="utf-8")

    carpk = tmp_path / "carpk"
    (carpk / "Images").mkdir(parents=True)
    (carpk / "Annotations").mkdir()
    (carpk / "ImageSets").mkdir()
    Image.new("RGB", (64, 64), (70, 70, 70)).save(carpk / "Images" / "parking.png")
    (carpk / "Annotations" / "parking.txt").write_text("1 1 12 12\n", encoding="utf-8")
    (carpk / "ImageSets" / "test.txt").write_text("parking\n", encoding="utf-8")

    registry = DatasetRegistry([GreenCitrusAdapter(citrus), CarpkAdapter(carpk)])
    citrus_config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_SINGLE_PASS,
        output_root=tmp_path / "runs-citrus",
        fsync=False,
        class_names=("fruit", "leaf", "background"),
        target_class="fruit",
        discovery_passes=(DiscoveryPassSpec(prompt="green citrus fruit", threshold=0.2),),
    )
    car_config = UnifiedExperimentConfig(
        mode=ExperimentMode.SAM3_SINGLE_PASS,
        output_root=tmp_path / "runs-carpk",
        fsync=False,
        class_names=("car", "non_car", "background"),
        target_class="car",
        discovery_passes=(DiscoveryPassSpec(prompt="car", threshold=0.2),),
    )
    citrus_result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("citrus"),
        config=citrus_config,
    ).run("green_citrus", "test", 0)
    car_result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("carpk"),
        config=car_config,
    ).run("carpk", "test", 0)
    assert citrus_result.run.evaluations[-1].metrics["candidate_pool_recall"] == 1.0
    assert car_result.run.evaluations[-1].metrics["candidate_pool_recall"] == 1.0
    assert validate_run_directory(citrus_result.run_directory).valid
    assert validate_run_directory(car_result.run_directory).valid


def test_resolved_configuration_captures_runtime_and_operational_settings(tmp_path: Path):
    registry = _generic_registry(tmp_path / "data")
    config = UnifiedExperimentConfig(
        mode=ExperimentMode.ASHT_QWEN,
        output_root=tmp_path / "runs",
        fsync=False,
        overwrite=True,
        checkpoint_every_events=3,
        raise_on_error=False,
        class_names=("target", "non_target", "background"),
        target_class="target",
        asht_config=_asht(),
    )
    result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("config", qwen=True),
        config=config,
    ).run("generic", "all", 0)
    resolved = result.run.resolved_config
    assert resolved["fsync"] is False
    assert resolved["overwrite"] is True
    assert resolved["checkpoint_every_events"] == 3
    assert resolved["raise_on_error"] is False
    assert resolved["runtime"]["test_runtime"] == "phase13"
    assert resolved["runtime"]["qwen_action_generator"]["config"]["max_retries"] == 0
    assert strict_json_load(result.run_directory / "config.json")["resolved_config"] == resolved


def test_pre_release_schema_payload_migrates_recursively(tmp_path: Path):
    from provenance import RunRecord, migrate_payload, migrated_record_from_dict

    registry = _generic_registry(tmp_path / "data")
    result = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("migration"),
        config=_config(ExperimentMode.SAM3_SINGLE_PASS, tmp_path / "runs"),
    ).run("generic", "all", 0)
    payload = result.run.to_dict()

    def downgrade(value):
        if isinstance(value, dict):
            return {
                key: ("0.9.0" if key == "schema_version" and "record_type" in value else downgrade(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [downgrade(item) for item in value]
        return value

    old = downgrade(payload)
    old.pop("events", None)
    old.pop("warnings", None)
    old.pop("metadata", None)
    migrated = migrate_payload(old)
    loaded = migrated_record_from_dict(old)
    assert migrated["schema_version"] == "1.0.0"
    assert isinstance(loaded, RunRecord)
    assert loaded.run_id == result.run.run_id
    assert loaded.events == ()


@pytest.mark.integration
def test_interrupted_run_can_be_reopened_and_finalized(tmp_path: Path):
    from provenance.run_store import RunStore

    registry = _generic_registry(tmp_path / "data")
    runner = UnifiedExperimentRunner(
        datasets=registry,
        runtime=_runtime("interrupt"),
        config=_config(ExperimentMode.SAM3_SINGLE_PASS, tmp_path / "runs"),
    )
    sample = registry.get("generic", "all", 0)
    initial, declared = runner._initial_run(sample)
    store = RunStore.create(tmp_path / "runs", initial, fsync=False)
    for artifact, source in declared:
        store.materialize_declared_artifact(artifact, source)
    with store.paths.events.open("ab") as handle:
        handle.write(b'{"incomplete":')
    reopened = RunStore.open(store.paths.root, fsync=False, repair_truncated_tail=True)
    interrupted = reopened.mark_interrupted(reason="simulated interruption")
    assert interrupted.status is RunStatus.INTERRUPTED
    report = validate_run_directory(store.paths.root, verify_graph_replay=False)
    assert report.valid


@pytest.mark.legacy
def test_legacy_execute_pass_contract_remains_callable(monkeypatch, tmp_path: Path):
    """Exercise the pre-ASHT pass boundary without importing heavy model packages."""

    import importlib.util
    import sys
    import types

    fake_inference = types.ModuleType("inference")
    fake_inference.apply_clahe = lambda image: image
    fake_inference.apply_nms_dualgate = lambda boxes, scores, confidence, **kwargs: (boxes, scores)
    fake_inference.apply_nms = lambda boxes, scores, confidence: (boxes, scores)
    fake_verifier = types.ModuleType("verifier")
    fake_verifier.verify = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "inference", fake_inference)
    monkeypatch.setitem(sys.modules, "verifier", fake_verifier)

    path = Path(__file__).resolve().parents[1] / "pipeline.py"
    spec = importlib.util.spec_from_file_location("phase13_legacy_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.initialize_canopy_roi = lambda processor, image, graph, use_canopy=True: [0, 0, 64, 64]

    def global_engine(processor, image, confidence, prompt, **kwargs):
        if prompt == "green leaf":
            return np.empty((0, 4), dtype=float), np.empty((0,), dtype=float)
        return np.asarray([[1.0, 1.0, 12.0, 12.0]]), np.asarray([0.95])

    module.global_engine = global_engine
    module.register_and_verify_candidates = lambda **kwargs: (1, 0)
    graph = types.SimpleNamespace(
        tree_roi=None,
        cached_leaf_boxes=None,
        cached_leaf_roi=None,
    )
    result = module.execute_pass(
        processor=object(),
        image_pil=Image.new("RGB", (64, 64)),
        graph=graph,
        conf=0.2,
        clahe=False,
        tiling=False,
        pass_number=1,
        prompt="target",
    )
    assert int(result) == 1
    assert result.raw_proposals == 1
    assert result.post_nms == 1
    assert result.accepted == 1
    assert result.n_sam_calls == 3


def test_runtime_configuration_filters_secret_values():
    runtime = UnifiedRuntime(
        backend=DeterministicBackend(),
        processor=object(),
        runtime_configuration={
            "endpoint": "local",
            "api_key": "never-store",
            "nested": {"access_token": "never-store", "batch_size": 4},
        },
    )
    resolved = runtime.resolved_configuration()
    assert resolved["endpoint"] == "local"
    assert "api_key" not in resolved
    assert "access_token" not in resolved["nested"]
    assert resolved["nested"]["batch_size"] == 4

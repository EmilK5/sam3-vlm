"""Single entry point for every experiment pipeline mode."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from agent.asht.runner_qwen import QwenAshtRunner
from agent.asht.runner_static import StaticAshtRunResult, StaticAshtRunner
from eval.dataset_adapters import CanonicalSample, DatasetRegistry
from experiments.config import ExperimentMode, UnifiedExperimentConfig
from experiments.discovery import DiscoveryExecutionResult, DiscoveryExperimentExecutor
from graph import OrchardGraph
from provenance.base import utc_now_iso
from provenance.contracts import (
    ArtifactRef,
    DatasetSampleRecord,
    EnvironmentRecord,
    ModelIdentityRecord,
    RepositoryStateRecord,
    RunRecord,
)
from provenance.ids import EntityKind, IdFactory, create_run_id
from provenance.io import sha256_file, sha256_json
from provenance.run_store import RunStore
from provenance.schema import ArtifactKind, RunStatus, TilingMode


class UnifiedRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyExecutionResult:
    """Adapter result for an existing policy that is not ASHT-native yet."""

    graph: OrchardGraph
    final_predictions: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class LegacyExecutor(Protocol):
    def __call__(
        self,
        *,
        sample: CanonicalSample,
        image,
        image_np: np.ndarray,
        graph: OrchardGraph,
        store: RunStore,
        config: UnifiedExperimentConfig,
        runtime: "UnifiedRuntime",
    ) -> LegacyExecutionResult:
        ...


@dataclass
class UnifiedRuntime:
    backend: Any
    processor: Any
    qwen_action_generator: Any | None = None
    legacy_executor: LegacyExecutor | None = None
    repository: RepositoryStateRecord | None = None
    environment: EnvironmentRecord | None = None
    models: tuple[ModelIdentityRecord, ...] = ()
    run_id_factory: Callable[[], str] = create_run_id

    def resolved_repository(self) -> RepositoryStateRecord:
        return self.repository or detect_repository_state()

    def resolved_environment(self) -> EnvironmentRecord:
        return self.environment or detect_environment()

    def resolved_models(self, model_id: str) -> tuple[ModelIdentityRecord, ...]:
        if self.models:
            return self.models
        return (
            ModelIdentityRecord(
                model_id=model_id,
                role="visual_sensor",
                name=model_id,
                provider="local",
            ),
        )


@dataclass(frozen=True)
class UnifiedExperimentResult:
    run: RunRecord
    run_directory: Path
    graph: OrchardGraph
    execution_result: DiscoveryExecutionResult | StaticAshtRunResult | LegacyExecutionResult


class UnifiedExperimentRunner:
    """Resolve one sample and dispatch it without dataset-specific branches."""

    def __init__(
        self,
        *,
        datasets: DatasetRegistry,
        runtime: UnifiedRuntime,
        config: UnifiedExperimentConfig,
    ) -> None:
        self.datasets = datasets
        self.runtime = runtime
        self.config = config

    def run(self, dataset_name: str, split: str, index: int) -> UnifiedExperimentResult:
        sample = self.datasets.get(dataset_name, split, index)
        return self.run_sample(sample)

    def run_sample(self, sample: CanonicalSample) -> UnifiedExperimentResult:
        _set_seeds(self.config.random_seed)
        image = sample.load_image()
        image_np = np.asarray(image)
        initial_run, declared_artifacts = self._initial_run(sample)
        store = RunStore.create(
            self.config.output_root,
            initial_run,
            reporting_level=self.config.reporting_level,
            overwrite=self.config.overwrite,
            fsync=self.config.fsync,
            checkpoint_every_events=self.config.checkpoint_every_events,
        )
        graph = OrchardGraph()
        try:
            for artifact, source_path in declared_artifacts:
                store.materialize_declared_artifact(artifact, source_path)
            execution = self._execute(
                sample=sample,
                image=image,
                image_np=image_np,
                graph=graph,
                store=store,
            )
            predictions = _final_predictions(execution, graph, self.config.target_class)
            graph_artifact = store.write_artifact_bytes(
                json.dumps(graph.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
                kind=ArtifactKind.GRAPH,
                relative_path="graph/final_graph.json",
                media_type="application/json",
                metadata={"graph_schema_version": graph.to_dict().get("graph_schema_version")},
            )
            predictions = {
                **predictions,
                "pipeline_mode": self.config.mode.value,
                "dataset_name": sample.dataset_name,
                "sample_key": sample.sample_key,
                "final_graph_artifact_id": graph_artifact.artifact_id,
            }
            final = store.finalize_success(
                final_predictions=predictions,
                metadata={
                    "pipeline_mode": self.config.mode.value,
                    "dataset_adapter": sample.dataset_name,
                    "sample_key": sample.sample_key,
                },
            )
            return UnifiedExperimentResult(
                run=final,
                run_directory=store.paths.root,
                graph=graph,
                execution_result=execution,
            )
        except BaseException as exc:
            failed = store.finalize_failure(
                exc,
                component="unified_experiment_runner",
                metadata={
                    "pipeline_mode": self.config.mode.value,
                    "dataset_adapter": sample.dataset_name,
                    "sample_key": sample.sample_key,
                },
            )
            if self.config.raise_on_error:
                raise
            return UnifiedExperimentResult(
                run=failed,
                run_directory=store.paths.root,
                graph=graph,
                execution_result=LegacyExecutionResult(
                    graph=graph,
                    final_predictions={},
                    metadata={"failed": True, "exception_type": type(exc).__name__},
                ),
            )

    def _execute(
        self,
        *,
        sample: CanonicalSample,
        image,
        image_np: np.ndarray,
        graph: OrchardGraph,
        store: RunStore,
    ) -> DiscoveryExecutionResult | StaticAshtRunResult | LegacyExecutionResult:
        mode = self.config.mode
        if mode in {
            ExperimentMode.SAM3_SINGLE_PASS,
            ExperimentMode.SAM3_FIXED_MULTIPASS,
            ExperimentMode.SAM3_ADAPTIVE_TILING,
        }:
            executor = DiscoveryExperimentExecutor(
                backend=self.runtime.backend,
                processor=self.runtime.processor,
                image_np=image_np,
                graph=graph,
                store=store,
                class_names=self.config.class_names,
                model_id=self.config.model_id,
                adaptive_tiling=self.config.adaptive_tiling,
                suppression_confidence=self.config.suppression_confidence,
                suppression_kwargs=dict(self.config.suppression_kwargs),
                cross_pass_dedup_metric=self.config.cross_pass_dedup_metric,
                cross_pass_dedup_threshold=self.config.cross_pass_dedup_threshold,
            )
            return executor.run(self.config.discovery_passes)

        if mode is ExperimentMode.LEGACY_VLM:
            if self.runtime.legacy_executor is None:
                raise UnifiedRunnerError(
                    "legacy_vlm mode requires a provenance-aware legacy_executor"
                )
            return self.runtime.legacy_executor(
                sample=sample,
                image=image,
                image_np=image_np,
                graph=graph,
                store=store,
                config=self.config,
                runtime=self.runtime,
            )

        asht_config = self.config.asht_config
        if asht_config is None:
            raise UnifiedRunnerError(f"{mode.value} requires asht_config")
        if tuple(asht_config.class_names) != tuple(self.config.class_names):
            raise UnifiedRunnerError("ASHT class_names must match unified class_names")
        if asht_config.target_class != self.config.target_class:
            raise UnifiedRunnerError("ASHT target_class must match unified target_class")

        if mode is ExperimentMode.ASHT_ADAPTIVE_TILING:
            asht_config = dataclasses.replace(
                asht_config,
                bootstrap_tiling_mode=TilingMode.DENSITY_ADAPTIVE,
                adaptive_tiling=self.config.adaptive_tiling,
            )
        elif mode is ExperimentMode.ASHT_AGENT_TILING:
            asht_config = dataclasses.replace(
                asht_config,
                adaptive_tiling=self.config.adaptive_tiling,
            )

        common = dict(
            backend=self.runtime.backend,
            processor=self.runtime.processor,
            image_np=image_np,
            graph=graph,
            config=asht_config,
            sink=store,
            model_id=self.config.model_id,
        )
        if mode is ExperimentMode.ASHT_STATIC:
            return StaticAshtRunner(**common).run()
        if mode in {
            ExperimentMode.ASHT_QWEN,
            ExperimentMode.ASHT_ADAPTIVE_TILING,
            ExperimentMode.ASHT_AGENT_TILING,
        }:
            if self.runtime.qwen_action_generator is None:
                raise UnifiedRunnerError(f"{mode.value} requires qwen_action_generator")
            return QwenAshtRunner(
                action_generator=self.runtime.qwen_action_generator,
                **common,
            ).run()
        raise UnifiedRunnerError(f"unsupported experiment mode: {mode}")

    def _initial_run(
        self, sample: CanonicalSample
    ) -> tuple[RunRecord, tuple[tuple[ArtifactRef, Path], ...]]:
        run_id = self.runtime.run_id_factory()
        factory = IdFactory(run_id)
        width, height = sample.image_size
        image_artifact = ArtifactRef(
            artifact_id=factory.new(EntityKind.ARTIFACT),
            kind=ArtifactKind.IMAGE,
            relative_path=str(Path("artifacts") / "input" / sample.image_path.name),
            media_type=_image_media_type(sample.image_path),
            sha256=sha256_file(sample.image_path),
            size_bytes=sample.image_path.stat().st_size,
            width=width,
            height=height,
            metadata={
                "source_path": str(sample.image_path),
                "sample_key": sample.sample_key,
                "role": "experiment_input",
            },
        )
        mask_artifacts = tuple(
            ArtifactRef(
                artifact_id=factory.new(EntityKind.ARTIFACT),
                kind=ArtifactKind.MASK,
                relative_path=str(
                    Path("artifacts")
                    / "ground_truth"
                    / "masks"
                    / f"{index:04d}_{path.name}"
                ),
                media_type=_image_media_type(path),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                metadata={
                    "source_path": str(path),
                    "sample_key": sample.sample_key,
                    "role": "ground_truth_mask",
                },
            )
            for index, path in enumerate(sample.ground_truth_mask_paths)
        )
        dataset_record = DatasetSampleRecord(
            image_id=factory.new(EntityKind.IMAGE),
            dataset_name=sample.dataset_name,
            dataset_version=sample.dataset_version,
            split=sample.split,
            image_artifact=image_artifact,
            image_sha256=image_artifact.sha256 or sample.image_sha256,
            width=width,
            height=height,
            target_concept=sample.target_concept,
            target_class=sample.target_class,
            sample_index=sample.sample_index,
            sequence_id=sample.sequence_id,
            confounder_classes=sample.confounder_classes,
            ground_truth_count=sample.ground_truth_count,
            ground_truth_boxes=sample.ground_truth_boxes,
            ground_truth_masks=mask_artifacts,
            exemplar_boxes=sample.exemplar_boxes,
            metadata={
                **dict(sample.metadata),
                "sample_key": sample.sample_key,
                "capabilities": sample.capabilities.as_dict(),
                "point_annotations": [list(point) for point in sample.point_annotations],
                "ground_truth_mask_paths": [
                    str(path) for path in sample.ground_truth_mask_paths
                ],
            },
        )
        resolved = dict(self.config.resolved_mapping())
        resolved["dataset"] = {
            "name": sample.dataset_name,
            "version": sample.dataset_version,
            "split": sample.split,
            "sample_key": sample.sample_key,
        }
        initial = RunRecord(
            run_id=run_id,
            status=RunStatus.CREATED,
            created_at=utc_now_iso(),
            started_at=None,
            completed_at=None,
            repository=self.runtime.resolved_repository(),
            environment=self.runtime.resolved_environment(),
            resolved_config=resolved,
            config_sha256=sha256_json(resolved),
            random_seeds={"python": self.config.random_seed, "numpy": self.config.random_seed},
            dataset_sample=dataset_record,
            models=self.runtime.resolved_models(self.config.model_id),
            id_counters=factory.snapshot(),
            passes=(),
            final_graph=(),
            final_predictions={},
            evaluations=(),
            artifacts=(image_artifact, *mask_artifacts),
            errors=(),
            metadata={
                "pipeline_mode": self.config.mode.value,
                "run_name": self.config.run_name,
            },
        )
        declared = ((image_artifact, sample.image_path),) + tuple(
            zip(mask_artifacts, sample.ground_truth_mask_paths)
        )
        return initial, declared


def _final_predictions(execution, graph: OrchardGraph, target_class: str) -> Mapping[str, Any]:
    if isinstance(execution, StaticAshtRunResult):
        return {
            "hard_count": execution.count.hard_count,
            "soft_count": execution.count.soft_count,
            "count_variance": execution.count.variance,
            "unresolved_count": execution.count.unresolved_count,
            "total_sam3_queries": execution.total_sam3_queries,
            "selected_node_ids": list(execution.selected_node_ids),
        }
    if isinstance(execution, DiscoveryExecutionResult):
        return {
            "hard_count": len(graph.nodes),
            "soft_count": float(len(graph.nodes)),
            "count_variance": 0.0,
            "unresolved_count": len(graph.nodes),
            "total_sam3_queries": execution.sam3_calls,
            "tile_calls": execution.tile_calls,
        }
    if isinstance(execution, LegacyExecutionResult):
        return dict(execution.final_predictions)
    return {
        "hard_count": sum(
            1
            for node in graph.nodes.values()
            if getattr(node, "final_declaration", None) == target_class
        )
    }


def detect_repository_state(root: Path | None = None) -> RepositoryStateRecord:
    root = Path(root or Path.cwd())

    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = command("rev-parse", "HEAD") or "unknown"
    branch = command("branch", "--show-current") or "unknown"
    status = command("status", "--porcelain")
    remote = command("remote", "get-url", "origin")
    return RepositoryStateRecord(
        commit=commit,
        branch=branch,
        dirty=bool(status),
        remote_url=remote,
    )


def detect_environment() -> EnvironmentRecord:
    allowed_prefixes = ("CUDA", "QWEN", "SAM3", "PYTHONHASHSEED")
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    selected_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(allowed_prefixes)
        and not any(marker in key.upper() for marker in secret_markers)
    }
    return EnvironmentRecord(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
        packages={},
        hardware={"machine": platform.machine(), "processor": platform.processor()},
        environment_variables=selected_environment,
    )


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


__all__ = [
    "LegacyExecutionResult",
    "UnifiedExperimentResult",
    "UnifiedExperimentRunner",
    "UnifiedRunnerError",
    "UnifiedRuntime",
    "detect_environment",
    "detect_repository_state",
]

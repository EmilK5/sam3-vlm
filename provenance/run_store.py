"""Crash-safe run-directory manager for complete experimental provenance."""

from __future__ import annotations

import dataclasses
import mimetypes
import shutil
import traceback as traceback_module
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from provenance.base import CanonicalRecord, utc_now_iso
from provenance.consolidation import consolidate_run
from provenance.contracts import (
    ArtifactRef,
    ErrorEventRecord,
    EventEnvelope,
    RunRecord,
)
from provenance.events import EventLogReader, EventLogWriter
from provenance.ids import EntityKind, IdFactory
from provenance.io import (
    StorageError,
    atomic_write_bytes,
    atomic_write_json,
    repair_truncated_jsonl_tail,
    safe_relative_path,
    sha256_file,
    strict_json_load,
)
from provenance.schema import ArtifactKind, ErrorSeverity, EventKind, RunStatus


class ReportingLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


_REPORTING_RANK = {
    ReportingLevel.MINIMAL: 0,
    ReportingLevel.STANDARD: 1,
    ReportingLevel.FULL: 2,
}


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    events: Path
    config: Path
    environment: Path
    partial_run: Path
    final_run: Path
    summary: Path
    artifacts: Path
    checkpoints: Path
    latest_checkpoint: Path

    @classmethod
    def under(cls, root: Path) -> "RunPaths":
        root = Path(root)
        checkpoints = root / "checkpoints"
        return cls(
            root=root,
            manifest=root / "manifest.json",
            events=root / "events.jsonl",
            config=root / "config.json",
            environment=root / "environment.json",
            partial_run=root / "run.partial.json",
            final_run=root / "run.json",
            summary=root / "summary.json",
            artifacts=root / "artifacts",
            checkpoints=checkpoints,
            latest_checkpoint=checkpoints / "latest.json",
        )


class RunStore:
    """Owns one self-contained run directory and its append-only event stream."""

    def __init__(
        self,
        *,
        paths: RunPaths,
        base_run: RunRecord,
        reporting_level: ReportingLevel,
        id_factory: IdFactory,
        event_writer: EventLogWriter,
        fsync: bool = True,
        checkpoint_every_events: int = 0,
    ) -> None:
        self.paths = paths
        self.base_run = base_run
        self.reporting_level = reporting_level
        self.id_factory = id_factory
        self.event_writer = event_writer
        self.fsync = fsync
        self.checkpoint_every_events = max(0, int(checkpoint_every_events))
        self._events_since_checkpoint = 0
        self._closed = False

    @property
    def run_id(self) -> str:
        return self.base_run.run_id

    @classmethod
    def create(
        cls,
        output_root: Path,
        initial_run: RunRecord,
        *,
        reporting_level: ReportingLevel = ReportingLevel.FULL,
        overwrite: bool = False,
        fsync: bool = True,
        checkpoint_every_events: int = 0,
    ) -> "RunStore":
        run_root = Path(output_root) / initial_run.run_id
        paths = RunPaths.under(run_root)
        if paths.root.exists():
            if not overwrite:
                raise StorageError(f"Run directory already exists: {paths.root}")
            shutil.rmtree(paths.root)
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        paths.checkpoints.mkdir(parents=True, exist_ok=True)

        started_at = initial_run.started_at or utc_now_iso()
        base_run = dataclasses.replace(
            initial_run,
            status=RunStatus.RUNNING,
            started_at=started_at,
            completed_at=None,
        )
        id_factory = IdFactory.restore(base_run.run_id, base_run.id_counters)
        event_writer = EventLogWriter(
            path=paths.events,
            run_id=base_run.run_id,
            id_factory=id_factory,
            fsync=fsync,
        )
        store = cls(
            paths=paths,
            base_run=base_run,
            reporting_level=reporting_level,
            id_factory=id_factory,
            event_writer=event_writer,
            fsync=fsync,
            checkpoint_every_events=checkpoint_every_events,
        )
        store._write_static_files()
        store.checkpoint(metadata={"checkpoint_reason": "run_created"})
        return store

    @classmethod
    def open(
        cls,
        run_directory: Path,
        *,
        fsync: bool = True,
        repair_truncated_tail: bool = True,
        checkpoint_every_events: int = 0,
    ) -> "RunStore":
        paths = RunPaths.under(Path(run_directory))
        if not paths.manifest.exists():
            raise StorageError(f"Missing run manifest: {paths.manifest}")
        if repair_truncated_tail:
            repair_truncated_jsonl_tail(paths.events)

        manifest = RunRecord.from_dict(strict_json_load(paths.manifest))
        checkpoint_run = None
        if paths.latest_checkpoint.exists():
            checkpoint_run = RunRecord.from_dict(strict_json_load(paths.latest_checkpoint))
        base_run = checkpoint_run or manifest

        reporting_text = str(base_run.metadata.get("reporting_level", ReportingLevel.FULL.value))
        try:
            reporting_level = ReportingLevel(reporting_text)
        except ValueError:
            reporting_level = ReportingLevel.FULL

        id_factory = IdFactory.restore(base_run.run_id, base_run.id_counters)
        writer = EventLogWriter.restore(
            paths.events,
            base_run.run_id,
            id_factory,
            fsync=fsync,
            repair_truncated_tail=False,
        )
        return cls(
            paths=paths,
            base_run=manifest,
            reporting_level=reporting_level,
            id_factory=id_factory,
            event_writer=writer,
            fsync=fsync,
            checkpoint_every_events=checkpoint_every_events,
        )

    def new_id(self, kind: EntityKind) -> str:
        self._ensure_open()
        return self.id_factory.new(kind)

    def append(
        self,
        payload: CanonicalRecord,
        *,
        event_kind: EventKind | None = None,
        pass_id: str | None = None,
        minimum_level: ReportingLevel = ReportingLevel.STANDARD,
    ) -> EventEnvelope | None:
        self._ensure_open()
        if _REPORTING_RANK[self.reporting_level] < _REPORTING_RANK[minimum_level]:
            return None
        event = self.event_writer.append(
            payload,
            event_kind=event_kind,
            pass_id=pass_id,
        )
        self._events_since_checkpoint += 1
        if (
            self.checkpoint_every_events > 0
            and self._events_since_checkpoint >= self.checkpoint_every_events
        ):
            self.checkpoint(metadata={"checkpoint_reason": "event_interval"})
        return event

    def read_events(self) -> tuple[EventEnvelope, ...]:
        return EventLogReader(self.paths.events).read_all()

    def current_snapshot(
        self,
        *,
        status: RunStatus = RunStatus.RUNNING,
        completed_at: str | None = None,
        final_predictions: Mapping[str, Any] | None = None,
        warnings: tuple[str, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_events: bool = False,
    ) -> RunRecord:
        merged_metadata = {"reporting_level": self.reporting_level.value}
        if metadata:
            merged_metadata.update(metadata)
        return consolidate_run(
            self.base_run,
            self.read_events(),
            status=status,
            completed_at=completed_at,
            id_counters=self.id_factory.snapshot(),
            final_predictions=final_predictions,
            warnings=warnings,
            metadata=merged_metadata,
            include_events=include_events,
        )

    def checkpoint(
        self,
        *,
        final_predictions: Mapping[str, Any] | None = None,
        warnings: tuple[str, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        self._ensure_open()
        snapshot = self.current_snapshot(
            status=RunStatus.RUNNING,
            completed_at=None,
            final_predictions=final_predictions,
            warnings=warnings,
            metadata=metadata,
        )
        atomic_write_json(self.paths.latest_checkpoint, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.partial_run, snapshot, fsync=self.fsync)
        self._write_summary(snapshot)
        self._events_since_checkpoint = 0
        return snapshot

    def finalize_success(
        self,
        *,
        final_predictions: Mapping[str, Any],
        completed_at: str | None = None,
        warnings: tuple[str, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        self._ensure_open()
        snapshot = self.current_snapshot(
            status=RunStatus.SUCCEEDED,
            completed_at=completed_at or utc_now_iso(),
            final_predictions=final_predictions,
            warnings=warnings,
            metadata=metadata,
            include_events=True,
        )
        atomic_write_json(self.paths.latest_checkpoint, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.partial_run, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.final_run, snapshot, fsync=self.fsync)
        self._write_summary(snapshot)
        self._closed = True
        return snapshot

    def finalize_failure(
        self,
        exception: BaseException,
        *,
        component: str = "run",
        pass_id: str | None = None,
        recoverable: bool = False,
        fallback_action: str | None = None,
        final_predictions: Mapping[str, Any] | None = None,
        completed_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        self._ensure_open()
        error = ErrorEventRecord(
            error_id=self.new_id(EntityKind.ERROR),
            run_id=self.run_id,
            severity=ErrorSeverity.ERROR if recoverable else ErrorSeverity.FATAL,
            component=component,
            message=str(exception) or type(exception).__name__,
            exception_type=type(exception).__name__,
            traceback="".join(
                traceback_module.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
            pass_id=pass_id,
            recoverable=recoverable,
            fallback_action=fallback_action,
        )
        self.append(error, minimum_level=ReportingLevel.MINIMAL)
        snapshot = self.current_snapshot(
            status=RunStatus.FAILED,
            completed_at=completed_at or utc_now_iso(),
            final_predictions=final_predictions,
            metadata=metadata,
            include_events=True,
        )
        atomic_write_json(self.paths.latest_checkpoint, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.partial_run, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.final_run, snapshot, fsync=self.fsync)
        self._write_summary(snapshot)
        self._closed = True
        return snapshot

    def mark_interrupted(
        self,
        *,
        reason: str = "Run interrupted before completion",
        final_predictions: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        self._ensure_open()
        error = ErrorEventRecord(
            error_id=self.new_id(EntityKind.ERROR),
            run_id=self.run_id,
            severity=ErrorSeverity.WARNING,
            component="run",
            message=reason,
            exception_type=None,
            traceback=None,
            recoverable=True,
        )
        self.append(error, minimum_level=ReportingLevel.MINIMAL)
        snapshot = self.current_snapshot(
            status=RunStatus.INTERRUPTED,
            completed_at=utc_now_iso(),
            final_predictions=final_predictions,
            include_events=True,
        )
        atomic_write_json(self.paths.latest_checkpoint, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.partial_run, snapshot, fsync=self.fsync)
        atomic_write_json(self.paths.final_run, snapshot, fsync=self.fsync)
        self._write_summary(snapshot)
        self._closed = True
        return snapshot

    def add_artifact_file(
        self,
        source_path: Path,
        *,
        kind: ArtifactKind,
        relative_path: str | Path | None = None,
        media_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        move: bool = False,
        minimum_level: ReportingLevel = ReportingLevel.STANDARD,
    ) -> ArtifactRef:
        self._ensure_open()
        source = Path(source_path)
        if not source.is_file():
            raise StorageError(f"Artifact source is not a file: {source}")
        destination_relative = safe_relative_path(
            relative_path or Path(kind.value) / source.name
        )
        destination = self.paths.artifacts / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise StorageError(f"Artifact destination already exists: {destination}")
        if move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)
        artifact = ArtifactRef(
            artifact_id=self.new_id(EntityKind.ARTIFACT),
            kind=kind,
            relative_path=str(Path("artifacts") / destination_relative),
            media_type=media_type
            or mimetypes.guess_type(destination.name)[0]
            or "application/octet-stream",
            sha256=sha256_file(destination),
            size_bytes=destination.stat().st_size,
            width=width,
            height=height,
            metadata=dict(metadata or {}),
        )
        self.append(
            artifact,
            event_kind=EventKind.ARTIFACT,
            minimum_level=minimum_level,
        )
        return artifact

    def write_artifact_bytes(
        self,
        data: bytes,
        *,
        kind: ArtifactKind,
        relative_path: str | Path,
        media_type: str = "application/octet-stream",
        width: int | None = None,
        height: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        minimum_level: ReportingLevel = ReportingLevel.STANDARD,
    ) -> ArtifactRef:
        self._ensure_open()
        destination_relative = safe_relative_path(relative_path)
        destination = self.paths.artifacts / destination_relative
        if destination.exists():
            raise StorageError(f"Artifact destination already exists: {destination}")
        atomic_write_bytes(destination, data, fsync=self.fsync)
        artifact = ArtifactRef(
            artifact_id=self.new_id(EntityKind.ARTIFACT),
            kind=kind,
            relative_path=str(Path("artifacts") / destination_relative),
            media_type=media_type,
            sha256=sha256_file(destination),
            size_bytes=len(data),
            width=width,
            height=height,
            metadata=dict(metadata or {}),
        )
        self.append(
            artifact,
            event_kind=EventKind.ARTIFACT,
            minimum_level=minimum_level,
        )
        return artifact

    def _write_static_files(self) -> None:
        manifest_metadata = dict(self.base_run.metadata)
        manifest_metadata["reporting_level"] = self.reporting_level.value
        manifest = dataclasses.replace(self.base_run, metadata=manifest_metadata)
        self.base_run = manifest
        atomic_write_json(self.paths.manifest, manifest, fsync=self.fsync)
        atomic_write_json(
            self.paths.config,
            {
                "schema_version": manifest.to_dict()["schema_version"],
                "run_id": manifest.run_id,
                "config_sha256": manifest.config_sha256,
                "resolved_config": manifest.resolved_config,
                "random_seeds": manifest.random_seeds,
            },
            fsync=self.fsync,
        )
        atomic_write_json(
            self.paths.environment,
            {
                "schema_version": manifest.to_dict()["schema_version"],
                "run_id": manifest.run_id,
                "repository": manifest.repository.to_dict(),
                "environment": manifest.environment.to_dict(),
                "models": [model.to_dict() for model in manifest.models],
            },
            fsync=self.fsync,
        )

    def _write_summary(self, snapshot: RunRecord) -> None:
        metrics = {
            evaluation.evaluator_name: dict(evaluation.metrics)
            for evaluation in snapshot.evaluations
        }
        atomic_write_json(
            self.paths.summary,
            {
                "schema_version": snapshot.to_dict()["schema_version"],
                "run_id": snapshot.run_id,
                "status": snapshot.status.value,
                "reporting_level": self.reporting_level.value,
                "updated_at": utc_now_iso(),
                "event_count": snapshot.metadata.get("event_count", 0),
                "last_event_sequence": snapshot.metadata.get(
                    "last_event_sequence", 0
                ),
                "pass_count": len(snapshot.passes),
                "graph_node_count": len(snapshot.final_graph),
                "artifact_count": len(snapshot.artifacts),
                "error_count": len(snapshot.errors),
                "final_predictions": snapshot.final_predictions,
                "evaluation_metrics": metrics,
            },
            fsync=self.fsync,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("RunStore has already been finalized")


__all__ = ["ReportingLevel", "RunPaths", "RunStore"]

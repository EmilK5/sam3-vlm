"""Deep validation for self-contained canonical experiment directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from graph import OrchardGraph
from provenance.contracts import (
    BeliefUpdateRecord,
    CandidateActionSetRecord,
    DedupComparisonRecord,
    EncodedObservationRecord,
    EvaluationResultRecord,
    GraphNodeSnapshotRecord,
    InformationGainRecord,
    PassRecord,
    RawDetectionRecord,
    RegistrationDecisionRecord,
    RunRecord,
    Sam3CallRecord,
    StoppingDecisionRecord,
    SurrogateKernelRecord,
    TileRecord,
)
from provenance.events import EventLogReader
from provenance.io import sha256_file, sha256_json, strict_json_load
from provenance.schema import RunStatus
from validation.reproducibility import semantic_run_fingerprint

_REQUIRED_FILES = (
    "manifest.json",
    "events.jsonl",
    "config.json",
    "environment.json",
    "run.json",
    "summary.json",
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    entity_id: str | None = None


@dataclass(frozen=True)
class RunValidationReport:
    run_directory: str
    run_id: str | None
    valid: bool
    issues: tuple[ValidationIssue, ...]
    counts: Mapping[str, int] = field(default_factory=dict)
    semantic_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_directory": self.run_directory,
            "run_id": self.run_id,
            "valid": self.valid,
            "issues": [issue.__dict__ for issue in self.issues],
            "counts": dict(self.counts),
            "semantic_fingerprint": self.semantic_fingerprint,
        }


def validate_run_directory(
    run_directory: str | Path,
    *,
    verify_artifacts: bool = True,
    verify_event_parity: bool = True,
    verify_graph_replay: bool = True,
) -> RunValidationReport:
    root = Path(run_directory)
    issues: list[ValidationIssue] = []

    def add(severity: str, code: str, message: str, *, path=None, entity_id=None):
        issues.append(ValidationIssue(severity, code, message, path, entity_id))

    for name in _REQUIRED_FILES:
        if not (root / name).is_file():
            add("error", "missing_file", f"Required file is missing: {name}", path=name)
    if any(issue.code == "missing_file" and issue.path == "run.json" for issue in issues):
        return _report(root, None, issues, {})

    try:
        run = RunRecord.from_dict(strict_json_load(root / "run.json"))
    except Exception as exc:
        add("error", "invalid_run_json", f"run.json cannot be decoded: {type(exc).__name__}: {exc}", path="run.json")
        return _report(root, None, issues, {})

    if run.status is RunStatus.RUNNING:
        add("warning", "unfinished_run", "run.json is still marked running", path="run.json")

    _validate_static_files(root, run, add)

    events = ()
    if (root / "events.jsonl").is_file():
        try:
            events = EventLogReader(root / "events.jsonl").read_all()
        except Exception as exc:
            add("error", "invalid_event_log", f"events.jsonl cannot be decoded: {type(exc).__name__}: {exc}", path="events.jsonl")
    if verify_event_parity and events:
        if tuple(events) != tuple(run.events):
            add("error", "event_parity", "run.json events differ from events.jsonl", path="events.jsonl")
        if run.metadata.get("event_count") != len(events):
            add("error", "event_count", f"metadata event_count={run.metadata.get('event_count')} but log contains {len(events)} events")
        expected_last = events[-1].sequence_number if events else 0
        if run.metadata.get("last_event_sequence") != expected_last:
            add("error", "event_sequence", f"metadata last_event_sequence={run.metadata.get('last_event_sequence')} but log ends at {expected_last}")

    _validate_lineage(run, add)
    if verify_artifacts:
        _validate_artifacts(root, run, add)
    if verify_graph_replay:
        _validate_graph_artifact(root, run, add)

    counts = {
        "events": len(events),
        "passes": len(run.passes),
        "graph_nodes": len(run.final_graph),
        "artifacts": len(run.artifacts),
        "evaluations": len(run.evaluations),
        "errors": len(run.errors),
    }
    return _report(root, run, issues, counts)


def validate_run_tree(root: str | Path, **kwargs: Any) -> tuple[RunValidationReport, ...]:
    base = Path(root)
    directories = sorted({path.parent for path in base.rglob("run.json")})
    return tuple(validate_run_directory(directory, **kwargs) for directory in directories)


def _report(root: Path, run: RunRecord | None, issues: list[ValidationIssue], counts: Mapping[str, int]) -> RunValidationReport:
    valid = not any(issue.severity == "error" for issue in issues)
    fingerprint = None
    if run is not None:
        try:
            fingerprint = semantic_run_fingerprint(run)
        except Exception as exc:
            issues.append(ValidationIssue("error", "fingerprint", f"Could not build semantic fingerprint: {exc}"))
            valid = False
    return RunValidationReport(str(root), None if run is None else run.run_id, valid, tuple(issues), dict(counts), fingerprint)


def _validate_static_files(root: Path, run: RunRecord, add) -> None:
    try:
        config = strict_json_load(root / "config.json")
        if config.get("run_id") != run.run_id:
            add("error", "config_run_id", "config.json run_id does not match run.json", path="config.json")
        if config.get("config_sha256") != run.config_sha256:
            add("error", "config_hash_record", "config.json config_sha256 does not match run.json", path="config.json")
        if config.get("resolved_config") != run.resolved_config:
            add("error", "config_payload", "config.json resolved_config does not match run.json", path="config.json")
    except Exception as exc:
        add("error", "invalid_config", f"config.json cannot be decoded: {type(exc).__name__}: {exc}", path="config.json")
    expected_hash = sha256_json(run.resolved_config)
    if expected_hash != run.config_sha256:
        add("error", "config_hash", f"Resolved configuration hash mismatch: expected {expected_hash}, recorded {run.config_sha256}")
    try:
        manifest = RunRecord.from_dict(strict_json_load(root / "manifest.json"))
        if manifest.run_id != run.run_id:
            add("error", "manifest_run_id", "manifest.json run_id does not match run.json", path="manifest.json")
        if manifest.config_sha256 != run.config_sha256:
            add("error", "manifest_config", "manifest configuration hash does not match final run", path="manifest.json")
        if manifest.dataset_sample.image_sha256 != run.dataset_sample.image_sha256:
            add("error", "manifest_image", "manifest image hash does not match final run", path="manifest.json")
    except Exception as exc:
        add("error", "invalid_manifest", f"manifest.json cannot be decoded: {type(exc).__name__}: {exc}", path="manifest.json")
    try:
        summary = strict_json_load(root / "summary.json")
        if summary.get("run_id") != run.run_id:
            add("error", "summary_run_id", "summary.json run_id does not match run.json", path="summary.json")
        if summary.get("status") != run.status.value:
            add("error", "summary_status", "summary status does not match run.json", path="summary.json")
        if summary.get("pass_count") != len(run.passes):
            add("error", "summary_pass_count", "summary pass_count does not match run.json", path="summary.json")
    except Exception as exc:
        add("error", "invalid_summary", f"summary.json cannot be decoded: {type(exc).__name__}: {exc}", path="summary.json")


def _validate_artifacts(root: Path, run: RunRecord, add) -> None:
    seen_paths: set[str] = set()
    for artifact in run.artifacts:
        if artifact.relative_path in seen_paths:
            add("error", "duplicate_artifact_path", f"Multiple artifact records use {artifact.relative_path}", entity_id=artifact.artifact_id)
        seen_paths.add(artifact.relative_path)
        path = root / artifact.relative_path
        if not path.is_file():
            add("error", "missing_artifact", f"Artifact file is missing: {artifact.relative_path}", path=artifact.relative_path, entity_id=artifact.artifact_id)
            continue
        if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
            add("error", "artifact_size", f"Artifact size mismatch for {artifact.relative_path}", path=artifact.relative_path, entity_id=artifact.artifact_id)
        if artifact.sha256 is not None and sha256_file(path) != artifact.sha256:
            add("error", "artifact_hash", f"Artifact hash mismatch for {artifact.relative_path}", path=artifact.relative_path, entity_id=artifact.artifact_id)
    input_path = root / run.dataset_sample.image_artifact.relative_path
    if input_path.is_file() and sha256_file(input_path) != run.dataset_sample.image_sha256:
        add("error", "input_image_hash", "Materialized input image hash differs from DatasetSampleRecord", path=run.dataset_sample.image_artifact.relative_path)


def _validate_graph_artifact(root: Path, run: RunRecord, add) -> None:
    artifact_id = run.final_predictions.get("final_graph_artifact_id")
    if artifact_id is None:
        add("warning", "graph_artifact_absent", "Final predictions do not reference a graph artifact")
        return
    artifact = next((item for item in run.artifacts if item.artifact_id == artifact_id), None)
    if artifact is None:
        add("error", "graph_artifact_reference", f"Unknown final graph artifact ID: {artifact_id}")
        return
    path = root / artifact.relative_path
    if not path.is_file():
        return
    try:
        graph = OrchardGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        add("error", "graph_replay", f"Final graph artifact cannot be reconstructed: {type(exc).__name__}: {exc}", path=artifact.relative_path)
        return
    snapshot_by_id = {node.graph_node_id: node for node in run.final_graph}
    if set(graph.nodes) != set(snapshot_by_id):
        add("error", "graph_node_set", "Final graph artifact node IDs differ from run.final_graph", path=artifact.relative_path)
        return
    for node_id, node in graph.nodes.items():
        snapshot = snapshot_by_id[node_id]
        if tuple(float(value) for value in node.box) != tuple(snapshot.box):
            add("error", "graph_box", f"Graph box differs for {node_id}", entity_id=node_id)
        if node.belief is not None and node.belief.as_mapping() != dict(snapshot.posterior):
            add("error", "graph_posterior", f"Graph posterior differs for {node_id}", entity_id=node_id)


def _validate_lineage(run: RunRecord, add) -> None:
    full = str(run.metadata.get("reporting_level", "full")) == "full"
    all_nodes = {node.graph_node_id for node in run.final_graph}
    passes = {record.pass_id for record in run.passes}
    actions: set[str] = set()
    qwen_calls: set[str] = set()
    sam3_calls: set[str] = set()
    tiles: set[str] = set()
    detections: set[str] = set()
    dedups: set[str] = set()
    kernels: set[str] = set()
    observations: set[str] = set()

    for event in run.events:
        if event.pass_id is not None and event.pass_id not in passes:
            add("error", "event_pass_reference", f"Event references unknown pass {event.pass_id}", entity_id=event.event_id)
        payload = event.payload
        if isinstance(payload, CandidateActionSetRecord):
            actions.update(action.action_id for action in payload.actions)
        elif isinstance(payload, Sam3CallRecord):
            sam3_calls.add(payload.sam3_call_id)
        elif isinstance(payload, TileRecord):
            tiles.add(payload.tile_id)
        elif isinstance(payload, RawDetectionRecord):
            detections.add(payload.raw_detection_id)
        elif isinstance(payload, DedupComparisonRecord):
            dedups.add(payload.dedup_decision_id)
        elif isinstance(payload, SurrogateKernelRecord):
            kernels.add(payload.kernel_id)
        elif isinstance(payload, EncodedObservationRecord):
            observations.add(payload.observation_id)
        qwen_id = getattr(payload, "qwen_call_id", None)
        if isinstance(qwen_id, str):
            qwen_calls.add(qwen_id)

    for record in run.passes:
        if record.candidate_action_set is not None:
            actions.update(action.action_id for action in record.candidate_action_set.actions)
        qwen_calls.update(item.qwen_call_id for item in record.qwen_calls)
        sam3_calls.update(item.sam3_call_id for item in record.sam3_calls)
        tiles.update(item.tile_id for item in record.tiles)
        detections.update(item.raw_detection_id for item in record.raw_detections)
        dedups.update(item.dedup_decision_id for item in record.dedup_comparisons)
        if record.kernel is not None:
            kernels.add(record.kernel.kernel_id)
        if record.observation is not None:
            observations.add(record.observation.observation_id)
        all_nodes.update(node.graph_node_id for node in record.graph_before)
        all_nodes.update(node.graph_node_id for node in record.graph_after)

    def require(identifier: str | None, known: set[str], code: str, owner: str):
        if identifier is None:
            return
        if identifier not in known:
            severity = "error" if full else "warning"
            add(severity, code, f"{owner} references unrecorded entity {identifier}", entity_id=owner)

    for record in run.passes:
        for call in record.sam3_calls:
            require(call.tile_id, tiles, "sam3_tile_reference", call.sam3_call_id)
            for detection_id in call.raw_detection_ids:
                require(detection_id, detections, "sam3_detection_reference", call.sam3_call_id)
            for node_id in (*call.positive_exemplar_node_ids, *call.negative_exemplar_node_ids):
                require(node_id, all_nodes, "sam3_exemplar_reference", call.sam3_call_id)
        for tile in record.tiles:
            for call_id in tile.sam3_call_ids:
                require(call_id, sam3_calls, "tile_call_reference", tile.tile_id)
            for detection_id in tile.raw_detection_ids:
                require(detection_id, detections, "tile_detection_reference", tile.tile_id)
        for detection in record.raw_detections:
            require(detection.sam3_call_id, sam3_calls, "detection_call_reference", detection.raw_detection_id)
            require(detection.tile_id, tiles, "detection_tile_reference", detection.raw_detection_id)
        for comparison in record.dedup_comparisons:
            require(comparison.new_detection_id, detections, "dedup_new_detection", comparison.dedup_decision_id)
            require(comparison.existing_detection_id, detections, "dedup_existing_detection", comparison.dedup_decision_id)
            require(comparison.existing_node_id, all_nodes, "dedup_node_reference", comparison.dedup_decision_id)
        for registration in record.registrations:
            require(registration.raw_detection_id, detections, "registration_detection", registration.registration_id)
            require(registration.graph_node_id, all_nodes, "registration_node", registration.registration_id)
            require(registration.selected_dedup_decision_id, dedups, "registration_dedup", registration.registration_id)
        for info in record.information_gain_records:
            require(info.action_id, actions, "ig_action_reference", info.information_gain_id)
            require(info.kernel_id, kernels, "ig_kernel_reference", info.information_gain_id)
        if record.observation is not None:
            require(record.observation.action_id, actions, "observation_action_reference", record.observation.observation_id)
            require(record.observation.target_node_id, all_nodes, "observation_node_reference", record.observation.observation_id)
            for detection_id in record.observation.matched_detection_ids:
                require(detection_id, detections, "observation_detection_reference", record.observation.observation_id)
        if record.kernel is not None:
            require(record.kernel.action_id, actions, "kernel_action_reference", record.kernel.kernel_id)
        if record.belief_update is not None:
            update = record.belief_update
            require(update.graph_node_id, all_nodes, "belief_node_reference", update.belief_update_id)
            require(update.action_id, actions, "belief_action_reference", update.belief_update_id)
            require(update.observation_id, observations, "belief_observation_reference", update.belief_update_id)
            require(update.kernel_id, kernels, "belief_kernel_reference", update.belief_update_id)
        if record.stopping_decision is not None:
            require(record.stopping_decision.graph_node_id, all_nodes, "stopping_node_reference", record.stopping_decision.stopping_id)

    for evaluation in run.evaluations:
        if evaluation.run_id != run.run_id:
            add("error", "evaluation_run_reference", f"Evaluation {evaluation.evaluation_id} references another run")
        if evaluation.image_id != run.dataset_sample.image_id:
            add("error", "evaluation_image_reference", f"Evaluation {evaluation.evaluation_id} references another image")


__all__ = [
    "RunValidationReport",
    "ValidationIssue",
    "validate_run_directory",
    "validate_run_tree",
]

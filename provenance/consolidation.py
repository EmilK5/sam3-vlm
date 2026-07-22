"""Deterministic consolidation of an event stream into a canonical RunRecord."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any, Iterable, Iterator, Mapping

from provenance.base import CanonicalRecord, ContractError
from provenance.contracts import (
    ArtifactRef,
    ErrorEventRecord,
    EvaluationResultRecord,
    EventEnvelope,
    GraphNodeSnapshotRecord,
    PassRecord,
    RegistrationDecisionRecord,
    RunRecord,
)
from provenance.schema import RunStatus


def consolidate_run(
    base_run: RunRecord,
    events: Iterable[EventEnvelope],
    *,
    status: RunStatus | None = None,
    completed_at: str | None = None,
    id_counters: Mapping[str, int] | None = None,
    final_predictions: Mapping[str, Any] | None = None,
    final_graph: tuple[GraphNodeSnapshotRecord, ...] | None = None,
    warnings: tuple[str, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_events: bool = False,
) -> RunRecord:
    """Fold events into a deterministic current/final run snapshot."""

    pass_by_id: OrderedDict[str, PassRecord] = OrderedDict(
        (record.pass_id, record) for record in base_run.passes
    )
    graph_by_id: OrderedDict[str, GraphNodeSnapshotRecord] = OrderedDict(
        (record.graph_node_id, record) for record in base_run.final_graph
    )
    evaluation_by_id: OrderedDict[str, EvaluationResultRecord] = OrderedDict(
        (record.evaluation_id, record) for record in base_run.evaluations
    )
    artifact_by_id: OrderedDict[str, ArtifactRef] = OrderedDict(
        (record.artifact_id, record) for record in base_run.artifacts
    )
    error_by_id: OrderedDict[str, ErrorEventRecord] = OrderedDict(
        (record.error_id, record) for record in base_run.errors
    )

    event_records = tuple(events)
    event_count = 0
    last_sequence = 0
    for event in event_records:
        if event.run_id != base_run.run_id:
            raise ContractError(
                f"Event {event.event_id} belongs to {event.run_id}, not {base_run.run_id}"
            )
        event_count += 1
        last_sequence = event.sequence_number
        payload = event.payload

        if isinstance(payload, PassRecord):
            for node in payload.graph_after:
                graph_by_id[node.graph_node_id] = node
            # Keep graph deltas in the crash-recovery event stream, but avoid
            # duplicating them inside every pass in the canonical run.json.
            pass_by_id[payload.pass_id] = dataclasses.replace(
                payload, graph_before=(), graph_after=()
            )
        elif isinstance(payload, GraphNodeSnapshotRecord):
            graph_by_id[payload.graph_node_id] = payload
        elif isinstance(payload, RegistrationDecisionRecord):
            if payload.node_after is not None:
                graph_by_id[payload.node_after.graph_node_id] = payload.node_after
        elif isinstance(payload, EvaluationResultRecord):
            evaluation_by_id[payload.evaluation_id] = payload
        elif isinstance(payload, ErrorEventRecord):
            error_by_id[payload.error_id] = payload
        elif isinstance(payload, ArtifactRef):
            artifact_by_id[payload.artifact_id] = payload

        for artifact in _walk_artifacts(payload):
            artifact_by_id[artifact.artifact_id] = artifact

    merged_metadata = dict(base_run.metadata)
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata.update(
        {
            "event_count": event_count,
            "last_event_sequence": last_sequence,
        }
    )

    merged_warnings = tuple(base_run.warnings if warnings is None else warnings)
    resolved_status = status or base_run.status
    resolved_completed_at = completed_at
    if resolved_completed_at is None and resolved_status is base_run.status:
        resolved_completed_at = base_run.completed_at

    return dataclasses.replace(
        base_run,
        status=resolved_status,
        completed_at=resolved_completed_at,
        id_counters=dict(id_counters or base_run.id_counters),
        passes=tuple(sorted(pass_by_id.values(), key=lambda item: (item.pass_index, item.pass_id))),
        final_graph=tuple(
            sorted(
                graph_by_id.values() if final_graph is None else final_graph,
                key=lambda item: item.graph_node_id,
            )
        ),
        final_predictions=dict(
            base_run.final_predictions if final_predictions is None else final_predictions
        ),
        evaluations=tuple(
            sorted(evaluation_by_id.values(), key=lambda item: item.evaluation_id)
        ),
        artifacts=tuple(sorted(artifact_by_id.values(), key=lambda item: item.artifact_id)),
        errors=tuple(sorted(error_by_id.values(), key=lambda item: item.error_id)),
        events=event_records if include_events else (),
        warnings=merged_warnings,
        metadata=merged_metadata,
    )


def _walk_artifacts(value: Any) -> Iterator[ArtifactRef]:
    if isinstance(value, ArtifactRef):
        yield value
        return
    if isinstance(value, CanonicalRecord) or dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk_artifacts(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_artifacts(item)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _walk_artifacts(item)


__all__ = ["consolidate_run"]

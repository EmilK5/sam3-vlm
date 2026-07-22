"""Append-only event stream support for provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from provenance.base import CanonicalRecord, ContractError, utc_now_iso
from provenance.contracts import (
    ArtifactRef,
    BeliefUpdateRecord,
    CandidateActionSetRecord,
    CostSnapshotRecord,
    DedupComparisonRecord,
    EncodedObservationRecord,
    ErrorEventRecord,
    EvaluationResultRecord,
    EventEnvelope,
    GraphNodeSnapshotRecord,
    InformationGainRecord,
    PassRecord,
    QwenCallRecord,
    RawDetectionRecord,
    RegistrationDecisionRecord,
    RunRecord,
    Sam3CallRecord,
    SensingActionRecord,
    StoppingDecisionRecord,
    SurrogateKernelRecord,
    TileRecord,
)
from provenance.ids import EntityKind, IdFactory
from provenance.io import (
    JsonLineCorruptionError,
    append_json_line,
    iter_json_lines,
    repair_truncated_jsonl_tail,
)
from provenance.schema import EventKind


_EVENT_KIND_BY_TYPE: tuple[tuple[type[CanonicalRecord], EventKind], ...] = (
    (RunRecord, EventKind.RUN),
    (PassRecord, EventKind.PASS),
    (SensingActionRecord, EventKind.ACTION),
    (CandidateActionSetRecord, EventKind.ACTION),
    (QwenCallRecord, EventKind.QWEN_CALL),
    (Sam3CallRecord, EventKind.SAM3_CALL),
    (RawDetectionRecord, EventKind.DETECTION),
    (TileRecord, EventKind.TILE),
    (DedupComparisonRecord, EventKind.DEDUP),
    (RegistrationDecisionRecord, EventKind.REGISTRATION),
    (GraphNodeSnapshotRecord, EventKind.GRAPH),
    (EncodedObservationRecord, EventKind.OBSERVATION),
    (SurrogateKernelRecord, EventKind.KERNEL),
    (InformationGainRecord, EventKind.INFORMATION_GAIN),
    (BeliefUpdateRecord, EventKind.BELIEF_UPDATE),
    (StoppingDecisionRecord, EventKind.STOPPING),
    (CostSnapshotRecord, EventKind.COST),
    (EvaluationResultRecord, EventKind.EVALUATION),
    (ArtifactRef, EventKind.ARTIFACT),
    (ErrorEventRecord, EventKind.ERROR),
)


def infer_event_kind(payload: CanonicalRecord) -> EventKind:
    for record_type, event_kind in _EVENT_KIND_BY_TYPE:
        if isinstance(payload, record_type):
            return event_kind
    raise ContractError(
        f"No EventKind mapping is registered for {type(payload).__name__}"
    )


def infer_pass_id(payload: CanonicalRecord) -> str | None:
    value = getattr(payload, "pass_id", None)
    return value if isinstance(value, str) else None


@dataclass
class EventLogWriter:
    path: Path
    run_id: str
    id_factory: IdFactory
    fsync: bool = True
    sequence_number: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.id_factory.run_id != self.run_id:
            raise ContractError("Event writer and ID factory must reference the same run")

    def append(
        self,
        payload: CanonicalRecord,
        *,
        event_kind: EventKind | None = None,
        pass_id: str | None = None,
        created_at: str | None = None,
    ) -> EventEnvelope:
        payload_run_id = getattr(payload, "run_id", None)
        if payload_run_id is not None and payload_run_id != self.run_id:
            raise ContractError(
                f"Payload run_id {payload_run_id!r} does not match {self.run_id!r}"
            )

        payload_pass_id = infer_pass_id(payload)
        if pass_id is not None and payload_pass_id is not None and pass_id != payload_pass_id:
            raise ContractError("Explicit pass_id conflicts with payload pass_id")
        resolved_pass_id = pass_id or payload_pass_id

        next_sequence = self.sequence_number + 1
        envelope = EventEnvelope(
            event_id=self.id_factory.new(EntityKind.EVENT),
            run_id=self.run_id,
            sequence_number=next_sequence,
            event_kind=event_kind or infer_event_kind(payload),
            payload=payload,
            pass_id=resolved_pass_id,
            created_at=created_at or utc_now_iso(),
        )
        append_json_line(self.path, envelope, fsync=self.fsync)
        self.sequence_number = next_sequence
        return envelope

    @classmethod
    def restore(
        cls,
        path: Path,
        run_id: str,
        id_factory: IdFactory,
        *,
        fsync: bool = True,
        repair_truncated_tail: bool = True,
    ) -> "EventLogWriter":
        path = Path(path)
        if repair_truncated_tail:
            repair_truncated_jsonl_tail(path)
        sequence = 0
        for event in EventLogReader(path).read_all():
            if event.run_id != run_id:
                raise ContractError(
                    f"Event log contains run {event.run_id!r}; expected {run_id!r}"
                )
            sequence = max(sequence, event.sequence_number)
            id_factory.observe(event)
        return cls(
            path=path,
            run_id=run_id,
            id_factory=id_factory,
            fsync=fsync,
            sequence_number=sequence,
        )


@dataclass(frozen=True)
class EventLogReader:
    path: Path

    def __iter__(self) -> Iterator[EventEnvelope]:
        yield from self.read_all()

    def read_all(
        self,
        *,
        tolerate_truncated_tail: bool = False,
    ) -> tuple[EventEnvelope, ...]:
        events: list[EventEnvelope] = []
        expected_sequence = 1
        seen_event_ids: set[str] = set()
        for line_number, payload in iter_json_lines(
            Path(self.path), tolerate_truncated_tail=tolerate_truncated_tail
        ):
            try:
                event = EventEnvelope.from_dict(payload)
            except Exception as exc:
                raise JsonLineCorruptionError(
                    f"Invalid event envelope at {self.path}:{line_number}: {exc}"
                ) from exc
            if event.sequence_number != expected_sequence:
                raise JsonLineCorruptionError(
                    f"Expected sequence {expected_sequence} at {self.path}:{line_number}, "
                    f"got {event.sequence_number}"
                )
            if event.event_id in seen_event_ids:
                raise JsonLineCorruptionError(
                    f"Duplicate event_id {event.event_id!r} at {self.path}:{line_number}"
                )
            events.append(event)
            seen_event_ids.add(event.event_id)
            expected_sequence += 1
        return tuple(events)


__all__ = [
    "EventLogReader",
    "EventLogWriter",
    "infer_event_kind",
    "infer_pass_id",
]

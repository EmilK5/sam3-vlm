"""Stable, namespaced identifiers for all entities emitted by a run."""

from __future__ import annotations

import dataclasses
import re
import secrets
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from provenance.base import ContractError, require_non_empty


class EntityKind(str, Enum):
    RUN = "run"
    IMAGE = "image"
    PASS = "pass"
    ACTION = "action"
    QWEN_CALL = "qwen"
    SAM3_CALL = "sam3"
    TILE = "tile"
    RAW_DETECTION = "det"
    CANDIDATE = "candidate"
    GRAPH_NODE = "node"
    DEDUP_DECISION = "dedup"
    REGISTRATION = "registration"
    VERIFICATION = "verification"
    OBSERVATION = "observation"
    BELIEF_UPDATE = "belief"
    INFORMATION_GAIN = "ig"
    KERNEL = "kernel"
    STOPPING = "stop"
    ARTIFACT = "artifact"
    EVALUATION = "evaluation"
    ERROR = "error"
    EVENT = "event"


_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def create_run_id(
    *,
    now: datetime | None = None,
    random_token: str | None = None,
) -> str:
    """Create a sortable run ID.

    ``random_token`` is injectable so unit tests and reproducible orchestration
    can generate deterministic IDs without monkeypatching global randomness.
    """

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    timestamp = current.strftime("%Y%m%dT%H%M%S%fZ")
    token = random_token or secrets.token_hex(4)
    token = _sanitize_component(token)
    return f"{EntityKind.RUN.value}_{timestamp}_{token}"


def validate_entity_id(value: str, *, expected_kind: EntityKind | None = None) -> None:
    require_non_empty(value, "entity_id")
    if not _ID_RE.fullmatch(value):
        raise ContractError(f"Invalid entity ID: {value!r}")
    if expected_kind is not None and not value.startswith(f"{expected_kind.value}_"):
        raise ContractError(
            f"Expected an ID with prefix {expected_kind.value!r}, got {value!r}"
        )


@dataclass
class IdFactory:
    """Run-scoped deterministic ID generator.

    Child IDs include a short token derived from the run ID and a monotonically
    increasing counter.  They remain unique when multiple run directories are
    aggregated while staying compact and human-readable.
    """

    run_id: str
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        validate_entity_id(self.run_id, expected_kind=EntityKind.RUN)
        self._run_token = _sanitize_component(self.run_id.rsplit("_", 1)[-1])
        normalized = defaultdict(int)
        for key, value in dict(self.counters).items():
            if int(value) < 0:
                raise ContractError("ID counters must be non-negative")
            normalized[str(key)] = int(value)
        self.counters = normalized

    def new(self, kind: EntityKind) -> str:
        if kind is EntityKind.RUN:
            raise ContractError("Use create_run_id() to create run IDs")
        key = kind.value
        self.counters[key] += 1
        return f"{key}_{self._run_token}_{self.counters[key]:06d}"

    def snapshot(self) -> dict[str, int]:
        return dict(sorted(self.counters.items()))

    def observe(self, value: object) -> None:
        """Advance counters past any run-scoped child IDs found in ``value``.

        This is used during crash recovery: events written after the most recent
        checkpoint may contain action, detection, or node IDs whose counters were
        not yet persisted separately.
        """

        self._observe(value, seen=set())

    def _observe(self, value: object, *, seen: set[int]) -> None:
        if isinstance(value, str):
            self._observe_string(value)
            return
        if value is None or isinstance(value, (bool, int, float, bytes)):
            return

        identity = id(value)
        if identity in seen:
            return

        if dataclasses.is_dataclass(value):
            seen.add(identity)
            for item in dataclasses.fields(value):
                self._observe(getattr(value, item.name), seen=seen)
            return
        if isinstance(value, Mapping):
            seen.add(identity)
            for key, item in value.items():
                self._observe(key, seen=seen)
                self._observe(item, seen=seen)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            seen.add(identity)
            for item in value:
                self._observe(item, seen=seen)

    def _observe_string(self, value: str) -> None:
        for kind in EntityKind:
            if kind is EntityKind.RUN:
                continue
            prefix = f"{kind.value}_{self._run_token}_"
            if not value.startswith(prefix):
                continue
            suffix = value[len(prefix):]
            if suffix.isdigit():
                self.counters[kind.value] = max(
                    self.counters.get(kind.value, 0), int(suffix)
                )
            return

    @classmethod
    def restore(cls, run_id: str, counters: Mapping[str, int]) -> "IdFactory":
        return cls(run_id=run_id, counters=dict(counters))


def _sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value)).strip("-._:")
    if not cleaned:
        raise ContractError("ID component becomes empty after sanitization")
    return cleaned


__all__ = [
    "EntityKind",
    "IdFactory",
    "create_run_id",
    "validate_entity_id",
]

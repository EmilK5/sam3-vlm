"""Stable semantic fingerprints for comparing repeated experiment runs.

The canonical provenance record intentionally stores timestamps, latencies,
hostnames, run-scoped IDs, and artifact paths.  Those values are useful for
forensics but should not make two otherwise identical deterministic runs look
different.  This module removes only explicitly nondeterministic fields and
renames entity IDs by first occurrence before hashing the remaining trace.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from provenance.contracts import RunRecord
from provenance.io import strict_json_load

_ID_PATTERN = re.compile(r"^(run|image|pass|action|qwen|sam3|tile|tiling|det|candidate|node|dedup|registration|verification|observation|belief|ig|kernel|stop|artifact|evaluation|error|event)_[A-Za-z0-9_.:-]+$")
_IGNORED_KEYS = {
    "created_at",
    "started_at",
    "completed_at",
    "updated_at",
    "latency_seconds",
    "runtime_seconds",
    "monetary_cost",
    "traceback",
    "hostname",
    "source_path",
    "run_directory",
    "checkpoint_reason",
}
_IGNORED_TOP_LEVEL = {
    "repository",
    "environment",
    "id_counters",
    "events",
    "artifacts",
    "errors",
    "warnings",
}


@dataclass(frozen=True)
class ReproducibilityComparison:
    equal: bool
    left_fingerprint: str
    right_fingerprint: str
    differences: tuple[str, ...]


def _load_run(value: RunRecord | str | Path) -> RunRecord:
    if isinstance(value, RunRecord):
        return value
    path = Path(value)
    if path.is_dir():
        path = path / "run.json"
    return RunRecord.from_dict(strict_json_load(path))


def semantic_run_trace(value: RunRecord | str | Path) -> Mapping[str, Any]:
    """Return a strict-JSON semantic trace with nondeterminism normalized."""

    run = _load_run(value)
    payload = run.to_dict()
    for key in _IGNORED_TOP_LEVEL:
        payload.pop(key, None)
    payload.pop("schema_version", None)
    payload.pop("record_type", None)
    normalizer = _Normalizer()
    return normalizer.normalize(payload)


def semantic_run_fingerprint(value: RunRecord | str | Path) -> str:
    trace = semantic_run_trace(value)
    encoded = json.dumps(
        trace,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_run_semantics(
    left: RunRecord | str | Path,
    right: RunRecord | str | Path,
    *,
    max_differences: int = 50,
) -> ReproducibilityComparison:
    left_trace = semantic_run_trace(left)
    right_trace = semantic_run_trace(right)
    differences: list[str] = []
    _diff(left_trace, right_trace, path="$", output=differences, limit=max_differences)
    return ReproducibilityComparison(
        equal=not differences,
        left_fingerprint=semantic_run_fingerprint(left),
        right_fingerprint=semantic_run_fingerprint(right),
        differences=tuple(differences),
    )


class _Normalizer:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def normalize(self, value: Any, *, key: str | None = None) -> Any:
        if key in _IGNORED_KEYS:
            return None
        if isinstance(value, Mapping):
            result = {}
            for child_key in sorted(value):
                if child_key in _IGNORED_KEYS:
                    continue
                normalized = self.normalize(value[child_key], key=str(child_key))
                result[str(child_key)] = normalized
            return result
        if isinstance(value, (list, tuple)):
            return [self.normalize(item) for item in value]
        if isinstance(value, str) and _ID_PATTERN.fullmatch(value):
            return self._canonical_id(value)
        return value

    def _canonical_id(self, value: str) -> str:
        existing = self._ids.get(value)
        if existing is not None:
            return existing
        prefix = value.split("_", 1)[0]
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        canonical = f"{prefix}#{self._counts[prefix]:06d}"
        self._ids[value] = canonical
        return canonical


def _diff(left: Any, right: Any, *, path: str, output: list[str], limit: int) -> None:
    if len(output) >= limit:
        return
    if type(left) is not type(right):
        output.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, Mapping):
        left_keys, right_keys = set(left), set(right)
        for key in sorted(left_keys - right_keys):
            output.append(f"{path}.{key}: missing on right")
            if len(output) >= limit:
                return
        for key in sorted(right_keys - left_keys):
            output.append(f"{path}.{key}: missing on left")
            if len(output) >= limit:
                return
        for key in sorted(left_keys & right_keys):
            _diff(left[key], right[key], path=f"{path}.{key}", output=output, limit=limit)
            if len(output) >= limit:
                return
        return
    if isinstance(left, list):
        if len(left) != len(right):
            output.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _diff(left_item, right_item, path=f"{path}[{index}]", output=output, limit=limit)
            if len(output) >= limit:
                return
        return
    if left != right:
        output.append(f"{path}: {left!r} != {right!r}")


__all__ = [
    "ReproducibilityComparison",
    "compare_run_semantics",
    "semantic_run_fingerprint",
    "semantic_run_trace",
]

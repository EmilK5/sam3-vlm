"""Low-level, dependency-free filesystem helpers for provenance storage.

All JSON writes are strict (no NaN/Infinity) and atomic when replacing a
snapshot. Event streams use append-only JSONL writes with optional fsync.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from provenance.base import CanonicalRecord, ContractError, to_jsonable


class StorageError(RuntimeError):
    """Raised when provenance data cannot be safely stored or recovered."""


class JsonLineCorruptionError(StorageError):
    """Raised when a JSONL file contains a malformed non-tail record."""


def strict_json_dumps(
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> str:
    """Serialize a canonical record or JSON-compatible value strictly."""

    payload = value.to_dict() if isinstance(value, CanonicalRecord) else to_jsonable(value)
    return json.dumps(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=False,
        allow_nan=False,
    )


def strict_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str, *, fsync: bool = True) -> None:
    """Atomically replace ``path`` with UTF-8 text.

    The temporary file is created in the destination directory so ``os.replace``
    remains atomic on a single filesystem.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        if fsync:
            _fsync_directory(path.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any, *, fsync: bool = True) -> None:
    atomic_write_text(path, strict_json_dumps(value), fsync=fsync)


def append_json_line(path: Path, value: Any, *, fsync: bool = True) -> None:
    """Append one compact JSON object to a JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (strict_json_dumps(value, indent=None) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StorageError(f"Failed to append JSON event to {path}")
            view = view[written:]
        if fsync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def iter_json_lines(
    path: Path,
    *,
    tolerate_truncated_tail: bool = False,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield ``(line_number, payload)`` from a JSONL file.

    Only a malformed final non-empty line may be ignored when
    ``tolerate_truncated_tail`` is enabled. Any earlier corruption is fatal.
    """

    path = Path(path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    non_empty_indices = [index for index, line in enumerate(lines) if line.strip()]
    final_non_empty = non_empty_indices[-1] if non_empty_indices else None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if tolerate_truncated_tail and index == final_non_empty:
                return
            raise JsonLineCorruptionError(
                f"Malformed JSONL record at {path}:{index + 1}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise JsonLineCorruptionError(
                f"JSONL record at {path}:{index + 1} is not an object"
            )
        yield index + 1, payload


def repair_truncated_jsonl_tail(path: Path) -> bool:
    """Remove one malformed final JSONL line, returning whether repair occurred."""

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    non_empty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not non_empty_indices:
        return False
    final_non_empty = non_empty_indices[-1]

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != final_non_empty:
                raise JsonLineCorruptionError(
                    f"Malformed non-tail JSONL record at {path}:{index + 1}"
                ) from exc
            repaired = b"".join(lines[:index])
            atomic_write_bytes(path, repaired)
            return True
        if not isinstance(payload, Mapping):
            raise JsonLineCorruptionError(
                f"JSONL record at {path}:{index + 1} is not an object"
            )
    return False


def atomic_write_bytes(path: Path, data: bytes, *, fsync: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        if fsync:
            _fsync_directory(path.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = strict_json_dumps(value, indent=None, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"Artifact path must remain inside the run directory: {value}")
    if not path.parts:
        raise ContractError("Artifact path must not be empty")
    return path


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "JsonLineCorruptionError",
    "StorageError",
    "append_json_line",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "iter_json_lines",
    "repair_truncated_jsonl_tail",
    "safe_relative_path",
    "sha256_file",
    "sha256_json",
    "strict_json_dumps",
    "strict_json_load",
]

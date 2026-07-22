"""Shared serialization and validation primitives for provenance records.

The contract layer is intentionally dependency-free.  Records may accept values
originating from NumPy, pathlib, enums, and datetime objects, but serialized
output is strict JSON with no NaN or Infinity values.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import math
import types
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = "1.0.0"

TRecord = TypeVar("TRecord", bound="CanonicalRecord")
_RECORD_REGISTRY: dict[str, type["CanonicalRecord"]] = {}


class ContractError(ValueError):
    """Raised when a provenance record violates its canonical contract."""


class SchemaVersionError(ContractError):
    """Raised when serialized data uses an unsupported schema version."""


class CanonicalRecord:
    """Base class for immutable, versioned, JSON-serializable records."""

    RECORD_TYPE: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        record_type = getattr(cls, "RECORD_TYPE", None)
        if record_type:
            previous = _RECORD_REGISTRY.get(record_type)
            if previous is not None and previous is not cls:
                raise RuntimeError(f"Duplicate provenance record type: {record_type}")
            _RECORD_REGISTRY[record_type] = cls

    def to_dict(self) -> dict[str, Any]:
        if not dataclasses.is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass")
        payload = {
            field.name: _to_jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": self.RECORD_TYPE,
            **payload,
        }

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls: type[TRecord], payload: Mapping[str, Any]) -> TRecord:
        record_type = payload.get("record_type")
        if record_type != cls.RECORD_TYPE:
            raise ContractError(
                f"Expected record_type={cls.RECORD_TYPE!r}, got {record_type!r}"
            )
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Unsupported schema version {version!r}; expected {SCHEMA_VERSION!r}"
            )

        allowed = {field.name for field in dataclasses.fields(cls)}
        supplied = set(payload) - {"schema_version", "record_type"}
        unknown = supplied - allowed
        if unknown:
            raise ContractError(
                f"Unknown fields for {cls.__name__}: {sorted(unknown)}"
            )

        type_hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if field.name not in payload:
                if field.default is not dataclasses.MISSING:
                    continue
                if field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
                    continue
                raise ContractError(
                    f"Missing required field {field.name!r} for {cls.__name__}"
                )
            annotation = type_hints.get(field.name, Any)
            kwargs[field.name] = _decode_value(payload[field.name], annotation)
        return cls(**kwargs)

    @classmethod
    def from_json(cls: type[TRecord], text: str) -> TRecord:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ContractError("A canonical record must deserialize from a JSON object")
        return cls.from_dict(payload)


def record_from_dict(payload: Mapping[str, Any]) -> CanonicalRecord:
    """Deserialize a canonical record using its embedded ``record_type`` tag."""

    record_type = payload.get("record_type")
    if not isinstance(record_type, str):
        raise ContractError("Serialized record is missing a string record_type")
    cls = _RECORD_REGISTRY.get(record_type)
    if cls is None:
        raise ContractError(f"Unknown provenance record type: {record_type!r}")
    return cls.from_dict(payload)


def registered_record_types() -> tuple[str, ...]:
    return tuple(sorted(_RECORD_REGISTRY))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")


def require_non_negative(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ContractError(f"{field_name} must be non-negative")


def require_probability(value: float, field_name: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ContractError(f"{field_name} must be a finite value in [0, 1]")


def require_probability_mapping(
    probabilities: Mapping[str, float],
    field_name: str,
    *,
    normalized: bool = False,
    tolerance: float = 1e-6,
) -> None:
    if not probabilities:
        raise ContractError(f"{field_name} must not be empty")
    for label, value in probabilities.items():
        require_non_empty(label, f"{field_name} label")
        require_probability(float(value), f"{field_name}[{label!r}]")
    if normalized:
        total = sum(float(value) for value in probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=tolerance, rel_tol=tolerance):
            raise ContractError(f"{field_name} must sum to 1; got {total}")


def require_box(box: tuple[float, float, float, float], field_name: str) -> None:
    if len(box) != 4:
        raise ContractError(f"{field_name} must contain exactly four coordinates")
    if not all(math.isfinite(float(value)) for value in box):
        raise ContractError(f"{field_name} coordinates must be finite")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 < x1 or y2 < y1:
        raise ContractError(f"{field_name} must satisfy x2 >= x1 and y2 >= y1")


def require_matrix_shape(
    matrix: tuple[tuple[float, ...], ...],
    rows: int,
    columns: int,
    field_name: str,
) -> None:
    if len(matrix) != rows:
        raise ContractError(f"{field_name} must have {rows} rows")
    for row_index, row in enumerate(matrix):
        if len(row) != columns:
            raise ContractError(
                f"{field_name}[{row_index}] must have {columns} columns"
            )
        for column_index, value in enumerate(row):
            if not math.isfinite(float(value)):
                raise ContractError(
                    f"{field_name}[{row_index}][{column_index}] must be finite"
                )


def to_jsonable(value: Any) -> Any:
    """Convert supported values to strict JSON-compatible Python objects."""

    return _to_jsonable(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, CanonicalRecord):
        return value.to_dict()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("Non-finite floating-point values are not valid JSON")
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        scalar = item_method()
        if scalar is not value:
            return _to_jsonable(scalar)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        converted = tolist_method()
        if converted is not value:
            return _to_jsonable(converted)

    raise TypeError(f"Unsupported value for canonical JSON serialization: {type(value)}")


def _decode_value(value: Any, annotation: Any) -> Any:
    if annotation is Any:
        if isinstance(value, dict) and "record_type" in value:
            return record_from_dict(value)
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors: list[Exception] = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _decode_value(value, option)
            except (TypeError, ValueError, ContractError) as exc:
                errors.append(exc)
        raise ContractError(
            f"Value {value!r} does not match any allowed union member: {errors}"
        )

    if origin is Literal:
        if value not in args:
            raise ContractError(f"Expected one of {args!r}, got {value!r}")
        return value

    if origin in (list, collections.abc.Sequence):
        if not isinstance(value, (list, tuple)):
            raise ContractError(f"Expected sequence, got {type(value).__name__}")
        item_type = args[0] if args else Any
        return [_decode_value(item, item_type) for item in value]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ContractError(f"Expected tuple-compatible sequence, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_value(item, args[0]) for item in value)
        if len(value) != len(args):
            raise ContractError(
                f"Tuple length mismatch: expected {len(args)}, got {len(value)}"
            )
        return tuple(_decode_value(item, item_type) for item, item_type in zip(value, args))

    if origin in (dict, Mapping, collections.abc.Mapping):
        if not isinstance(value, Mapping):
            raise ContractError(f"Expected mapping, got {type(value).__name__}")
        key_type, value_type = args if args else (Any, Any)
        return {
            _decode_value(key, key_type): _decode_value(item, value_type)
            for key, item in value.items()
        }

    if annotation is CanonicalRecord:
        if not isinstance(value, Mapping):
            raise ContractError("Expected object for CanonicalRecord")
        return record_from_dict(value)

    if isinstance(annotation, type) and issubclass(annotation, CanonicalRecord):
        if not isinstance(value, Mapping):
            raise ContractError(f"Expected object for {annotation.__name__}")
        return annotation.from_dict(value)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)

    if annotation in (str, int, float, bool):
        if annotation is bool and not isinstance(value, bool):
            raise ContractError(f"Expected bool, got {type(value).__name__}")
        return annotation(value)

    return value

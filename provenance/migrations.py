"""Forward migration helpers for pre-release canonical provenance payloads.

The public 1.0.0 schema is the first stable contract.  During development,
0.9.0 payloads used the same record shapes but lacked some optional/defaulted
fields.  This module upgrades those payloads recursively before normal strict
contract decoding.  Unknown versions remain hard errors.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from provenance.base import (
    SCHEMA_VERSION,
    CanonicalRecord,
    ContractError,
    SchemaVersionError,
    record_from_dict,
)

_PRE_RELEASE_VERSION = "0.9.0"


def migrate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a canonical record tree to the current schema version."""

    migrated = _migrate_value(deepcopy(dict(payload)))
    if not isinstance(migrated, dict):
        raise ContractError("A migrated canonical payload must remain an object")
    return migrated


def migrated_record_from_dict(payload: Mapping[str, Any]) -> CanonicalRecord:
    return record_from_dict(migrate_payload(payload))


def _migrate_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_migrate_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    result = {str(key): _migrate_value(item) for key, item in value.items()}
    if "record_type" not in result:
        return result

    version = result.get("schema_version")
    if version == SCHEMA_VERSION:
        return result
    if version != _PRE_RELEASE_VERSION:
        raise SchemaVersionError(
            f"No provenance migration path from {version!r} to {SCHEMA_VERSION!r}"
        )

    record_type = result.get("record_type")
    if not isinstance(record_type, str) or not record_type:
        raise ContractError("Migrated record is missing record_type")

    # All 0.9.0 additions were optional/defaulted.  Materialize important
    # collection fields so migrated JSON is explicit and easier to inspect.
    if record_type == "run":
        result.setdefault("events", [])
        result.setdefault("warnings", [])
        result.setdefault("metadata", {})
    elif record_type == "pass":
        result.setdefault("tiling_decision", None)
        result.setdefault("warnings", [])
        result.setdefault("metadata", {})
    elif record_type == "graph_node_snapshot":
        result.setdefault("mask_artifact", None)
        result.setdefault("legacy_scores", {})
        result.setdefault("metadata", {})

    result["schema_version"] = SCHEMA_VERSION
    return result


__all__ = ["migrate_payload", "migrated_record_from_dict"]

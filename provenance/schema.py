"""Enumerations shared by canonical provenance records."""

from __future__ import annotations

from enum import Enum

from provenance.base import SCHEMA_VERSION


class StringEnum(str, Enum):
    """Enum whose values serialize naturally as JSON strings."""


class RunStatus(StringEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ActionKind(StringEnum):
    QUERY = "query"
    VERIFY = "verify"
    DISCOVER = "discover"
    EXPLORE = "explore"
    STOP = "stop"


class ActionFamily(StringEnum):
    TARGET = "target"
    NEGATIVE = "negative"
    EXPLORE = "explore"
    SYSTEM = "system"


class TilingMode(StringEnum):
    OFF = "off"
    ALWAYS = "always"
    DENSITY_ADAPTIVE = "density_adaptive"
    AGENT_CONTROLLED = "agent_controlled"


class DedupDecision(StringEnum):
    KEEP_DISTINCT = "keep_distinct"
    SUPPRESS_NEW = "suppress_new"
    SUPPRESS_EXISTING = "suppress_existing"
    MERGE = "merge"
    NOT_APPLICABLE = "not_applicable"


class RegistrationOutcome(StringEnum):
    CREATED_NODE = "created_node"
    UPDATED_NODE = "updated_node"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class NodeStatus(StringEnum):
    UNRESOLVED = "unresolved"
    STOPPED = "stopped"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REJECTED = "rejected"


class StopReason(StringEnum):
    CONFIDENCE = "confidence"
    BUDGET = "budget"
    NO_VALID_ACTION = "no_valid_action"
    USER = "user"
    POLICY = "policy"
    ERROR = "error"
    NOT_STOPPED = "not_stopped"


class ErrorSeverity(StringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ArtifactKind(StringEnum):
    IMAGE = "image"
    MASK = "mask"
    CROP = "crop"
    OVERLAY = "overlay"
    LOG = "log"
    CONFIG = "config"
    GRAPH = "graph"
    OTHER = "other"


class CoordinateSpace(StringEnum):
    IMAGE_GLOBAL = "image_global"
    ROI_LOCAL = "roi_local"
    TILE_LOCAL = "tile_local"
    NORMALIZED = "normalized"


class ObservationLevel(StringEnum):
    NOT_FOUND = "not_found"
    WEAK_MATCH = "weak_match"
    STRONG_MATCH = "strong_match"
    CUSTOM = "custom"


class EventKind(StringEnum):
    RUN = "run"
    PASS = "pass"
    ACTION = "action"
    QWEN_CALL = "qwen_call"
    SAM3_CALL = "sam3_call"
    DETECTION = "detection"
    TILE = "tile"
    DEDUP = "dedup"
    REGISTRATION = "registration"
    GRAPH = "graph"
    OBSERVATION = "observation"
    KERNEL = "kernel"
    INFORMATION_GAIN = "information_gain"
    BELIEF_UPDATE = "belief_update"
    STOPPING = "stopping"
    COST = "cost"
    EVALUATION = "evaluation"
    ARTIFACT = "artifact"
    ERROR = "error"


__all__ = [
    "SCHEMA_VERSION",
    "ActionFamily",
    "ActionKind",
    "ArtifactKind",
    "CoordinateSpace",
    "DedupDecision",
    "ErrorSeverity",
    "EventKind",
    "NodeStatus",
    "ObservationLevel",
    "RegistrationOutcome",
    "RunStatus",
    "StopReason",
    "StringEnum",
    "TilingMode",
]

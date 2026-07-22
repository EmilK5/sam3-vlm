"""Canonical data contracts for every entity used or produced by the pipeline.

Phase 1 defines records only.  No current runner, graph, model call, or policy
imports this module yet; later phases can adopt the contracts incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from provenance.base import (
    CanonicalRecord,
    ContractError,
    require_box,
    require_matrix_shape,
    require_non_empty,
    require_non_negative,
    require_probability,
    require_probability_mapping,
    utc_now_iso,
)
from provenance.ids import EntityKind, validate_entity_id
from provenance.schema import (
    ActionFamily,
    ActionKind,
    ArtifactKind,
    CoordinateSpace,
    DedupDecision,
    ErrorSeverity,
    EventKind,
    NodeStatus,
    ObservationLevel,
    RegistrationOutcome,
    RunStatus,
    StopReason,
    TilingMode,
)

BoxXYXY = tuple[float, float, float, float]
ProbabilityMap = Mapping[str, float]
JsonMap = Mapping[str, Any]


@dataclass(frozen=True)
class RepositoryStateRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "repository_state"

    commit: str
    branch: str
    dirty: bool
    remote_url: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.commit, "commit")
        require_non_empty(self.branch, "branch")


@dataclass(frozen=True)
class EnvironmentRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "environment"

    python_version: str
    platform: str
    hostname: str
    packages: Mapping[str, str] = field(default_factory=dict)
    hardware: JsonMap = field(default_factory=dict)
    environment_variables: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.python_version, "python_version")
        require_non_empty(self.platform, "platform")
        require_non_empty(self.hostname, "hostname")


@dataclass(frozen=True)
class ModelIdentityRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "model_identity"

    model_id: str
    role: str
    name: str
    provider: str
    version: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    endpoint: str | None = None
    configuration: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.model_id, "model_id")
        require_non_empty(self.role, "role")
        require_non_empty(self.name, "name")
        require_non_empty(self.provider, "provider")


@dataclass(frozen=True)
class ArtifactRef(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "artifact_ref"

    artifact_id: str
    kind: ArtifactKind
    relative_path: str
    media_type: str
    sha256: str | None = None
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.artifact_id, expected_kind=EntityKind.ARTIFACT)
        require_non_empty(self.relative_path, "relative_path")
        require_non_empty(self.media_type, "media_type")
        if self.size_bytes is not None:
            require_non_negative(self.size_bytes, "size_bytes")
        if self.width is not None:
            require_non_negative(self.width, "width")
        if self.height is not None:
            require_non_negative(self.height, "height")


@dataclass(frozen=True)
class DatasetSampleRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "dataset_sample"

    image_id: str
    dataset_name: str
    dataset_version: str
    split: str
    image_artifact: ArtifactRef
    image_sha256: str
    width: int
    height: int
    target_concept: str
    target_class: str
    sample_index: int | None = None
    sequence_id: str | None = None
    confounder_classes: tuple[str, ...] = ()
    ground_truth_count: int | None = None
    ground_truth_boxes: tuple[BoxXYXY, ...] = ()
    ground_truth_masks: tuple[ArtifactRef, ...] = ()
    exemplar_boxes: tuple[BoxXYXY, ...] = ()
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.image_id, expected_kind=EntityKind.IMAGE)
        require_non_empty(self.dataset_name, "dataset_name")
        require_non_empty(self.dataset_version, "dataset_version")
        require_non_empty(self.split, "split")
        require_non_empty(self.image_sha256, "image_sha256")
        require_non_empty(self.target_concept, "target_concept")
        require_non_empty(self.target_class, "target_class")
        require_non_negative(self.width, "width")
        require_non_negative(self.height, "height")
        if self.sample_index is not None:
            require_non_negative(self.sample_index, "sample_index")
        if self.ground_truth_count is not None:
            require_non_negative(self.ground_truth_count, "ground_truth_count")
        for index, box in enumerate(self.ground_truth_boxes):
            require_box(box, f"ground_truth_boxes[{index}]")
        for index, box in enumerate(self.exemplar_boxes):
            require_box(box, f"exemplar_boxes[{index}]")


@dataclass(frozen=True)
class SensingActionRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "sensing_action"

    action_id: str
    action_kind: ActionKind
    family: ActionFamily
    prompt: str
    region: BoxXYXY
    threshold: float
    semantic_key: str
    target_node_id: str | None = None
    positive_exemplar_node_ids: tuple[str, ...] = ()
    negative_exemplar_node_ids: tuple[str, ...] = ()
    tiling_mode: TilingMode = TilingMode.OFF
    tile_scale: float | None = None
    beta_by_class: ProbabilityMap = field(default_factory=dict)
    rationale: str | None = None
    expected_visual_distinction: str | None = None
    generator: str = "system"
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.action_id, expected_kind=EntityKind.ACTION)
        require_non_empty(self.prompt, "prompt")
        require_non_empty(self.semantic_key, "semantic_key")
        require_non_empty(self.generator, "generator")
        require_box(self.region, "region")
        require_probability(self.threshold, "threshold")
        if self.tile_scale is not None and self.tile_scale <= 0:
            raise ContractError("tile_scale must be positive")
        if self.beta_by_class:
            require_probability_mapping(self.beta_by_class, "beta_by_class")


@dataclass(frozen=True)
class CandidateActionSetRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "candidate_action_set"

    candidate_set_id: str
    pass_id: str
    target_node_id: str | None
    actions: tuple[SensingActionRecord, ...]
    source_qwen_call_id: str | None = None
    generation_policy: str = "static"
    validation_errors: tuple[str, ...] = ()
    rejected_action_payloads: tuple[JsonMap, ...] = ()
    fallback_used: bool = False
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.candidate_set_id, expected_kind=EntityKind.CANDIDATE)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_non_empty(self.generation_policy, "generation_policy")
        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ContractError("Candidate action IDs must be unique")


@dataclass(frozen=True)
class QwenCallRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "qwen_call"

    qwen_call_id: str
    pass_id: str
    model_id: str
    system_prompt: str
    user_prompt: str
    image_artifact_ids: tuple[str, ...]
    request_payload: JsonMap
    raw_response: JsonMap | str | None
    parsed_response: JsonMap | None
    reasoning_summary: str | None
    validation_errors: tuple[str, ...]
    retries: int
    fallback_used: bool
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    sampling_parameters: JsonMap = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_entity_id(self.qwen_call_id, expected_kind=EntityKind.QWEN_CALL)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_non_empty(self.model_id, "model_id")
        require_non_negative(self.retries, "retries")
        require_non_negative(self.latency_seconds, "latency_seconds")
        if self.input_tokens is not None:
            require_non_negative(self.input_tokens, "input_tokens")
        if self.output_tokens is not None:
            require_non_negative(self.output_tokens, "output_tokens")


@dataclass(frozen=True)
class Sam3CallRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "sam3_call"

    sam3_call_id: str
    pass_id: str
    model_id: str
    prompt: str
    region: BoxXYXY
    threshold: float
    positive_exemplar_node_ids: tuple[str, ...]
    negative_exemplar_node_ids: tuple[str, ...]
    preprocessing: JsonMap
    decoder_settings: JsonMap
    raw_output_artifact_ids: tuple[str, ...]
    raw_detection_ids: tuple[str, ...]
    tile_id: str | None
    latency_seconds: float
    coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_GLOBAL
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_entity_id(self.sam3_call_id, expected_kind=EntityKind.SAM3_CALL)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_non_empty(self.model_id, "model_id")
        require_non_empty(self.prompt, "prompt")
        require_box(self.region, "region")
        require_probability(self.threshold, "threshold")
        require_non_negative(self.latency_seconds, "latency_seconds")


@dataclass(frozen=True)
class TilingDecisionRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "tiling_decision"

    tiling_decision_id: str
    pass_id: str
    mode: TilingMode
    triggered: bool
    trigger_source: str
    tiling_rule: str
    reason: str
    source_detection_ids: tuple[str, ...]
    source_roi: BoxXYXY
    density_metrics: JsonMap
    reliability_metrics: JsonMap
    tile_size: int | None
    overlap_pixels: int | None
    stride_pixels: int | None
    base_tile_count: int
    selected_tile_count: int
    thresholds: JsonMap
    selected_roi: BoxXYXY | None = None
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(
            self.tiling_decision_id, expected_kind=EntityKind.TILING_DECISION
        )
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_non_empty(self.trigger_source, "trigger_source")
        require_non_empty(self.tiling_rule, "tiling_rule")
        require_non_empty(self.reason, "reason")
        require_box(self.source_roi, "source_roi")
        if self.selected_roi is not None:
            require_box(self.selected_roi, "selected_roi")
        for name, value in (
            ("tile_size", self.tile_size),
            ("overlap_pixels", self.overlap_pixels),
            ("stride_pixels", self.stride_pixels),
        ):
            if value is not None:
                require_non_negative(value, name)
        require_non_negative(self.base_tile_count, "base_tile_count")
        require_non_negative(self.selected_tile_count, "selected_tile_count")
        if self.selected_tile_count > self.base_tile_count:
            raise ContractError("selected_tile_count cannot exceed base_tile_count")


@dataclass(frozen=True)
class TileRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "tile"

    tile_id: str
    pass_id: str
    tile_index: int
    grid_row: int
    grid_column: int
    local_box: BoxXYXY
    global_box: BoxXYXY
    overlap_fraction: float
    source_roi: BoxXYXY
    sam3_call_ids: tuple[str, ...]
    raw_detection_ids: tuple[str, ...]
    transform: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.tile_id, expected_kind=EntityKind.TILE)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_non_negative(self.tile_index, "tile_index")
        require_non_negative(self.grid_row, "grid_row")
        require_non_negative(self.grid_column, "grid_column")
        require_box(self.local_box, "local_box")
        require_box(self.global_box, "global_box")
        require_box(self.source_roi, "source_roi")
        require_probability(self.overlap_fraction, "overlap_fraction")


@dataclass(frozen=True)
class RawDetectionRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "raw_detection"

    raw_detection_id: str
    sam3_call_id: str
    pass_id: str
    box_local: BoxXYXY
    box_global: BoxXYXY
    score: float
    prompt: str
    coordinate_space: CoordinateSpace
    tile_id: str | None = None
    mask_artifact: ArtifactRef | None = None
    mask_rle: JsonMap | None = None
    logits_artifact: ArtifactRef | None = None
    geometry: JsonMap = field(default_factory=dict)
    raw_index: int = 0

    def __post_init__(self) -> None:
        validate_entity_id(self.raw_detection_id, expected_kind=EntityKind.RAW_DETECTION)
        validate_entity_id(self.sam3_call_id, expected_kind=EntityKind.SAM3_CALL)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_box(self.box_local, "box_local")
        require_box(self.box_global, "box_global")
        require_probability(self.score, "score")
        require_non_empty(self.prompt, "prompt")
        require_non_negative(self.raw_index, "raw_index")


@dataclass(frozen=True)
class DedupComparisonRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "dedup_comparison"

    dedup_decision_id: str
    pass_id: str
    new_detection_id: str
    existing_detection_id: str | None
    existing_node_id: str | None
    stage: str
    metrics: Mapping[str, float]
    thresholds: Mapping[str, float]
    decision: DedupDecision
    reason: str
    selected_survivor_id: str | None = None
    comparison_rank: int | None = None

    def __post_init__(self) -> None:
        validate_entity_id(
            self.dedup_decision_id, expected_kind=EntityKind.DEDUP_DECISION
        )
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(
            self.new_detection_id, expected_kind=EntityKind.RAW_DETECTION
        )
        require_non_empty(self.stage, "stage")
        require_non_empty(self.reason, "reason")
        for name, value in self.metrics.items():
            require_non_empty(name, "metric name")
            if float(value) < 0:
                raise ContractError(f"metrics[{name!r}] must be non-negative")
        for name, value in self.thresholds.items():
            require_non_empty(name, "threshold name")
            if float(value) < 0:
                raise ContractError(f"thresholds[{name!r}] must be non-negative")
        if self.comparison_rank is not None:
            require_non_negative(self.comparison_rank, "comparison_rank")


@dataclass(frozen=True)
class RegistrationDecisionRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "registration_decision"

    registration_id: str
    pass_id: str
    raw_detection_id: str
    outcome: RegistrationOutcome
    graph_node_id: str | None
    candidate_node_ids: tuple[str, ...]
    selected_dedup_decision_id: str | None
    reason: str
    node_before: "GraphNodeSnapshotRecord | None" = None
    node_after: "GraphNodeSnapshotRecord | None" = None

    def __post_init__(self) -> None:
        validate_entity_id(self.registration_id, expected_kind=EntityKind.REGISTRATION)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(
            self.raw_detection_id, expected_kind=EntityKind.RAW_DETECTION
        )
        require_non_empty(self.reason, "reason")
        if self.outcome in (
            RegistrationOutcome.CREATED_NODE,
            RegistrationOutcome.UPDATED_NODE,
        ) and self.graph_node_id is None:
            raise ContractError("A created or updated node requires graph_node_id")


@dataclass(frozen=True)
class GraphNodeSnapshotRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "graph_node_snapshot"

    graph_node_id: str
    pass_id: str
    box: BoxXYXY
    status: NodeStatus
    source_detection_ids: tuple[str, ...]
    found_in_passes: tuple[int, ...]
    prior: ProbabilityMap
    posterior: ProbabilityMap
    temporary_map_class: str | None
    final_declaration: str | None
    query_count: int
    exemplar_eligible_positive: bool
    exemplar_eligible_negative: bool
    mask_artifact: ArtifactRef | None = None
    legacy_scores: Mapping[str, float] = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.graph_node_id, expected_kind=EntityKind.GRAPH_NODE)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_box(self.box, "box")
        require_probability_mapping(self.prior, "prior", normalized=True)
        require_probability_mapping(self.posterior, "posterior", normalized=True)
        if set(self.prior) != set(self.posterior):
            raise ContractError("prior and posterior must use the same class labels")
        require_non_negative(self.query_count, "query_count")
        for pass_number in self.found_in_passes:
            require_non_negative(pass_number, "found_in_passes entry")


@dataclass(frozen=True)
class EncodedObservationRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "encoded_observation"

    observation_id: str
    pass_id: str
    action_id: str
    target_node_id: str
    level: ObservationLevel
    label: str
    overlap_metrics: Mapping[str, float]
    score_statistics: Mapping[str, float]
    bin_thresholds: Mapping[str, float]
    matched_detection_ids: tuple[str, ...]
    encoder_version: str
    raw_features: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.observation_id, expected_kind=EntityKind.OBSERVATION)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(self.action_id, expected_kind=EntityKind.ACTION)
        validate_entity_id(self.target_node_id, expected_kind=EntityKind.GRAPH_NODE)
        require_non_empty(self.label, "label")
        require_non_empty(self.encoder_version, "encoder_version")


@dataclass(frozen=True)
class SurrogateKernelRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "surrogate_kernel"

    kernel_id: str
    pass_id: str
    action_id: str
    class_names: tuple[str, ...]
    observation_labels: tuple[str, ...]
    beta_by_class: ProbabilityMap
    present_profile: tuple[float, ...]
    absent_profile: tuple[float, ...]
    kernel_matrix: tuple[tuple[float, ...], ...]
    epsilon: float
    profile_source: str
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.kernel_id, expected_kind=EntityKind.KERNEL)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(self.action_id, expected_kind=EntityKind.ACTION)
        if not self.class_names:
            raise ContractError("class_names must not be empty")
        if not self.observation_labels:
            raise ContractError("observation_labels must not be empty")
        if len(set(self.class_names)) != len(self.class_names):
            raise ContractError("class_names must be unique")
        if len(set(self.observation_labels)) != len(self.observation_labels):
            raise ContractError("observation_labels must be unique")
        require_probability_mapping(self.beta_by_class, "beta_by_class")
        if set(self.beta_by_class) != set(self.class_names):
            raise ContractError("beta_by_class keys must match class_names")
        if len(self.present_profile) != len(self.observation_labels):
            raise ContractError("present_profile length must match observation_labels")
        if len(self.absent_profile) != len(self.observation_labels):
            raise ContractError("absent_profile length must match observation_labels")
        for index, value in enumerate(self.present_profile):
            require_probability(value, f"present_profile[{index}]")
        for index, value in enumerate(self.absent_profile):
            require_probability(value, f"absent_profile[{index}]")
        if abs(sum(self.present_profile) - 1.0) > 1e-6:
            raise ContractError("present_profile must sum to 1")
        if abs(sum(self.absent_profile) - 1.0) > 1e-6:
            raise ContractError("absent_profile must sum to 1")
        require_matrix_shape(
            self.kernel_matrix,
            len(self.class_names),
            len(self.observation_labels),
            "kernel_matrix",
        )
        for row_index, row in enumerate(self.kernel_matrix):
            for column_index, value in enumerate(row):
                require_probability(
                    value, f"kernel_matrix[{row_index}][{column_index}]"
                )
            if abs(sum(row) - 1.0) > 1e-6:
                raise ContractError(f"kernel_matrix row {row_index} must sum to 1")
        if not 0.0 < self.epsilon < 0.5:
            raise ContractError("epsilon must lie in (0, 0.5)")
        require_non_empty(self.profile_source, "profile_source")


@dataclass(frozen=True)
class InformationGainRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "information_gain"

    information_gain_id: str
    pass_id: str
    action_id: str
    kernel_id: str
    posterior: ProbabilityMap
    predictive_observation: ProbabilityMap
    expected_information_gain: float
    expected_cost: float | None
    cost_adjusted_score: float | None
    selected: bool
    rank: int
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        validate_entity_id(
            self.information_gain_id, expected_kind=EntityKind.INFORMATION_GAIN
        )
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(self.action_id, expected_kind=EntityKind.ACTION)
        validate_entity_id(self.kernel_id, expected_kind=EntityKind.KERNEL)
        require_probability_mapping(self.posterior, "posterior", normalized=True)
        require_probability_mapping(
            self.predictive_observation,
            "predictive_observation",
            normalized=True,
        )
        require_non_negative(
            self.expected_information_gain, "expected_information_gain"
        )
        if self.expected_cost is not None:
            require_non_negative(self.expected_cost, "expected_cost")
        require_non_negative(self.rank, "rank")


@dataclass(frozen=True)
class BeliefUpdateRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "belief_update"

    belief_update_id: str
    pass_id: str
    graph_node_id: str
    action_id: str
    observation_id: str
    kernel_id: str
    posterior_before: ProbabilityMap
    likelihood_by_class: ProbabilityMap
    predictive_observation_probability: float
    posterior_after: ProbabilityMap
    entropy_before: float
    entropy_after: float
    expected_information_gain: float
    realized_kl: float
    realized_entropy_change: float
    numerical_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_entity_id(
            self.belief_update_id, expected_kind=EntityKind.BELIEF_UPDATE
        )
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(self.graph_node_id, expected_kind=EntityKind.GRAPH_NODE)
        validate_entity_id(self.action_id, expected_kind=EntityKind.ACTION)
        validate_entity_id(self.observation_id, expected_kind=EntityKind.OBSERVATION)
        validate_entity_id(self.kernel_id, expected_kind=EntityKind.KERNEL)
        require_probability_mapping(
            self.posterior_before, "posterior_before", normalized=True
        )
        require_probability_mapping(
            self.posterior_after, "posterior_after", normalized=True
        )
        if set(self.posterior_before) != set(self.posterior_after):
            raise ContractError("Posterior class labels changed during update")
        require_probability_mapping(self.likelihood_by_class, "likelihood_by_class")
        require_probability(
            self.predictive_observation_probability,
            "predictive_observation_probability",
        )
        require_non_negative(self.entropy_before, "entropy_before")
        require_non_negative(self.entropy_after, "entropy_after")
        require_non_negative(
            self.expected_information_gain, "expected_information_gain"
        )
        require_non_negative(self.realized_kl, "realized_kl")


@dataclass(frozen=True)
class StoppingDecisionRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "stopping_decision"

    stopping_id: str
    pass_id: str
    graph_node_id: str | None
    posterior: ProbabilityMap
    threshold: float
    should_stop: bool
    declaration: str | None
    reason: StopReason
    query_count: int
    budget_remaining: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.stopping_id, expected_kind=EntityKind.STOPPING)
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        require_probability_mapping(self.posterior, "posterior", normalized=True)
        require_probability(self.threshold, "threshold")
        require_non_negative(self.query_count, "query_count")
        if self.should_stop and self.reason is StopReason.NOT_STOPPED:
            raise ContractError("A stopping decision cannot use NOT_STOPPED")
        if not self.should_stop and self.reason is not StopReason.NOT_STOPPED:
            raise ContractError("A continuing decision must use NOT_STOPPED")


@dataclass(frozen=True)
class CostSnapshotRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "cost_snapshot"

    pass_id: str
    sam3_calls: int
    qwen_calls: int
    tile_calls: int
    verify_calls: int
    orchestration_calls: int
    input_tokens: int
    output_tokens: int
    runtime_seconds: float
    normalized_cost: float
    monetary_cost: float | None = None
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        for name in (
            "sam3_calls",
            "qwen_calls",
            "tile_calls",
            "verify_calls",
            "orchestration_calls",
            "input_tokens",
            "output_tokens",
        ):
            require_non_negative(getattr(self, name), name)
        require_non_negative(self.runtime_seconds, "runtime_seconds")
        require_non_negative(self.normalized_cost, "normalized_cost")
        if self.monetary_cost is not None:
            require_non_negative(self.monetary_cost, "monetary_cost")


@dataclass(frozen=True)
class EvaluationResultRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "evaluation_result"

    evaluation_id: str
    run_id: str
    image_id: str
    evaluator_name: str
    evaluator_version: str
    metrics: Mapping[str, float | int | None]
    counts: Mapping[str, float | int]
    matched_pairs: tuple[tuple[str, str], ...] = ()
    unmatched_prediction_ids: tuple[str, ...] = ()
    unmatched_ground_truth_ids: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.evaluation_id, expected_kind=EntityKind.EVALUATION)
        validate_entity_id(self.run_id, expected_kind=EntityKind.RUN)
        validate_entity_id(self.image_id, expected_kind=EntityKind.IMAGE)
        require_non_empty(self.evaluator_name, "evaluator_name")
        require_non_empty(self.evaluator_version, "evaluator_version")


@dataclass(frozen=True)
class ErrorEventRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "error_event"

    error_id: str
    run_id: str
    severity: ErrorSeverity
    component: str
    message: str
    exception_type: str | None = None
    traceback: str | None = None
    pass_id: str | None = None
    related_entity_ids: tuple[str, ...] = ()
    recoverable: bool = True
    fallback_action: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_entity_id(self.error_id, expected_kind=EntityKind.ERROR)
        validate_entity_id(self.run_id, expected_kind=EntityKind.RUN)
        require_non_empty(self.component, "component")
        require_non_empty(self.message, "message")


@dataclass(frozen=True)
class PassRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "pass"

    pass_id: str
    run_id: str
    pass_index: int
    started_at: str
    completed_at: str | None
    target_node_id: str | None
    graph_before: tuple[GraphNodeSnapshotRecord, ...]
    candidate_action_set: CandidateActionSetRecord | None
    information_gain_records: tuple[InformationGainRecord, ...]
    selected_action_id: str | None
    qwen_calls: tuple[QwenCallRecord, ...]
    sam3_calls: tuple[Sam3CallRecord, ...]
    tiles: tuple[TileRecord, ...]
    raw_detections: tuple[RawDetectionRecord, ...]
    dedup_comparisons: tuple[DedupComparisonRecord, ...]
    registrations: tuple[RegistrationDecisionRecord, ...]
    observation: EncodedObservationRecord | None
    kernel: SurrogateKernelRecord | None
    belief_update: BeliefUpdateRecord | None
    stopping_decision: StoppingDecisionRecord | None
    graph_after: tuple[GraphNodeSnapshotRecord, ...]
    cost_after: CostSnapshotRecord
    continuation_reason: str | None
    tiling_decision: TilingDecisionRecord | None = None
    warnings: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.pass_id, expected_kind=EntityKind.PASS)
        validate_entity_id(self.run_id, expected_kind=EntityKind.RUN)
        require_non_negative(self.pass_index, "pass_index")
        require_non_empty(self.started_at, "started_at")
        if self.completed_at is not None:
            require_non_empty(self.completed_at, "completed_at")
        if self.selected_action_id is not None:
            candidate_ids = (
                {action.action_id for action in self.candidate_action_set.actions}
                if self.candidate_action_set is not None
                else set()
            )
            if self.selected_action_id not in candidate_ids:
                raise ContractError(
                    "selected_action_id must belong to candidate_action_set"
                )


@dataclass(frozen=True)
class EventEnvelope(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "event"

    event_id: str
    run_id: str
    sequence_number: int
    event_kind: EventKind
    payload: CanonicalRecord
    pass_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_entity_id(self.event_id, expected_kind=EntityKind.EVENT)
        validate_entity_id(self.run_id, expected_kind=EntityKind.RUN)
        require_non_negative(self.sequence_number, "sequence_number")


@dataclass(frozen=True)
class RunRecord(CanonicalRecord):
    RECORD_TYPE: ClassVar[str] = "run"

    run_id: str
    status: RunStatus
    created_at: str
    started_at: str | None
    completed_at: str | None
    repository: RepositoryStateRecord
    environment: EnvironmentRecord
    resolved_config: JsonMap
    config_sha256: str
    random_seeds: Mapping[str, int]
    dataset_sample: DatasetSampleRecord
    models: tuple[ModelIdentityRecord, ...]
    id_counters: Mapping[str, int]
    passes: tuple[PassRecord, ...]
    final_graph: tuple[GraphNodeSnapshotRecord, ...]
    final_predictions: JsonMap
    evaluations: tuple[EvaluationResultRecord, ...]
    artifacts: tuple[ArtifactRef, ...]
    errors: tuple[ErrorEventRecord, ...]
    events: tuple[EventEnvelope, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_entity_id(self.run_id, expected_kind=EntityKind.RUN)
        require_non_empty(self.created_at, "created_at")
        require_non_empty(self.config_sha256, "config_sha256")
        for seed_name, seed in self.random_seeds.items():
            require_non_empty(seed_name, "random seed name")
            if not isinstance(seed, int):
                raise ContractError(f"Random seed {seed_name!r} must be an integer")
        pass_ids = [pass_record.pass_id for pass_record in self.passes]
        if len(pass_ids) != len(set(pass_ids)):
            raise ContractError("Pass IDs must be unique within a run")
        if any(pass_record.run_id != self.run_id for pass_record in self.passes):
            raise ContractError("Every pass must reference the containing run")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ContractError("Event IDs must be unique within a run")
        if any(event.run_id != self.run_id for event in self.events):
            raise ContractError("Every event must reference the containing run")
        if self.events:
            sequences = [event.sequence_number for event in self.events]
            if sequences != list(range(1, len(sequences) + 1)):
                raise ContractError("Run events must have contiguous sequence numbers")
        if self.status is RunStatus.SUCCEEDED and self.completed_at is None:
            raise ContractError("A succeeded run must have completed_at")


__all__ = [
    "ArtifactRef",
    "BeliefUpdateRecord",
    "BoxXYXY",
    "CandidateActionSetRecord",
    "CostSnapshotRecord",
    "DatasetSampleRecord",
    "DedupComparisonRecord",
    "EncodedObservationRecord",
    "EnvironmentRecord",
    "ErrorEventRecord",
    "EvaluationResultRecord",
    "EventEnvelope",
    "GraphNodeSnapshotRecord",
    "InformationGainRecord",
    "ModelIdentityRecord",
    "PassRecord",
    "QwenCallRecord",
    "RawDetectionRecord",
    "RegistrationDecisionRecord",
    "RepositoryStateRecord",
    "RunRecord",
    "Sam3CallRecord",
    "SensingActionRecord",
    "StoppingDecisionRecord",
    "SurrogateKernelRecord",
    "TileRecord",
    "TilingDecisionRecord",
]

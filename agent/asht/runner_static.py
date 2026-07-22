"""Static-action end-to-end ASHT controller.

This is the Phase 6 runner.  It intentionally uses a fixed action bank so the
Bayesian controller, graph integration, and provenance can be validated before
Qwen becomes the action generator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from agent.asht.action_bank import MaterializedStaticAction, StaticActionBank
from agent.asht.counting import CountEstimate, estimate_count
from agent.asht.information import (
    entropy,
    expected_information_gain,
    predictive_distribution,
    rank_actions,
)
from agent.asht.observations import ObservationEncoderConfig, encode_target_observation
from agent.asht.tiling import AdaptiveTilingConfig, execute_adaptive_tiling
from agent.asht.update import UpdateIds, perform_update
from graph import OrchardGraph, OrchardNode
from pipeline_stages import (
    DetectionBatch,
    Sam3QuerySpec,
    StageContext,
    execute_sam3_request,
    register_detection_batch,
    suppress_detection_batch,
)
from provenance.base import CanonicalRecord, utc_now_iso
from provenance.contracts import (
    CandidateActionSetRecord,
    CostSnapshotRecord,
    DedupComparisonRecord,
    EncodedObservationRecord,
    InformationGainRecord,
    PassRecord,
    RawDetectionRecord,
    RegistrationDecisionRecord,
    Sam3CallRecord,
    SensingActionRecord,
    StoppingDecisionRecord,
    SurrogateKernelRecord,
    TileRecord,
    TilingDecisionRecord,
)
from provenance.ids import EntityKind, IdFactory, create_run_id
from provenance.run_store import ReportingLevel
from provenance.schema import CoordinateSpace, StopReason, TilingMode


@dataclass(frozen=True)
class StaticAshtConfig:
    class_names: tuple[str, ...]
    target_class: str
    prior: Mapping[str, float]
    action_bank: StaticActionBank
    stopping_threshold: float = 0.90
    max_total_queries: int = 30
    max_queries_per_node: int = 4
    use_cost_adjusted_eig: bool = False
    observation_encoder: ObservationEncoderConfig = field(
        default_factory=ObservationEncoderConfig
    )
    suppression_confidence: float = 0.0
    suppression_kwargs: Mapping[str, Any] = field(default_factory=dict)
    bootstrap_query: Sam3QuerySpec | None = None
    bootstrap_dedup_metric: str = "iou"
    bootstrap_dedup_threshold: float = 0.40
    bootstrap_tiling_mode: TilingMode = TilingMode.OFF
    adaptive_tiling: AdaptiveTilingConfig = field(default_factory=AdaptiveTilingConfig)

    def __post_init__(self) -> None:
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be non-empty and unique")
        if self.target_class not in self.class_names:
            raise ValueError("target_class must belong to class_names")
        if set(self.prior) != set(self.class_names):
            raise ValueError("prior keys must match class_names")
        if not math.isclose(sum(float(value) for value in self.prior.values()), 1.0, abs_tol=1e-6):
            raise ValueError("prior must sum to one")
        if not 0.0 <= self.stopping_threshold <= 1.0:
            raise ValueError("stopping_threshold must lie in [0, 1]")
        if self.max_total_queries < 1 or self.max_queries_per_node < 1:
            raise ValueError("query budgets must be positive")
        if not 0.0 <= self.suppression_confidence <= 1.0:
            raise ValueError("suppression_confidence must lie in [0, 1]")


@dataclass(frozen=True)
class StaticAshtRunResult:
    run_id: str
    graph: OrchardGraph
    passes: tuple[PassRecord, ...]
    count: CountEstimate
    total_sam3_queries: int
    selected_node_ids: tuple[str, ...]
    records: tuple[CanonicalRecord, ...]


class _RecordCollector:
    """Collect records locally while optionally forwarding them to RunStore."""

    def __init__(self, sink=None, *, run_id: str | None = None):
        self.sink = sink
        if sink is not None:
            self.run_id = sink.run_id
            self._factory = None
        else:
            self.run_id = run_id or create_run_id()
            self._factory = IdFactory(self.run_id)
        self.records: list[CanonicalRecord] = []

    def new_id(self, kind: EntityKind) -> str:
        if self.sink is not None:
            return self.sink.new_id(kind)
        return self._factory.new(kind)

    def append(
        self,
        payload: CanonicalRecord,
        *,
        event_kind=None,
        pass_id: str | None = None,
        minimum_level: ReportingLevel = ReportingLevel.STANDARD,
    ):
        self.records.append(payload)
        if self.sink is not None:
            return self.sink.append(
                payload,
                event_kind=event_kind,
                pass_id=pass_id,
                minimum_level=minimum_level,
            )
        return None


class _PassCollector:
    def __init__(self, root: _RecordCollector):
        self.root = root
        self.run_id = root.run_id
        self.records: list[CanonicalRecord] = []

    def new_id(self, kind: EntityKind) -> str:
        return self.root.new_id(kind)

    def append(self, payload, **kwargs):
        self.records.append(payload)
        return self.root.append(payload, **kwargs)

    def records_of(self, record_type):
        return tuple(value for value in self.records if isinstance(value, record_type))


class StaticAshtRunner:
    def __init__(
        self,
        *,
        backend,
        processor,
        image_np: np.ndarray,
        graph: OrchardGraph,
        config: StaticAshtConfig,
        sink=None,
        model_id: str = "sam3",
    ) -> None:
        image = np.asarray(image_np)
        if image.ndim < 2:
            raise ValueError("image_np must have at least height and width dimensions")
        self.backend = backend
        self.processor = processor
        self.image_np = image
        self.graph = graph
        self.config = config
        self.model_id = model_id
        self.root = _RecordCollector(sink)
        self.passes: list[PassRecord] = []
        self.total_sam3_queries = 0
        self.bootstrap_sam3_queries = 0
        self.total_tile_calls = 0
        self.total_qwen_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.selected_node_ids: list[str] = []
        self._pass_index = 0
        self._started = time.perf_counter()

    def run(self) -> StaticAshtRunResult:
        if not self.graph.nodes:
            if self.config.bootstrap_query is None:
                raise ValueError("an empty graph requires bootstrap_query")
            self._run_bootstrap()
        self.graph.initialize_beliefs(self.config.prior)

        while True:
            unresolved = self.graph.unresolved_nodes()
            if not unresolved:
                break
            if self.total_sam3_queries >= self.config.max_total_queries:
                for node in unresolved:
                    self._terminal_stop(node, StopReason.BUDGET)
                break
            node = self._select_node(unresolved)
            self.selected_node_ids.append(node.id)
            actions = self.config.action_bank.materialize(
                graph=self.graph,
                node=node,
                class_names=self.config.class_names,
                image_width=int(self.image_np.shape[1]),
                image_height=int(self.image_np.shape[0]),
                id_source=self.root,
            )
            if not actions:
                self._terminal_stop(node, StopReason.NO_VALID_ACTION)
                continue
            self._run_verification_pass(node, actions)

        count = estimate_count(
            (node.belief for node in self.graph.nodes.values() if node.belief is not None),
            self.config.target_class,
        )
        return StaticAshtRunResult(
            run_id=self.root.run_id,
            graph=self.graph,
            passes=tuple(self.passes),
            count=count,
            total_sam3_queries=self.total_sam3_queries,
            selected_node_ids=tuple(self.selected_node_ids),
            records=tuple(self.root.records),
        )

    def _run_bootstrap(self) -> None:
        pass_sink, pass_id, started_at, graph_before = self._start_pass()
        context = StageContext(pass_id=pass_id, model_id=self.model_id, sink=pass_sink)
        batch = execute_sam3_request(
            self.backend,
            self.processor,
            self.image_np,
            self.config.bootstrap_query,
            context=context,
        )
        self.total_sam3_queries += 1
        self.bootstrap_sam3_queries += 1
        initial_suppression = suppress_detection_batch(
            batch,
            self.backend,
            confidence=self.config.suppression_confidence,
            context=context,
            **dict(self.config.suppression_kwargs),
        )
        final_suppression = initial_suppression
        tiling_decision = None
        if self.config.bootstrap_tiling_mode is not TilingMode.OFF:
            tiled = execute_adaptive_tiling(
                backend=self.backend,
                processor=self.processor,
                image_np=self.image_np,
                base_query=self.config.bootstrap_query,
                stage1_batch=initial_suppression.kept,
                mode=self.config.bootstrap_tiling_mode,
                context=context,
                config=self.config.adaptive_tiling,
                agent_requested=(
                    self.config.bootstrap_tiling_mode is TilingMode.AGENT_CONTROLLED
                ),
                max_additional_calls=max(
                    0, self.config.max_total_queries - self.total_sam3_queries
                ),
            )
            tiling_decision = tiled.decision
            self.total_sam3_queries += tiled.sam3_call_count
            self.bootstrap_sam3_queries += tiled.sam3_call_count
            self.total_tile_calls += tiled.sam3_call_count
            if tiled.decision.triggered:
                final_suppression = suppress_detection_batch(
                    tiled.merged,
                    self.backend,
                    confidence=self.config.suppression_confidence,
                    context=context,
                    **dict(self.config.suppression_kwargs),
                )
        registration = register_detection_batch(
            final_suppression.kept,
            self.graph,
            pass_number=self._pass_index,
            dedup_metric=self.config.bootstrap_dedup_metric,
            dedup_threshold=self.config.bootstrap_dedup_threshold,
            signature=self.config.bootstrap_query.prompt,
            class_names=self.config.class_names,
            context=context,
        )
        self.graph.initialize_beliefs(self.config.prior)
        cost = self._cost_record(pass_id)
        pass_record = PassRecord(
            pass_id=pass_id,
            run_id=self.root.run_id,
            pass_index=self._pass_index,
            started_at=started_at,
            completed_at=utc_now_iso(),
            target_node_id=None,
            graph_before=graph_before,
            candidate_action_set=None,
            information_gain_records=(),
            selected_action_id=None,
            qwen_calls=(),
            sam3_calls=pass_sink.records_of(Sam3CallRecord),
            tiles=pass_sink.records_of(TileRecord),
            raw_detections=pass_sink.records_of(RawDetectionRecord),
            dedup_comparisons=pass_sink.records_of(DedupComparisonRecord),
            registrations=pass_sink.records_of(RegistrationDecisionRecord),
            observation=None,
            kernel=None,
            belief_update=None,
            stopping_decision=None,
            graph_after=self._snapshots(pass_id),
            cost_after=cost,
            continuation_reason="bootstrap_complete",
            tiling_decision=tiling_decision,
            metadata={
                "created_node_ids": list(registration.created_node_ids),
                "updated_node_ids": list(registration.updated_node_ids),
                "removed_detection_ids": list(final_suppression.removed_detection_ids),
                "bootstrap_tiling_mode": self.config.bootstrap_tiling_mode.value,
            },
        )
        self._finish_pass(pass_sink, pass_record)

    def _run_verification_pass(
        self,
        node: OrchardNode,
        actions: tuple[MaterializedStaticAction, ...],
        *,
        prepared_pass=None,
        candidate_set_override: CandidateActionSetRecord | None = None,
        qwen_calls=(),
    ) -> None:
        if prepared_pass is None:
            pass_sink, pass_id, started_at, graph_before = self._start_pass()
        else:
            pass_sink, pass_id, started_at, graph_before = prepared_pass
        for qwen_call in qwen_calls:
            pass_sink.append(qwen_call, pass_id=pass_id)
        for action in actions:
            pass_sink.append(action.record, pass_id=pass_id)
        candidate_set = candidate_set_override or CandidateActionSetRecord(
            candidate_set_id=pass_sink.new_id(EntityKind.CANDIDATE),
            pass_id=pass_id,
            target_node_id=node.id,
            actions=tuple(action.record for action in actions),
            generation_policy="static_action_bank",
        )
        pass_sink.append(candidate_set, pass_id=pass_id)

        kernels = {action.record.action_id: action.kernel for action in actions}
        costs = {action.record.action_id: action.expected_cost for action in actions}
        ranked = rank_actions(
            node.belief.posterior,
            kernels,
            expected_costs=(costs if self.config.use_cost_adjusted_eig else None),
        )
        ranking_by_id = {row.action_id: row for row in ranked}
        selected_id = ranked[0].action_id
        selected = next(action for action in actions if action.record.action_id == selected_id)

        kernel_ids = {
            action.record.action_id: pass_sink.new_id(EntityKind.KERNEL)
            for action in actions
        }
        ig_ids = {
            action.record.action_id: pass_sink.new_id(EntityKind.INFORMATION_GAIN)
            for action in actions
        }
        ig_records: list[InformationGainRecord] = []
        for action in actions:
            if action.record.action_id == selected_id:
                continue
            kernel_record = _kernel_record(
                action,
                kernel_id=kernel_ids[action.record.action_id],
                pass_id=pass_id,
            )
            rank = ranking_by_id[action.record.action_id]
            ig_record = _ig_record(
                action,
                kernel_id=kernel_record.kernel_id,
                information_gain_id=ig_ids[action.record.action_id],
                pass_id=pass_id,
                posterior=node.belief.as_mapping(),
                rank=rank.rank,
                selected=False,
                use_cost_adjustment=self.config.use_cost_adjusted_eig,
            )
            pass_sink.append(kernel_record, pass_id=pass_id)
            pass_sink.append(ig_record, pass_id=pass_id)
            ig_records.append(ig_record)

        query_image, query = self._query_for_action(selected.record)
        context = StageContext(pass_id=pass_id, model_id=self.model_id, sink=pass_sink)
        batch = execute_sam3_request(
            self.backend,
            self.processor,
            query_image,
            query,
            context=context,
        )
        self.total_sam3_queries += 1
        initial_suppression = suppress_detection_batch(
            batch,
            self.backend,
            confidence=self.config.suppression_confidence,
            context=context,
            **dict(self.config.suppression_kwargs),
        )
        final_suppression = initial_suppression
        tiling_decision = None
        if selected.record.tiling_mode is not TilingMode.OFF:
            tiled = execute_adaptive_tiling(
                backend=self.backend,
                processor=self.processor,
                image_np=query_image,
                base_query=query,
                stage1_batch=initial_suppression.kept,
                mode=selected.record.tiling_mode,
                context=context,
                config=self.config.adaptive_tiling,
                tile_scale=selected.record.tile_scale,
                agent_requested=(
                    selected.record.tiling_mode is TilingMode.AGENT_CONTROLLED
                ),
                max_additional_calls=max(
                    0, self.config.max_total_queries - self.total_sam3_queries
                ),
            )
            tiling_decision = tiled.decision
            self.total_sam3_queries += tiled.sam3_call_count
            self.total_tile_calls += tiled.sam3_call_count
            if tiled.decision.triggered:
                final_suppression = suppress_detection_batch(
                    tiled.merged,
                    self.backend,
                    confidence=self.config.suppression_confidence,
                    context=context,
                    **dict(self.config.suppression_kwargs),
                )

        observation = encode_target_observation(
            node.box,
            final_suppression.kept.boxes_global,
            final_suppression.kept.scores,
            config=self.config.observation_encoder,
        )
        matched_detection_ids = tuple(
            final_suppression.kept.detection_ids[index]
            for index in observation.matched_indices
            if final_suppression.kept.detection_ids
        )
        node_budget_hit = node.belief.query_count + 1 >= self.config.max_queries_per_node
        global_budget_hit = self.total_sam3_queries >= self.config.max_total_queries
        selected_rank = ranking_by_id[selected_id]
        update_ids = UpdateIds(
            pass_id=pass_id,
            action_id=selected_id,
            kernel_id=kernel_ids[selected_id],
            observation_id=pass_sink.new_id(EntityKind.OBSERVATION),
            information_gain_id=ig_ids[selected_id],
            belief_update_id=pass_sink.new_id(EntityKind.BELIEF_UPDATE),
            stopping_id=pass_sink.new_id(EntityKind.STOPPING),
        )
        update = perform_update(
            node.belief,
            selected.kernel,
            observation,
            ids=update_ids,
            stopping_threshold=self.config.stopping_threshold,
            matched_detection_ids=matched_detection_ids,
            expected_cost=selected.expected_cost,
            rank=selected_rank.rank,
            selected=True,
            budget_remaining={
                "total_queries": max(0, self.config.max_total_queries - self.total_sam3_queries),
                "node_queries": max(0, self.config.max_queries_per_node - (node.belief.query_count + 1)),
            },
            budget_exhausted=node_budget_hit or global_budget_hit,
        )
        for record in (
            update.kernel_record,
            update.information_gain_record,
            update.observation_record,
            update.belief_update_record,
            update.stopping_record,
        ):
            pass_sink.append(record, pass_id=pass_id)
        ig_records.append(update.information_gain_record)
        ig_records.sort(key=lambda value: value.rank)

        node.record_verification(
            action_id=selected.record.action_id,
            semantic_key=selected.record.semantic_key,
            observation_id=update.observation_record.observation_id,
            belief_update_id=update.belief_update_record.belief_update_id,
            positive_exemplar_node_ids=selected.record.positive_exemplar_node_ids,
            negative_exemplar_node_ids=selected.record.negative_exemplar_node_ids,
        )
        cost = self._cost_record(pass_id)
        pass_record = PassRecord(
            pass_id=pass_id,
            run_id=self.root.run_id,
            pass_index=self._pass_index,
            started_at=started_at,
            completed_at=utc_now_iso(),
            target_node_id=node.id,
            graph_before=graph_before,
            candidate_action_set=candidate_set,
            information_gain_records=tuple(ig_records),
            selected_action_id=selected_id,
            qwen_calls=tuple(qwen_calls),
            sam3_calls=pass_sink.records_of(Sam3CallRecord),
            tiles=pass_sink.records_of(TileRecord),
            raw_detections=pass_sink.records_of(RawDetectionRecord),
            dedup_comparisons=pass_sink.records_of(DedupComparisonRecord),
            registrations=(),
            observation=update.observation_record,
            kernel=update.kernel_record,
            belief_update=update.belief_update_record,
            stopping_decision=update.stopping_record,
            graph_after=self._snapshots(pass_id),
            cost_after=cost,
            continuation_reason=(
                "node_stopped" if update.stop_decision.should_stop else "node_unresolved"
            ),
            tiling_decision=tiling_decision,
            metadata={
                "selected_semantic_key": selected.record.semantic_key,
                "removed_detection_ids": list(final_suppression.removed_detection_ids),
                "tiling_mode": selected.record.tiling_mode.value,
            },
        )
        self._finish_pass(pass_sink, pass_record)

    def _terminal_stop(self, node: OrchardNode, reason: StopReason) -> None:
        pass_sink, pass_id, started_at, graph_before = self._start_pass()
        text_reason = "budget" if reason is StopReason.BUDGET else "no_valid_action"
        node.belief.mark_stopped(reason=text_reason)
        stopping = StoppingDecisionRecord(
            stopping_id=pass_sink.new_id(EntityKind.STOPPING),
            pass_id=pass_id,
            graph_node_id=node.id,
            posterior=node.belief.as_mapping(),
            threshold=self.config.stopping_threshold,
            should_stop=True,
            declaration=node.belief.decision,
            reason=reason,
            query_count=node.belief.query_count,
            budget_remaining={
                "total_queries": max(0, self.config.max_total_queries - self.total_sam3_queries),
                "node_queries": max(0, self.config.max_queries_per_node - node.belief.query_count),
            },
        )
        pass_sink.append(stopping, pass_id=pass_id)
        cost = self._cost_record(pass_id)
        pass_record = PassRecord(
            pass_id=pass_id,
            run_id=self.root.run_id,
            pass_index=self._pass_index,
            started_at=started_at,
            completed_at=utc_now_iso(),
            target_node_id=node.id,
            graph_before=graph_before,
            candidate_action_set=None,
            information_gain_records=(),
            selected_action_id=None,
            qwen_calls=(),
            sam3_calls=(),
            tiles=(),
            raw_detections=(),
            dedup_comparisons=(),
            registrations=(),
            observation=None,
            kernel=None,
            belief_update=None,
            stopping_decision=stopping,
            graph_after=self._snapshots(pass_id),
            cost_after=cost,
            continuation_reason=reason.value,
        )
        self._finish_pass(pass_sink, pass_record)

    def _select_node(self, nodes: tuple[OrchardNode, ...]) -> OrchardNode:
        return sorted(
            nodes,
            key=lambda node: (-entropy(node.belief.posterior), node.id),
        )[0]

    def _query_for_action(self, action: SensingActionRecord):
        x1, y1, x2, y2 = _integer_region(action.region, self.image_np.shape)
        crop = self.image_np[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"action {action.action_id} produced an empty ROI")
        region = (float(x1), float(y1), float(x2), float(y2))
        positive_boxes = tuple(
            _box_to_local(self.graph.nodes[node_id].box, x1, y1, x2 - x1, y2 - y1)
            for node_id in action.positive_exemplar_node_ids
            if node_id in self.graph.nodes
        )
        negative_boxes = tuple(
            _box_to_local(self.graph.nodes[node_id].box, x1, y1, x2 - x1, y2 - y1)
            for node_id in action.negative_exemplar_node_ids
            if node_id in self.graph.nodes
        )
        query = Sam3QuerySpec(
            prompt=action.prompt,
            threshold=action.threshold,
            region=region,
            positive_boxes=positive_boxes,
            negative_boxes=negative_boxes,
            positive_exemplar_node_ids=action.positive_exemplar_node_ids,
            negative_exemplar_node_ids=action.negative_exemplar_node_ids,
            coordinate_space=CoordinateSpace.ROI_LOCAL,
            return_masks=False,
            decoder_settings={"semantic_key": action.semantic_key},
        )
        return crop, query

    def _start_pass(self):
        self._pass_index += 1
        pass_sink = _PassCollector(self.root)
        pass_id = pass_sink.new_id(EntityKind.PASS)
        return pass_sink, pass_id, utc_now_iso(), self._snapshots(pass_id)

    def _finish_pass(self, pass_sink: _PassCollector, record: PassRecord) -> None:
        for snapshot in record.graph_after:
            pass_sink.append(snapshot, pass_id=record.pass_id)
        pass_sink.append(record.cost_after, pass_id=record.pass_id)
        pass_sink.append(record, pass_id=record.pass_id)
        self.passes.append(record)

    def _snapshots(self, pass_id: str):
        return self.graph.snapshot_records(
            pass_id=pass_id,
            class_names=self.config.class_names,
            positive_class=self.config.action_bank.positive_class,
            negative_class=self.config.action_bank.negative_class,
            positive_threshold=self.config.action_bank.positive_exemplar_threshold,
            negative_threshold=self.config.action_bank.negative_exemplar_threshold,
        )

    def _cost_record(self, pass_id: str) -> CostSnapshotRecord:
        return CostSnapshotRecord(
            pass_id=pass_id,
            sam3_calls=self.total_sam3_queries,
            qwen_calls=self.total_qwen_calls,
            tile_calls=self.total_tile_calls,
            verify_calls=max(0, self.total_sam3_queries - self.bootstrap_sam3_queries),
            orchestration_calls=self.total_qwen_calls,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
            runtime_seconds=float(time.perf_counter() - self._started),
            normalized_cost=float(self.total_sam3_queries + self.total_qwen_calls),
            metadata={"cost_semantics": "cumulative"},
        )


def _kernel_record(
    action: MaterializedStaticAction,
    *,
    kernel_id: str,
    pass_id: str,
) -> SurrogateKernelRecord:
    kernel = action.kernel
    return SurrogateKernelRecord(
        kernel_id=kernel_id,
        pass_id=pass_id,
        action_id=action.record.action_id,
        class_names=kernel.class_names,
        observation_labels=kernel.observation_labels,
        beta_by_class=dict(kernel.beta_by_class),
        present_profile=tuple(float(value) for value in kernel.present_profile),
        absent_profile=tuple(float(value) for value in kernel.absent_profile),
        kernel_matrix=tuple(
            tuple(float(value) for value in row) for row in kernel.matrix
        ),
        epsilon=kernel.epsilon,
        profile_source=kernel.profile_source,
    )


def _ig_record(
    action: MaterializedStaticAction,
    *,
    kernel_id: str,
    information_gain_id: str,
    pass_id: str,
    posterior: Mapping[str, float],
    rank: int,
    selected: bool,
    use_cost_adjustment: bool,
) -> InformationGainRecord:
    vector = [posterior[name] for name in action.kernel.class_names]
    predictive = predictive_distribution(vector, action.kernel)
    eig = expected_information_gain(vector, action.kernel)
    return InformationGainRecord(
        information_gain_id=information_gain_id,
        pass_id=pass_id,
        action_id=action.record.action_id,
        kernel_id=kernel_id,
        posterior=dict(posterior),
        predictive_observation={
            label: float(predictive[index])
            for index, label in enumerate(action.kernel.observation_labels)
        },
        expected_information_gain=eig,
        expected_cost=action.expected_cost,
        cost_adjusted_score=(
            eig / action.expected_cost if use_cost_adjustment else None
        ),
        selected=selected,
        rank=rank,
    )


def _integer_region(region, image_shape):
    height, width = int(image_shape[0]), int(image_shape[1])
    x1 = max(0, min(width, int(math.floor(float(region[0])))))
    y1 = max(0, min(height, int(math.floor(float(region[1])))))
    x2 = max(x1, min(width, int(math.ceil(float(region[2])))))
    y2 = max(y1, min(height, int(math.ceil(float(region[3])))))
    return x1, y1, x2, y2


def _box_to_local(box, x1: int, y1: int, width: int, height: int):
    bx1, by1, bx2, by2 = (float(value) for value in box)
    return (
        max(0.0, min(float(width), bx1 - x1)),
        max(0.0, min(float(height), by1 - y1)),
        max(0.0, min(float(width), bx2 - x1)),
        max(0.0, min(float(height), by2 - y1)),
    )


__all__ = ["StaticAshtConfig", "StaticAshtRunResult", "StaticAshtRunner"]

"""Provenance-complete discovery-only experiment executor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.asht.tiling import (
    AdaptiveTilingConfig,
    concatenate_batches,
    execute_adaptive_tiling,
    suppress_cross_tile_duplicates,
)
from experiments.config import DiscoveryPassSpec
from graph import OrchardGraph
from pipeline_stages import (
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
    PassRecord,
    RawDetectionRecord,
    RegistrationDecisionRecord,
    Sam3CallRecord,
    SensingActionRecord,
    TileRecord,
    TilingDecisionRecord,
)
from provenance.ids import EntityKind
from provenance.run_store import ReportingLevel
from provenance.schema import ActionFamily, ActionKind, CoordinateSpace, TilingMode


@dataclass(frozen=True)
class DiscoveryExecutionResult:
    graph: OrchardGraph
    passes: tuple[PassRecord, ...]
    sam3_calls: int
    tile_calls: int
    records: tuple[CanonicalRecord, ...]


class _PassSink:
    def __init__(self, store) -> None:
        self.store = store
        self.run_id = store.run_id
        self.records: list[CanonicalRecord] = []

    def new_id(self, kind: EntityKind) -> str:
        return self.store.new_id(kind)

    def append(self, payload, **kwargs):
        # Pass-local records are consolidated into one PassRecord.  Writing
        # every nested item separately duplicates the same data in events.jsonl.
        self.records.append(payload)
        return None

    def records_of(self, record_type):
        return tuple(item for item in self.records if isinstance(item, record_type))


class DiscoveryExperimentExecutor:
    """Run one or more discovery passes under a single canonical interface."""

    def __init__(
        self,
        *,
        backend: Any,
        processor: Any,
        image_np: np.ndarray,
        graph: OrchardGraph,
        store,
        class_names: tuple[str, ...],
        model_id: str = "sam3",
        adaptive_tiling: AdaptiveTilingConfig | None = None,
        suppression_confidence: float = 0.0,
        suppression_kwargs: dict[str, Any] | None = None,
        cross_pass_dedup_metric: str = "iou",
        cross_pass_dedup_threshold: float = 0.40,
    ) -> None:
        self.backend = backend
        self.processor = processor
        self.image_np = np.asarray(image_np)
        self.graph = graph
        self.store = store
        self.class_names = class_names
        self.model_id = model_id
        self.adaptive_tiling = adaptive_tiling or AdaptiveTilingConfig()
        self.suppression_confidence = suppression_confidence
        self.suppression_kwargs = dict(suppression_kwargs or {})
        self.cross_pass_dedup_metric = cross_pass_dedup_metric
        self.cross_pass_dedup_threshold = cross_pass_dedup_threshold
        self.passes: list[PassRecord] = []
        self.records: list[CanonicalRecord] = []
        self.sam3_calls = 0
        self.tile_calls = 0
        self._started = time.perf_counter()

    def run(self, pass_specs: tuple[DiscoveryPassSpec, ...]) -> DiscoveryExecutionResult:
        for pass_index, specification in enumerate(pass_specs):
            self._run_pass(pass_index, specification, total_passes=len(pass_specs))
        return DiscoveryExecutionResult(
            graph=self.graph,
            passes=tuple(self.passes),
            sam3_calls=self.sam3_calls,
            tile_calls=self.tile_calls,
            records=tuple(self.records),
        )

    def _run_pass(
        self,
        pass_index: int,
        specification: DiscoveryPassSpec,
        *,
        total_passes: int,
    ) -> None:
        scope = _PassSink(self.store)
        pass_id = scope.new_id(EntityKind.PASS)
        started_at = utc_now_iso()
        graph_before = ()
        height, width = int(self.image_np.shape[0]), int(self.image_np.shape[1])
        region = specification.region or (0.0, 0.0, float(width), float(height))
        action = SensingActionRecord(
            action_id=scope.new_id(EntityKind.ACTION),
            action_kind=ActionKind.DISCOVER,
            family=ActionFamily.SYSTEM,
            prompt=specification.prompt,
            region=region,
            threshold=specification.threshold,
            semantic_key=specification.signature or specification.prompt.strip().lower(),
            tiling_mode=specification.tiling_mode,
            tile_scale=specification.tile_scale,
            generator="unified_experiment_runner",
            metadata=dict(specification.metadata),
        )
        candidate_set = CandidateActionSetRecord(
            candidate_set_id=scope.new_id(EntityKind.CANDIDATE),
            pass_id=pass_id,
            target_node_id=None,
            actions=(action,),
            generation_policy="configured_discovery_pass",
        )
        scope.append(action, pass_id=pass_id)
        scope.append(candidate_set, pass_id=pass_id)

        query = Sam3QuerySpec(
            prompt=specification.prompt,
            threshold=specification.threshold,
            region=region,
            coordinate_space=CoordinateSpace.IMAGE_GLOBAL,
            return_masks=specification.return_masks,
            preprocessing={"unified_runner": True},
            decoder_settings={"pass_index": pass_index},
        )
        context = StageContext(pass_id=pass_id, model_id=self.model_id, sink=scope)
        stage_one = execute_sam3_request(
            self.backend,
            self.processor,
            self.image_np,
            query,
            context=context,
        )
        self.sam3_calls += 1
        initial = suppress_detection_batch(
            stage_one,
            self.backend,
            confidence=self.suppression_confidence,
            context=context,
            **self.suppression_kwargs,
        )

        tiling_result = None
        final_batch = initial.kept
        extra_comparisons: tuple[DedupComparisonRecord, ...] = ()
        if specification.tiling_mode is not TilingMode.OFF:
            tiling_result = execute_adaptive_tiling(
                backend=self.backend,
                processor=self.processor,
                image_np=self.image_np,
                base_query=query,
                stage1_batch=initial.kept,
                mode=specification.tiling_mode,
                context=context,
                config=self.adaptive_tiling,
                tile_scale=specification.tile_scale,
                agent_requested=specification.tiling_mode is TilingMode.AGENT_CONTROLLED,
            )
            self.sam3_calls += tiling_result.sam3_call_count
            self.tile_calls += tiling_result.sam3_call_count
            if tiling_result.sam3_call_count:
                combined = concatenate_batches(
                    (initial.kept, tiling_result.merged),
                    source_region=region,
                    prompt=specification.prompt,
                )
                final_batch, extra_comparisons = suppress_cross_tile_duplicates(
                    combined,
                    pass_id=pass_id,
                    id_source=scope,
                    iou_threshold=self.adaptive_tiling.cross_tile_iou_threshold,
                )
                for comparison in extra_comparisons:
                    scope.append(
                        comparison,
                        pass_id=pass_id,
                        minimum_level=ReportingLevel.FULL,
                    )

        registration = register_detection_batch(
            final_batch,
            self.graph,
            pass_number=pass_index + 1,
            dedup_metric=self.cross_pass_dedup_metric,
            dedup_threshold=self.cross_pass_dedup_threshold,
            signature=specification.signature or action.semantic_key,
            context=context,
            class_names=self.class_names,
        )
        graph_after = self._snapshots_for_ids(
            pass_id,
            (*registration.created_node_ids, *registration.updated_node_ids),
        )
        cost = CostSnapshotRecord(
            pass_id=pass_id,
            sam3_calls=self.sam3_calls,
            qwen_calls=0,
            tile_calls=self.tile_calls,
            verify_calls=0,
            orchestration_calls=0,
            input_tokens=0,
            output_tokens=0,
            runtime_seconds=float(time.perf_counter() - self._started),
            normalized_cost=float(self.sam3_calls),
            metadata={"cost_semantics": "cumulative"},
        )
        scope.append(cost, pass_id=pass_id)
        record = PassRecord(
            pass_id=pass_id,
            run_id=self.store.run_id,
            pass_index=pass_index,
            started_at=started_at,
            completed_at=utc_now_iso(),
            target_node_id=None,
            graph_before=graph_before,
            candidate_action_set=candidate_set,
            information_gain_records=(),
            selected_action_id=action.action_id,
            qwen_calls=(),
            sam3_calls=scope.records_of(Sam3CallRecord),
            tiles=scope.records_of(TileRecord),
            raw_detections=scope.records_of(RawDetectionRecord),
            dedup_comparisons=scope.records_of(DedupComparisonRecord),
            registrations=scope.records_of(RegistrationDecisionRecord),
            observation=None,
            kernel=None,
            belief_update=None,
            stopping_decision=None,
            graph_after=graph_after,
            cost_after=cost,
            continuation_reason=(
                "more_configured_passes"
                if pass_index + 1 < total_passes
                else "configured_discovery_sequence_complete"
            ),
            tiling_decision=(
                tiling_result.decision if tiling_result is not None else None
            ),
            metadata={
                "executor": "discovery",
                "created_node_ids": list(registration.created_node_ids),
                "updated_node_ids": list(registration.updated_node_ids),
                "raw_detection_count": len(stage_one),
                "final_detection_count": len(final_batch),
                "extra_cross_tile_comparison_count": len(extra_comparisons),
            },
        )
        self.store.append(record, pass_id=pass_id)
        self.passes.append(record)
        self.records.extend(scope.records)

    def _snapshots_for_ids(self, pass_id: str, node_ids):
        wanted = set(node_ids)
        return tuple(
            snapshot
            for snapshot in self.graph.snapshot_records(
                pass_id=pass_id,
                class_names=self.class_names,
                positive_class=self.class_names[0],
                negative_class=self.class_names[1] if len(self.class_names) > 1 else None,
            )
            if snapshot.graph_node_id in wanted
        )


__all__ = ["DiscoveryExecutionResult", "DiscoveryExperimentExecutor"]

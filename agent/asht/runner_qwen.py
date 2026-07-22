"""Qwen-generated candidate-action ASHT controller.

Qwen proposes a finite action set; the inherited numerical controller ranks it
with EIG, executes SAM3, and performs the same Bayesian update as the static
runner.  Qwen never selects the winning action and never inserts detections.
"""

from __future__ import annotations

from agent.asht.counting import estimate_count
from agent.asht.qwen_generator import QwenCandidateActionGenerator
from agent.asht.runner_static import StaticAshtRunResult, StaticAshtRunner
from provenance.base import utc_now_iso
from provenance.contracts import (
    CostSnapshotRecord,
    PassRecord,
    QwenCallRecord,
    StoppingDecisionRecord,
)
from provenance.ids import EntityKind
from provenance.schema import StopReason


class QwenAshtRunner(StaticAshtRunner):
    def __init__(self, *, action_generator: QwenCandidateActionGenerator, **kwargs):
        super().__init__(**kwargs)
        self.action_generator = action_generator

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
            prepared = self._start_pass()
            pass_sink, pass_id, _, _ = prepared
            generated = self.action_generator.generate(
                graph=self.graph,
                node=node,
                class_names=self.config.class_names,
                image=self.image_np,
                image_width=int(self.image_np.shape[1]),
                image_height=int(self.image_np.shape[0]),
                pass_id=pass_id,
                id_source=pass_sink,
            )
            self.total_qwen_calls += 1
            self.total_input_tokens += generated.qwen_call.input_tokens or 0
            self.total_output_tokens += generated.qwen_call.output_tokens or 0
            if not generated.actions:
                self._finish_generation_failure(
                    node,
                    prepared,
                    qwen_call=generated.qwen_call,
                    candidate_set=generated.candidate_set,
                )
                continue
            self._run_verification_pass(
                node,
                generated.actions,
                prepared_pass=prepared,
                candidate_set_override=generated.candidate_set,
                qwen_calls=(generated.qwen_call,),
            )

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

    def _finish_generation_failure(
        self,
        node,
        prepared_pass,
        *,
        qwen_call: QwenCallRecord,
        candidate_set,
    ) -> None:
        pass_sink, pass_id, started_at, graph_before = prepared_pass
        pass_sink.append(qwen_call, pass_id=pass_id)
        pass_sink.append(candidate_set, pass_id=pass_id)
        node.belief.mark_stopped(reason="no_valid_action")
        stopping = StoppingDecisionRecord(
            stopping_id=pass_sink.new_id(EntityKind.STOPPING),
            pass_id=pass_id,
            graph_node_id=node.id,
            posterior=node.belief.as_mapping(),
            threshold=self.config.stopping_threshold,
            should_stop=True,
            declaration=node.belief.decision,
            reason=StopReason.NO_VALID_ACTION,
            query_count=node.belief.query_count,
            budget_remaining={
                "total_queries": max(
                    0, self.config.max_total_queries - self.total_sam3_queries
                ),
                "node_queries": max(
                    0, self.config.max_queries_per_node - node.belief.query_count
                ),
            },
        )
        pass_sink.append(stopping, pass_id=pass_id)
        cost: CostSnapshotRecord = self._cost_record(pass_id)
        record = PassRecord(
            pass_id=pass_id,
            run_id=self.root.run_id,
            pass_index=self._pass_index,
            started_at=started_at,
            completed_at=utc_now_iso(),
            target_node_id=node.id,
            graph_before=graph_before,
            candidate_action_set=candidate_set,
            information_gain_records=(),
            selected_action_id=None,
            qwen_calls=(qwen_call,),
            sam3_calls=(),
            tiles=(),
            raw_detections=(),
            dedup_comparisons=(),
            registrations=(),
            observation=None,
            kernel=None,
            belief_update=None,
            stopping_decision=stopping,
            graph_after=self._snapshots_for_ids(pass_id, (node.id,)),
            cost_after=cost,
            continuation_reason="no_valid_action",
            metadata={"generation_failed": True},
        )
        self._finish_pass(pass_sink, record)


__all__ = ["QwenAshtRunner"]

"""One complete mathematical update with canonical provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agent.asht.belief import PatchBelief
from agent.asht.information import (
    entropy,
    expected_information_gain,
    kl_divergence,
    predictive_distribution,
    realized_entropy_reduction,
)
from agent.asht.kernels import SurrogateKernel
from agent.asht.observations import EncodedObservation
from agent.asht.stopping import StopDecision, posterior_stop
from provenance.contracts import (
    BeliefUpdateRecord,
    EncodedObservationRecord,
    InformationGainRecord,
    StoppingDecisionRecord,
    SurrogateKernelRecord,
)
from provenance.ids import EntityKind
from provenance.schema import StopReason


@dataclass(frozen=True)
class UpdateIds:
    pass_id: str
    action_id: str
    kernel_id: str
    observation_id: str
    information_gain_id: str
    belief_update_id: str
    stopping_id: str


@dataclass(frozen=True)
class MathematicalUpdate:
    kernel_record: SurrogateKernelRecord
    information_gain_record: InformationGainRecord
    observation_record: EncodedObservationRecord
    belief_update_record: BeliefUpdateRecord
    stopping_record: StoppingDecisionRecord
    stop_decision: StopDecision


def perform_update(
    belief: PatchBelief,
    kernel: SurrogateKernel,
    observation: EncodedObservation,
    *,
    ids: UpdateIds,
    stopping_threshold: float,
    matched_detection_ids: tuple[str, ...] = (),
    expected_cost: float | None = None,
    rank: int = 1,
    selected: bool = True,
    budget_remaining: Mapping[str, object] | None = None,
    budget_exhausted: bool = False,
) -> MathematicalUpdate:
    eig = expected_information_gain(belief.posterior, kernel)
    predictive = predictive_distribution(belief.posterior, kernel)
    likelihood = kernel.likelihood(observation.index)
    entropy_before = entropy(belief.posterior)
    before, after, predictive_probability = belief.apply_likelihood(
        likelihood,
        evidence_id=ids.observation_id,
    )
    entropy_after = entropy(after)
    realized_kl = kl_divergence(after, before)
    entropy_change = realized_entropy_reduction(before, after)
    stop = posterior_stop(
        belief.class_names,
        after,
        threshold=stopping_threshold,
        budget_exhausted=budget_exhausted,
    )
    if stop.should_stop:
        belief.mark_stopped(reason=stop.reason, declaration=stop.declaration)

    kernel_record = SurrogateKernelRecord(
        kernel_id=ids.kernel_id,
        pass_id=ids.pass_id,
        action_id=ids.action_id,
        class_names=kernel.class_names,
        observation_labels=kernel.observation_labels,
        beta_by_class=dict(kernel.beta_by_class),
        present_profile=tuple(float(v) for v in kernel.present_profile),
        absent_profile=tuple(float(v) for v in kernel.absent_profile),
        kernel_matrix=tuple(tuple(float(v) for v in row) for row in kernel.matrix),
        epsilon=kernel.epsilon,
        profile_source=kernel.profile_source,
    )
    ig_record = InformationGainRecord(
        information_gain_id=ids.information_gain_id,
        pass_id=ids.pass_id,
        action_id=ids.action_id,
        kernel_id=ids.kernel_id,
        posterior={name: float(before[i]) for i, name in enumerate(belief.class_names)},
        predictive_observation={
            label: float(predictive[index])
            for index, label in enumerate(kernel.observation_labels)
        },
        expected_information_gain=eig,
        expected_cost=expected_cost,
        cost_adjusted_score=(eig / expected_cost if expected_cost else None),
        selected=selected,
        rank=rank,
    )
    observation_record = EncodedObservationRecord(
        observation_id=ids.observation_id,
        pass_id=ids.pass_id,
        action_id=ids.action_id,
        target_node_id=belief.node_id,
        level=observation.level,
        label=observation.label,
        overlap_metrics=dict(observation.overlap_metrics),
        score_statistics=dict(observation.score_statistics),
        bin_thresholds=dict(observation.bin_thresholds),
        matched_detection_ids=matched_detection_ids,
        encoder_version=str(observation.raw_features.get("encoder_version", "unknown")),
        raw_features=dict(observation.raw_features),
    )
    update_record = BeliefUpdateRecord(
        belief_update_id=ids.belief_update_id,
        pass_id=ids.pass_id,
        graph_node_id=belief.node_id,
        action_id=ids.action_id,
        observation_id=ids.observation_id,
        kernel_id=ids.kernel_id,
        posterior_before={name: float(before[i]) for i, name in enumerate(belief.class_names)},
        likelihood_by_class={name: float(likelihood[i]) for i, name in enumerate(belief.class_names)},
        predictive_observation_probability=predictive_probability,
        posterior_after={name: float(after[i]) for i, name in enumerate(belief.class_names)},
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        expected_information_gain=eig,
        realized_kl=realized_kl,
        realized_entropy_change=entropy_change,
    )
    stop_reason = {
        "confidence": StopReason.CONFIDENCE,
        "budget": StopReason.BUDGET,
        "not_stopped": StopReason.NOT_STOPPED,
    }[stop.reason]
    stopping_record = StoppingDecisionRecord(
        stopping_id=ids.stopping_id,
        pass_id=ids.pass_id,
        graph_node_id=belief.node_id,
        posterior={name: float(after[i]) for i, name in enumerate(belief.class_names)},
        threshold=stopping_threshold,
        should_stop=stop.should_stop,
        declaration=stop.declaration,
        reason=stop_reason,
        query_count=belief.query_count,
        budget_remaining=dict(budget_remaining or {}),
    )
    return MathematicalUpdate(
        kernel_record=kernel_record,
        information_gain_record=ig_record,
        observation_record=observation_record,
        belief_update_record=update_record,
        stopping_record=stopping_record,
        stop_decision=stop,
    )


def allocate_update_ids(sink, *, pass_id: str, action_id: str) -> UpdateIds:
    return UpdateIds(
        pass_id=pass_id,
        action_id=action_id,
        kernel_id=sink.new_id(EntityKind.KERNEL),
        observation_id=sink.new_id(EntityKind.OBSERVATION),
        information_gain_id=sink.new_id(EntityKind.INFORMATION_GAIN),
        belief_update_id=sink.new_id(EntityKind.BELIEF_UPDATE),
        stopping_id=sink.new_id(EntityKind.STOPPING),
    )


__all__ = [
    "MathematicalUpdate",
    "UpdateIds",
    "allocate_update_ids",
    "perform_update",
]

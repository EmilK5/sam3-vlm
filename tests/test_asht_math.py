from __future__ import annotations

import numpy as np

from agent.asht import (
    ObservationEncoderConfig,
    PatchBelief,
    SensorProfile,
    UpdateIds,
    build_surrogate_kernel,
    encode_target_observation,
    entropy,
    estimate_count,
    expected_information_gain,
    kl_divergence,
    perform_update,
    predictive_distribution,
    rank_actions,
)
from provenance.ids import EntityKind, IdFactory, create_run_id


def _kernel(beta):
    profile = SensorProfile(
        observation_labels=("not_found", "weak_match", "strong_match"),
        present=(0.05, 0.15, 0.80),
        absent=(0.80, 0.15, 0.05),
        source="unit-test",
    )
    return build_surrogate_kernel(
        ("fruit", "leaf", "background"),
        beta,
        profile,
    )


def test_identical_class_kernels_have_zero_information_gain():
    kernel = _kernel({"fruit": 0.5, "leaf": 0.5, "background": 0.5})
    eig = expected_information_gain([0.2, 0.5, 0.3], kernel)
    assert eig < 1e-12


def test_informative_kernel_has_positive_eig_and_normalized_prediction():
    kernel = _kernel({"fruit": 0.95, "leaf": 0.1, "background": 0.02})
    posterior = [0.2, 0.6, 0.2]
    predictive = predictive_distribution(posterior, kernel)
    assert np.isclose(predictive.sum(), 1.0)
    assert expected_information_gain(posterior, kernel) > 0.0


def test_action_ranking_supports_cost_adjustment():
    strong = _kernel({"fruit": 0.95, "leaf": 0.05, "background": 0.05})
    weak = _kernel({"fruit": 0.6, "leaf": 0.4, "background": 0.4})
    ranked = rank_actions(
        [1 / 3, 1 / 3, 1 / 3],
        {"strong": strong, "weak": weak},
        expected_costs={"strong": 10.0, "weak": 1.0},
    )
    assert {row.action_id for row in ranked} == {"strong", "weak"}
    assert ranked[0].score >= ranked[1].score


def test_observation_encoder_uses_overlap_and_score():
    observation = encode_target_observation(
        [0, 0, 10, 10],
        [[0, 0, 10, 10], [20, 20, 30, 30]],
        [0.9, 0.99],
        config=ObservationEncoderConfig(),
    )
    assert observation.label == "strong_match"
    assert observation.best_index == 0


def test_complete_update_emits_valid_canonical_records_and_stops():
    factory = IdFactory(create_run_id(random_token="math"))
    pass_id = factory.new(EntityKind.PASS)
    action_id = factory.new(EntityKind.ACTION)
    node_id = factory.new(EntityKind.GRAPH_NODE)
    belief = PatchBelief.from_mapping(
        node_id,
        {"fruit": 0.4, "leaf": 0.5, "background": 0.1},
    )
    kernel = _kernel({"fruit": 0.99, "leaf": 0.01, "background": 0.01})
    observation = encode_target_observation(
        [0, 0, 10, 10], [[0, 0, 10, 10]], [0.99]
    )
    ids = UpdateIds(
        pass_id=pass_id,
        action_id=action_id,
        kernel_id=factory.new(EntityKind.KERNEL),
        observation_id=factory.new(EntityKind.OBSERVATION),
        information_gain_id=factory.new(EntityKind.INFORMATION_GAIN),
        belief_update_id=factory.new(EntityKind.BELIEF_UPDATE),
        stopping_id=factory.new(EntityKind.STOPPING),
    )
    result = perform_update(
        belief,
        kernel,
        observation,
        ids=ids,
        stopping_threshold=0.8,
    )
    assert result.belief_update_record.posterior_after["fruit"] > 0.8
    assert result.stop_decision.should_stop
    assert result.stop_decision.declaration == "fruit"
    assert belief.status == "stopped"
    # Canonical round-trip is part of the contract.
    assert result.belief_update_record.from_json(
        result.belief_update_record.to_json()
    ) == result.belief_update_record


def test_expected_rig_matches_eig_by_enumerating_observations():
    posterior = np.array([0.2, 0.6, 0.2])
    kernel = _kernel({"fruit": 0.9, "leaf": 0.1, "background": 0.05})
    predictive = predictive_distribution(posterior, kernel)
    expected_kl = 0.0
    for index, probability in enumerate(predictive):
        likelihood = kernel.matrix[:, index]
        updated = posterior * likelihood
        updated = updated / updated.sum()
        expected_kl += probability * kl_divergence(updated, posterior)
    assert np.isclose(expected_kl, expected_information_gain(posterior, kernel))


def test_soft_count_is_sum_of_posteriors_with_variance():
    a = PatchBelief.from_mapping("node_a", {"fruit": 0.8, "leaf": 0.1, "background": 0.1})
    b = PatchBelief.from_mapping("node_b", {"fruit": 0.4, "leaf": 0.5, "background": 0.1})
    result = estimate_count([a, b], "fruit")
    assert np.isclose(result.soft_count, 1.2)
    assert result.hard_count == 1
    assert result.variance > 0.0
    assert result.unresolved_count == 2

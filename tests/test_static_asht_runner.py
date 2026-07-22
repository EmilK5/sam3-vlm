from __future__ import annotations

import numpy as np

from agent.asht.action_bank import StaticActionBank, StaticActionTemplate
from agent.asht.kernels import SensorProfile
from agent.asht.runner_static import StaticAshtConfig, StaticAshtRunner
from graph import OrchardGraph
from provenance.schema import ActionFamily, NodeStatus


class FakeBackend:
    @staticmethod
    def run_raw_inference(
        processor,
        image_np,
        threshold,
        *,
        prompt,
        pos_boxes,
        neg_boxes,
        disable_size_filter,
        return_masks,
    ):
        # The test node is [10,10,20,20]; its padded ROI starts at [7,7].
        boxes = np.array([[3.0, 3.0, 13.0, 13.0]])
        scores = np.array([0.99])
        if return_masks:
            return boxes, scores, [np.ones((10, 10), dtype=bool)]
        return boxes, scores

    @staticmethod
    def apply_nms_dualgate(boxes, scores, confidence, return_indices=False, **kwargs):
        indices = np.arange(len(scores), dtype=int)
        return boxes, scores, indices

    @staticmethod
    def apply_nms(boxes, scores, confidence, return_indices=False, **kwargs):
        indices = np.arange(len(scores), dtype=int)
        return boxes, scores, indices


def _bank():
    profile = SensorProfile(
        observation_labels=("not_found", "weak_match", "strong_match"),
        present=(0.01, 0.09, 0.90),
        absent=(0.90, 0.09, 0.01),
        source="runner-test",
    )
    return StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="fruit",
                family=ActionFamily.TARGET,
                prompt="round green fruit",
                semantic_key="round+green+fruit",
                beta_by_class={"fruit": 0.99, "leaf": 0.01, "background": 0.01},
                threshold=0.5,
            ),
        ),
        profile=profile,
    )


def _config(**overrides):
    values = dict(
        class_names=("fruit", "leaf", "background"),
        target_class="fruit",
        prior={"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3},
        action_bank=_bank(),
        stopping_threshold=0.90,
        max_total_queries=5,
        max_queries_per_node=2,
    )
    values.update(overrides)
    return StaticAshtConfig(**values)


def test_static_runner_selects_action_updates_belief_and_stops():
    graph = OrchardGraph()
    node_id = graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    runner = StaticAshtRunner(
        backend=FakeBackend,
        processor=None,
        image_np=np.zeros((40, 40, 3), dtype=np.uint8),
        graph=graph,
        config=_config(),
    )
    result = runner.run()

    node = result.graph.nodes[node_id]
    assert result.total_sam3_queries == 1
    assert len(result.passes) == 1
    assert node.belief.status == "stopped"
    assert node.belief.decision == "fruit"
    assert node.posterior["fruit"] > 0.90
    assert result.count.hard_count == 1
    assert result.count.soft_count > 0.90

    pass_record = result.passes[0]
    assert pass_record.candidate_action_set is not None
    assert pass_record.selected_action_id is not None
    assert pass_record.observation.label == "strong_match"
    assert pass_record.belief_update.posterior_after["fruit"] > 0.90
    assert pass_record.stopping_decision.should_stop
    assert pass_record.graph_after[0].status is NodeStatus.STOPPED
    assert len(node.verification_action_ids) == 1
    assert len(node.observation_ids) == 1
    assert len(node.belief_update_ids) == 1
    assert pass_record.from_json(pass_record.to_json()) == pass_record


def test_static_runner_marks_remaining_nodes_when_global_budget_ends():
    graph = OrchardGraph()
    first = graph.add_candidate([10, 10, 20, 20], 0.8, 1, node_id="node_budget_000001")
    second = graph.add_candidate([10, 10, 20, 20], 0.7, 1, node_id="node_budget_000002")
    result = StaticAshtRunner(
        backend=FakeBackend,
        processor=None,
        image_np=np.zeros((40, 40, 3), dtype=np.uint8),
        graph=graph,
        config=_config(max_total_queries=1),
    ).run()

    assert result.total_sam3_queries == 1
    assert result.graph.nodes[first].belief.resolved
    assert result.graph.nodes[second].belief.status == "budget_exhausted"
    assert any(
        pass_record.stopping_decision is not None
        and pass_record.stopping_decision.reason.value == "budget"
        for pass_record in result.passes
    )


def test_static_runner_stops_when_action_bank_is_exhausted():
    graph = OrchardGraph()
    node_id = graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    # Set an unreachable confidence threshold. After the one semantic action is
    # consumed, the next controller pass must stop with no_valid_action.
    result = StaticAshtRunner(
        backend=FakeBackend,
        processor=None,
        image_np=np.zeros((40, 40, 3), dtype=np.uint8),
        graph=graph,
        config=_config(stopping_threshold=0.999999, max_queries_per_node=4),
    ).run()

    node = result.graph.nodes[node_id]
    assert result.total_sam3_queries == 1
    assert node.belief.status == "stopped"
    assert node.belief.stop_reason == "no_valid_action"
    assert result.passes[-1].stopping_decision.reason.value == "no_valid_action"


def test_static_runner_bootstraps_empty_graph_then_verifies():
    from pipeline_stages import Sam3QuerySpec

    graph = OrchardGraph()
    config = _config(
        bootstrap_query=Sam3QuerySpec(
            prompt="green fruit",
            threshold=0.5,
            region=(0, 0, 40, 40),
        ),
        max_total_queries=3,
    )
    result = StaticAshtRunner(
        backend=FakeBackend,
        processor=None,
        image_np=np.zeros((40, 40, 3), dtype=np.uint8),
        graph=graph,
        config=config,
    ).run()

    assert result.total_sam3_queries == 2
    assert len(result.graph.nodes) == 1
    assert len(result.passes) == 2
    bootstrap = result.passes[0]
    assert bootstrap.target_node_id is None
    assert len(bootstrap.registrations) == 1
    node = next(iter(result.graph.nodes.values()))
    assert len(node.source_detection_ids) == 1
    assert node.belief.resolved

from __future__ import annotations

import json

import numpy as np

from agent.asht.action_bank import StaticActionBank, StaticActionTemplate
from agent.asht.kernels import SensorProfile
from agent.asht.qwen_generator import (
    QwenActionGeneratorConfig,
    QwenCandidateActionGenerator,
)
from agent.asht.runner_qwen import QwenAshtRunner
from agent.asht.runner_static import StaticAshtConfig
from graph import OrchardGraph
from provenance.ids import IdFactory, create_run_id
from provenance.schema import ActionFamily


class IdSource:
    def __init__(self):
        self.factory = IdFactory(create_run_id(random_token="qwen-test"))

    def new_id(self, kind):
        return self.factory.new(kind)


class FakeQwen:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


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
        boxes = np.array([[3.0, 3.0, 13.0, 13.0]])
        scores = np.array([0.99])
        return (boxes, scores, [np.ones((10, 10), bool)]) if return_masks else (boxes, scores)

    @staticmethod
    def apply_nms_dualgate(boxes, scores, confidence, return_indices=False, **kwargs):
        idx = np.arange(len(scores), dtype=int)
        return boxes, scores, idx

    @staticmethod
    def apply_nms(boxes, scores, confidence, return_indices=False, **kwargs):
        idx = np.arange(len(scores), dtype=int)
        return boxes, scores, idx


def _profile():
    return SensorProfile(
        observation_labels=("not_found", "weak_match", "strong_match"),
        present=(0.01, 0.09, 0.90),
        absent=(0.90, 0.09, 0.01),
        source="qwen-test",
    )


def _fallback():
    return StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="fallback-fruit",
                family=ActionFamily.TARGET,
                prompt="round fruit",
                semantic_key="fallback+round+fruit",
                beta_by_class={"fruit": 0.95, "leaf": 0.03, "background": 0.02},
            ),
        ),
        profile=_profile(),
    )


def _valid_response():
    return {
        "content": json.dumps(
            {
                "reasoning_summary": "Test a target-specific conjunction.",
                "actions": [
                    {
                        "family": "target",
                        "prompt": "round green citrus fruit",
                        "semantic_key": "round+green+citrus",
                        "beta_by_class": {
                            "fruit": 0.99,
                            "leaf": 0.01,
                            "background": 0.01,
                        },
                        "threshold": 0.5,
                        "rationale": "Distinguish round fruit from flat leaves.",
                    }
                ],
            }
        ),
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }


def test_qwen_generator_validates_and_materializes_actions():
    graph = OrchardGraph()
    node_id = graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    graph.initialize_beliefs({"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3})
    generator = QwenCandidateActionGenerator(
        client=FakeQwen(_valid_response()),
        profile=_profile(),
        fallback_bank=_fallback(),
    )
    ids = IdSource()
    pass_id = ids.new_id(__import__("provenance.ids", fromlist=["EntityKind"]).EntityKind.PASS)
    result = generator.generate(
        graph=graph,
        node=graph.nodes[node_id],
        class_names=("fruit", "leaf", "background"),
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        image_width=40,
        image_height=40,
        pass_id=pass_id,
        id_source=ids,
    )
    assert len(result.actions) == 1
    assert not result.candidate_set.fallback_used
    assert result.candidate_set.source_qwen_call_id == result.qwen_call.qwen_call_id
    assert result.qwen_call.input_tokens == 100
    assert result.qwen_call.reasoning_summary.startswith("Test")
    assert result.actions[0].record.generator == "qwen_candidate_generator"


def test_qwen_generator_uses_static_fallback_after_invalid_response():
    graph = OrchardGraph()
    node_id = graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    graph.initialize_beliefs({"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3})
    generator = QwenCandidateActionGenerator(
        client=FakeQwen({"content": "not-json"}),
        profile=_profile(),
        fallback_bank=_fallback(),
        config=QwenActionGeneratorConfig(max_retries=0),
    )
    ids = IdSource()
    from provenance.ids import EntityKind

    result = generator.generate(
        graph=graph,
        node=graph.nodes[node_id],
        class_names=("fruit", "leaf", "background"),
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        image_width=40,
        image_height=40,
        pass_id=ids.new_id(EntityKind.PASS),
        id_source=ids,
    )
    assert result.candidate_set.fallback_used
    assert result.qwen_call.fallback_used
    assert result.actions[0].record.generator == "static_action_bank"
    assert result.qwen_call.validation_errors


def test_qwen_runner_records_call_and_uses_numeric_selector():
    graph = OrchardGraph()
    node_id = graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    generator = QwenCandidateActionGenerator(
        client=FakeQwen(_valid_response()),
        profile=_profile(),
        fallback_bank=_fallback(),
    )
    config = StaticAshtConfig(
        class_names=("fruit", "leaf", "background"),
        target_class="fruit",
        prior={"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3},
        action_bank=_fallback(),
        stopping_threshold=0.90,
        max_total_queries=3,
        max_queries_per_node=2,
    )
    result = QwenAshtRunner(
        action_generator=generator,
        backend=FakeBackend,
        processor=None,
        image_np=np.zeros((40, 40, 3), dtype=np.uint8),
        graph=graph,
        config=config,
    ).run()
    assert result.graph.nodes[node_id].belief.decision == "fruit"
    assert len(result.passes) == 1
    pass_record = result.passes[0]
    assert len(pass_record.qwen_calls) == 1
    assert pass_record.candidate_action_set.generation_policy == "qwen_candidate_generator"
    assert pass_record.information_gain_records[0].selected
    assert pass_record.cost_after.qwen_calls == 1
    assert pass_record.cost_after.input_tokens == 100


def test_qwen_region_that_misses_target_is_rejected_and_falls_back():
    payload = {
        "content": json.dumps(
            {
                "actions": [
                    {
                        "family": "target",
                        "prompt": "fruit",
                        "semantic_key": "bad-region",
                        "region": [30, 30, 39, 39],
                        "beta_by_class": {
                            "fruit": 0.9,
                            "leaf": 0.1,
                            "background": 0.1,
                        },
                    }
                ]
            }
        )
    }
    graph = OrchardGraph()
    node_id = graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    graph.initialize_beliefs({"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3})
    generator = QwenCandidateActionGenerator(
        client=FakeQwen(payload),
        profile=_profile(),
        fallback_bank=_fallback(),
        config=QwenActionGeneratorConfig(max_retries=0),
    )
    ids = IdSource()
    from provenance.ids import EntityKind

    result = generator.generate(
        graph=graph,
        node=graph.nodes[node_id],
        class_names=("fruit", "leaf", "background"),
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        image_width=40,
        image_height=40,
        pass_id=ids.new_id(EntityKind.PASS),
        id_source=ids,
    )
    assert result.candidate_set.fallback_used
    assert result.candidate_set.rejected_action_payloads
    assert any("must intersect" in error for error in result.candidate_set.validation_errors)


def test_qwen_generated_agent_tiling_is_executed_and_recorded():
    response = {
        "content": json.dumps(
            {
                "reasoning_summary": "Zoom the ambiguous candidate.",
                "actions": [
                    {
                        "family": "target",
                        "prompt": "small round green citrus fruit",
                        "semantic_key": "small+round+green+citrus",
                        "beta_by_class": {
                            "fruit": 0.99,
                            "leaf": 0.01,
                            "background": 0.01,
                        },
                        "tiling_mode": "agent_controlled",
                        "tile_scale": 1.0,
                    }
                ],
            }
        )
    }
    graph = OrchardGraph()
    graph.add_candidate([10, 10, 20, 20], 0.8, 1)
    generator = QwenCandidateActionGenerator(
        client=FakeQwen(response),
        profile=_profile(),
        fallback_bank=_fallback(),
    )
    config = StaticAshtConfig(
        class_names=("fruit", "leaf", "background"),
        target_class="fruit",
        prior={"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3},
        action_bank=_fallback(),
        stopping_threshold=0.90,
        max_total_queries=3,
        max_queries_per_node=1,
    )
    result = QwenAshtRunner(
        action_generator=generator,
        backend=FakeBackend,
        processor=None,
        image_np=np.zeros((40, 40, 3), dtype=np.uint8),
        graph=graph,
        config=config,
    ).run()
    record = result.passes[0]
    assert record.tiling_decision is not None
    assert record.tiling_decision.triggered
    assert record.candidate_action_set.actions[0].tiling_mode.value == "agent_controlled"
    assert len(record.tiles) == 1
    assert record.cost_after.qwen_calls == 1
    assert record.cost_after.tile_calls == 1

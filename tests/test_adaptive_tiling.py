from __future__ import annotations

import numpy as np

from agent.asht.action_bank import StaticActionBank, StaticActionTemplate
from agent.asht.kernels import SensorProfile
from agent.asht.runner_static import StaticAshtConfig, StaticAshtRunner
from agent.asht.tiling import (
    AdaptiveTilingConfig,
    assess_density,
    calculate_tile_parameters,
    generate_tiles,
)
from graph import OrchardGraph
from pipeline_stages import Sam3QuerySpec
from provenance.schema import ActionFamily, TilingMode


class TilingBackend:
    calls = 0

    @classmethod
    def run_raw_inference(
        cls,
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
        cls.calls += 1
        h, w = image_np.shape[:2]
        box = np.array([[max(0.0, w * 0.25), max(0.0, h * 0.25), max(1.0, w * 0.55), max(1.0, h * 0.55)]])
        score = np.array([0.95])
        return (box, score, [np.ones((h, w), bool)]) if return_masks else (box, score)

    @staticmethod
    def apply_nms_dualgate(boxes, scores, confidence, return_indices=False, **kwargs):
        idx = np.arange(len(scores), dtype=int)
        return boxes, scores, idx

    @staticmethod
    def apply_nms(boxes, scores, confidence, return_indices=False, **kwargs):
        idx = np.arange(len(scores), dtype=int)
        return boxes, scores, idx


def _bank(tiling_mode=TilingMode.OFF):
    profile = SensorProfile(
        observation_labels=("not_found", "weak_match", "strong_match"),
        present=(0.01, 0.09, 0.90),
        absent=(0.90, 0.09, 0.01),
        source="tiling-test",
    )
    return StaticActionBank(
        templates=(
            StaticActionTemplate(
                name="fruit",
                family=ActionFamily.TARGET,
                prompt="round green fruit",
                semantic_key="round+green+fruit",
                beta_by_class={"fruit": 0.99, "leaf": 0.01, "background": 0.01},
                tiling_mode=tiling_mode,
            ),
        ),
        profile=profile,
    )


def test_sam3count_density_rules_and_tile_parameters():
    boxes = np.array([[i, 0, i + 1, 1] for i in range(91)], dtype=float)
    assessment = assess_density(boxes, image_width=200, image_height=200)
    assert assessment.use_tiling
    assert assessment.tiling_rule == "LARGE"
    tile_size, overlap, stride, rule = calculate_tile_parameters(1200, 800, "MEDIUM")
    assert rule.name == "MEDIUM"
    assert tile_size == 400
    assert overlap == 120
    assert stride == 280


def test_roi_guided_tile_generation_is_deterministic():
    specs = generate_tiles(
        source_region=(100, 200, 500, 600),
        source_width=400,
        source_height=400,
        tile_size=200,
        overlap=50,
        roi_global=(250, 350, 400, 500),
    )
    assert specs
    assert [spec.index for spec in specs] == list(range(len(specs)))
    assert all(spec.global_box[0] >= 100 for spec in specs)
    assert all(spec.global_box[1] >= 200 for spec in specs)


def test_static_runner_bootstrap_records_adaptive_tiles_and_budget():
    TilingBackend.calls = 0
    graph = OrchardGraph()
    config = StaticAshtConfig(
        class_names=("fruit", "leaf", "background"),
        target_class="fruit",
        prior={"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3},
        action_bank=_bank(),
        stopping_threshold=0.90,
        max_total_queries=4,
        max_queries_per_node=1,
        bootstrap_query=Sam3QuerySpec(
            prompt="green fruit",
            threshold=0.5,
            region=(0, 0, 200, 200),
        ),
        bootstrap_tiling_mode=TilingMode.ALWAYS,
        adaptive_tiling=AdaptiveTilingConfig(max_tiles_per_action=3),
    )
    result = StaticAshtRunner(
        backend=TilingBackend,
        processor=None,
        image_np=np.zeros((200, 200, 3), dtype=np.uint8),
        graph=graph,
        config=config,
    ).run()
    bootstrap = result.passes[0]
    assert bootstrap.tiling_decision is not None
    assert bootstrap.tiling_decision.triggered
    assert len(bootstrap.tiles) == 3
    assert bootstrap.cost_after.tile_calls == 3
    assert bootstrap.cost_after.sam3_calls == 4
    assert result.total_sam3_queries == 4
    assert len(bootstrap.raw_detections) == 4


def test_static_verification_action_can_request_tiled_sensing():
    TilingBackend.calls = 0
    graph = OrchardGraph()
    node_id = graph.add_candidate([40, 40, 80, 80], 0.8, 1)
    config = StaticAshtConfig(
        class_names=("fruit", "leaf", "background"),
        target_class="fruit",
        prior={"fruit": 1 / 3, "leaf": 1 / 3, "background": 1 / 3},
        action_bank=_bank(TilingMode.ALWAYS),
        stopping_threshold=0.90,
        max_total_queries=4,
        max_queries_per_node=1,
        adaptive_tiling=AdaptiveTilingConfig(max_tiles_per_action=2),
    )
    result = StaticAshtRunner(
        backend=TilingBackend,
        processor=None,
        image_np=np.zeros((120, 120, 3), dtype=np.uint8),
        graph=graph,
        config=config,
    ).run()
    record = result.passes[0]
    assert record.target_node_id == node_id
    assert record.tiling_decision is not None
    assert record.tiling_decision.triggered
    assert len(record.tiles) >= 1
    assert record.cost_after.tile_calls == len(record.tiles)
    assert record.from_json(record.to_json()) == record

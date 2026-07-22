from __future__ import annotations

import numpy as np

from graph import OrchardGraph
from pipeline_stages import DetectionBatch, StageContext, register_detection_batch
from provenance.ids import EntityKind, IdFactory, create_run_id


class FakeSink:
    def __init__(self):
        self.factory = IdFactory(create_run_id(random_token="graph"))
        self.records = []

    def new_id(self, kind):
        return self.factory.new(kind)

    def append(self, payload, **kwargs):
        self.records.append(payload)


def test_graph_round_trip_preserves_belief_lineage_and_mask():
    graph = OrchardGraph()
    node_id = graph.add_candidate(
        [1, 2, 11, 12],
        0.8,
        1,
        node_id="node_roundtrip_000001",
        source_detection_id="det_roundtrip_000001",
    )
    node = graph.nodes[node_id]
    node.mask = np.array([[False, True], [True, True]], dtype=bool)
    belief = node.initialize_belief(
        {"fruit": 0.2, "leaf": 0.7, "background": 0.1}
    )
    belief.apply_likelihood([0.9, 0.1, 0.1], evidence_id="observation_roundtrip_000001")
    belief.mark_stopped(reason="confidence")
    node.record_verification(
        action_id="action_roundtrip_000001",
        semantic_key="round+green",
        observation_id="observation_roundtrip_000001",
        belief_update_id="belief_roundtrip_000001",
    )
    node.dedup_decision_ids.append("dedup_roundtrip_000001")
    node.registration_ids.append("registration_roundtrip_000001")

    payload = graph.to_dict()
    restored = OrchardGraph.from_dict(payload)

    assert restored.to_dict() == payload
    restored_node = restored.nodes[node_id]
    assert np.array_equal(restored_node.mask, node.mask)
    assert restored_node.belief.as_mapping() == node.belief.as_mapping()
    assert restored_node.temporary_map_class == "fruit"
    assert restored_node.final_declaration == "fruit"
    assert restored_node.classification == "unresolved"


def test_graph_loads_legacy_node_without_inventing_belief():
    legacy = {
        "nodes": [
            {
                "id": "node_legacy_000001",
                "box": [0, 0, 10, 10],
                "found_in_pass": 1,
                "scores": {
                    "detection_confidence": 0.9,
                    "fruit_verification": 0.8,
                    "leaf_verification": 0.1,
                },
                "classification": "fruit",
                "support": 1,
                "jitter": 0.0,
                "area": 100.0,
                "signatures": ["fruit"],
            }
        ]
    }
    graph = OrchardGraph.from_dict(legacy)
    node = graph.nodes["node_legacy_000001"]
    assert node.belief is None
    assert node.classification == "fruit"
    positive, negative = graph.get_exemplars()
    assert positive.shape == (1, 4)
    assert negative.shape == (0, 4)


def test_probabilistic_exemplars_require_stopped_high_confidence_nodes():
    graph = OrchardGraph()
    node_id = graph.add_candidate([0, 0, 10, 10], 0.9, 1)
    node = graph.nodes[node_id]
    belief = node.initialize_belief(
        {"fruit": 0.95, "leaf": 0.04, "background": 0.01}
    )

    positive, _ = graph.get_exemplars()
    assert positive.shape == (0, 4)

    belief.mark_stopped(reason="confidence", declaration="fruit")
    positive, _ = graph.get_exemplars(positive_threshold=0.90)
    assert positive.shape == (1, 4)


def test_registration_allocates_run_scoped_node_and_records_lineage():
    sink = FakeSink()
    pass_id = sink.new_id(EntityKind.PASS)
    context = StageContext(pass_id=pass_id, model_id="sam3", sink=sink)
    detection_id = sink.new_id(EntityKind.RAW_DETECTION)
    call_id = sink.new_id(EntityKind.SAM3_CALL)
    batch = DetectionBatch(
        boxes_local=np.array([[0, 0, 10, 10]], dtype=float),
        boxes_global=np.array([[0, 0, 10, 10]], dtype=float),
        scores=np.array([0.9]),
        detection_ids=(detection_id,),
        prompt="fruit",
        sam3_call_id=call_id,
    )
    graph = OrchardGraph()
    result = register_detection_batch(
        batch,
        graph,
        pass_number=1,
        context=context,
    )
    node = graph.nodes[result.created_node_ids[0]]
    assert node.id.startswith("node_graph_")
    assert node.source_detection_ids == [detection_id]
    assert len(node.registration_ids) == 1

    second_detection_id = sink.new_id(EntityKind.RAW_DETECTION)
    second = DetectionBatch(
        boxes_local=np.array([[0.1, 0.1, 10.1, 10.1]], dtype=float),
        boxes_global=np.array([[0.1, 0.1, 10.1, 10.1]], dtype=float),
        scores=np.array([0.8]),
        detection_ids=(second_detection_id,),
        prompt="fruit",
        sam3_call_id=call_id,
    )
    register_detection_batch(second, graph, pass_number=2, context=context)
    assert second_detection_id in node.source_detection_ids
    assert node.found_in_passes == [1, 2]
    assert len(node.dedup_decision_ids) == 1
    assert len(node.registration_ids) == 2

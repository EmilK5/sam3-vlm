from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from graph import OrchardGraph
from pipeline_stages import (
    DetectionBatch,
    Sam3QuerySpec,
    StageContext,
    StageError,
    execute_sam3_request,
    normalize_sam3_output,
    register_detection_batch,
    suppress_detection_batch,
)
from provenance.ids import EntityKind, IdFactory, create_run_id
from provenance.schema import CoordinateSpace


class FakeSink:
    def __init__(self):
        self.factory = IdFactory(create_run_id(random_token="stages"))
        self.records = []

    def new_id(self, kind):
        return self.factory.new(kind)

    def append(self, payload, **kwargs):
        self.records.append(payload)
        return payload


class FakeBackend:
    @staticmethod
    def run_raw_inference(
        processor,
        image_np,
        confidence,
        prompt,
        pos_boxes=None,
        neg_boxes=None,
        disable_size_filter=False,
        return_masks=False,
    ):
        boxes = np.array([[1.0, 2.0, 11.0, 12.0], [2.0, 2.0, 12.0, 12.0]])
        scores = np.array([0.9, 0.7])
        if return_masks:
            return boxes, scores, [np.ones((20, 20), bool), np.ones((20, 20), bool)]
        return boxes, scores

    @staticmethod
    def apply_nms_dualgate(boxes, scores, confidence, return_indices=False, **kwargs):
        idx = np.array([0], dtype=int)
        return boxes[idx], scores[idx], idx

    @staticmethod
    def apply_nms(boxes, scores, confidence, return_indices=False, **kwargs):
        idx = np.array([0], dtype=int)
        return boxes[idx], scores[idx], idx


def _context():
    sink = FakeSink()
    pass_id = sink.new_id(EntityKind.PASS)
    return sink, StageContext(pass_id=pass_id, model_id="sam3-test", sink=sink)


def test_normalize_rejects_misaligned_outputs():
    query = Sam3QuerySpec("fruit", 0.5, (0, 0, 100, 100))
    try:
        normalize_sam3_output([[0, 0, 1, 1]], [0.5, 0.6], query=query)
    except StageError:
        pass
    else:
        raise AssertionError("misaligned SAM3 output should fail")


def test_execute_sam3_request_translates_roi_and_emits_records():
    sink, context = _context()
    query = Sam3QuerySpec(
        prompt="green fruit",
        threshold=0.5,
        region=(100, 200, 300, 400),
        coordinate_space=CoordinateSpace.ROI_LOCAL,
    )
    batch = execute_sam3_request(
        FakeBackend,
        processor=None,
        image_np=np.zeros((200, 200, 3), dtype=np.uint8),
        query=query,
        context=context,
    )
    assert batch.boxes_global[0].tolist() == [101.0, 202.0, 111.0, 212.0]
    assert len(batch.detection_ids) == 2
    record_types = [record.RECORD_TYPE for record in sink.records]
    assert record_types.count("sam3_call") == 1
    assert record_types.count("raw_detection") == 2


def test_suppression_preserves_lineage_and_emits_decision():
    sink, context = _context()
    call_id = sink.new_id(EntityKind.SAM3_CALL)
    ids = tuple(sink.new_id(EntityKind.RAW_DETECTION) for _ in range(2))
    batch = DetectionBatch(
        boxes_local=np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float),
        boxes_global=np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float),
        scores=np.array([0.9, 0.8]),
        detection_ids=ids,
        prompt="fruit",
        sam3_call_id=call_id,
    )
    result = suppress_detection_batch(
        batch,
        FakeBackend,
        confidence=0.5,
        context=context,
    )
    assert result.kept.detection_ids == (ids[0],)
    assert result.removed_detection_ids == (ids[1],)
    assert len(result.comparisons) == 1
    assert result.comparisons[0].selected_survivor_id == ids[0]


def test_registration_is_separate_and_records_create_then_update():
    sink, context = _context()
    graph = OrchardGraph()
    call_id = sink.new_id(EntityKind.SAM3_CALL)
    first_id = sink.new_id(EntityKind.RAW_DETECTION)
    batch = DetectionBatch(
        boxes_local=np.array([[0, 0, 10, 10]], float),
        boxes_global=np.array([[0, 0, 10, 10]], float),
        scores=np.array([0.9]),
        detection_ids=(first_id,),
        prompt="fruit",
        sam3_call_id=call_id,
    )
    first = register_detection_batch(batch, graph, pass_number=1, context=context)
    assert len(first.created_node_ids) == 1
    node = graph.nodes[first.created_node_ids[0]]
    assert node.classification == "unresolved"

    second_id = sink.new_id(EntityKind.RAW_DETECTION)
    second_batch = DetectionBatch(
        boxes_local=np.array([[0.2, 0.2, 10.2, 10.2]], float),
        boxes_global=np.array([[0.2, 0.2, 10.2, 10.2]], float),
        scores=np.array([0.8]),
        detection_ids=(second_id,),
        prompt="fruit",
        sam3_call_id=call_id,
    )
    second = register_detection_batch(second_batch, graph, pass_number=2, context=context)
    assert second.updated_node_ids == (node.id,)
    assert len(graph.nodes) == 1
    assert node.support == 2
    assert second.rejected_detection_ids == (second_id,)


def test_pipeline_staged_entry_point_keeps_verification_out(monkeypatch):
    import importlib
    import sys
    import types

    inf = types.ModuleType("inference")
    inf.run_raw_inference = FakeBackend.run_raw_inference
    inf.apply_nms_dualgate = FakeBackend.apply_nms_dualgate
    inf.apply_nms = FakeBackend.apply_nms
    inf.apply_clahe = lambda image: image
    verifier_package = types.ModuleType("verifier")
    verifier_package.__path__ = []
    verify_module = types.ModuleType("verifier.verify")
    verifier_package.verify = verify_module

    monkeypatch.setitem(sys.modules, "inference", inf)
    monkeypatch.setitem(sys.modules, "verifier", verifier_package)
    monkeypatch.setitem(sys.modules, "verifier.verify", verify_module)
    sys.modules.pop("pipeline", None)
    pipeline = importlib.import_module("pipeline")

    graph = OrchardGraph()
    query = Sam3QuerySpec("fruit", 0.5, (0, 0, 50, 50))
    result = pipeline.execute_staged_pass(
        processor=None,
        image_np=np.zeros((50, 50, 3), dtype=np.uint8),
        graph=graph,
        query=query,
        pass_number=1,
    )
    assert len(result["raw_batch"]) == 2
    assert len(result["suppression"].kept) == 1
    assert len(result["registration"].created_node_ids) == 1
    node = graph.nodes[result["registration"].created_node_ids[0]]
    assert node.classification == "unresolved"

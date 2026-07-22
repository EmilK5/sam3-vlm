from datetime import datetime, timezone
from dataclasses import fields

import pytest

from provenance import (
    SCHEMA_VERSION,
    ActionFamily,
    ActionKind,
    ArtifactKind,
    ContractError,
    CoordinateSpace,
    CostSnapshotRecord,
    DatasetSampleRecord,
    EnvironmentRecord,
    EventEnvelope,
    EventKind,
    GraphNodeSnapshotRecord,
    IdFactory,
    ModelIdentityRecord,
    NodeStatus,
    PassRecord,
    RepositoryStateRecord,
    RunRecord,
    RunStatus,
    SensingActionRecord,
    TilingMode,
    create_run_id,
    record_from_dict,
    registered_record_types,
)
from provenance.ids import EntityKind


def make_fixture_records():
    run_id = create_run_id(
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        random_token="fixture",
    )
    ids = IdFactory(run_id)
    artifact = __import__("provenance").ArtifactRef(
        artifact_id=ids.new(EntityKind.ARTIFACT),
        kind=ArtifactKind.IMAGE,
        relative_path="artifacts/input.png",
        media_type="image/png",
        sha256="abc123",
        size_bytes=42,
        width=640,
        height=480,
    )
    sample = DatasetSampleRecord(
        image_id=ids.new(EntityKind.IMAGE),
        dataset_name="CARPK",
        dataset_version="official",
        split="test",
        image_artifact=artifact,
        image_sha256="abc123",
        width=640,
        height=480,
        target_concept="car",
        target_class="car",
        confounder_classes=("background",),
        ground_truth_count=1,
        ground_truth_boxes=((1.0, 2.0, 30.0, 40.0),),
    )
    pass_id = ids.new(EntityKind.PASS)
    node = GraphNodeSnapshotRecord(
        graph_node_id=ids.new(EntityKind.GRAPH_NODE),
        pass_id=pass_id,
        box=(1.0, 2.0, 30.0, 40.0),
        status=NodeStatus.UNRESOLVED,
        source_detection_ids=(),
        found_in_passes=(1,),
        prior={"car": 0.5, "background": 0.5},
        posterior={"car": 0.8, "background": 0.2},
        temporary_map_class="car",
        final_declaration=None,
        query_count=1,
        exemplar_eligible_positive=False,
        exemplar_eligible_negative=False,
    )
    action = SensingActionRecord(
        action_id=ids.new(EntityKind.ACTION),
        action_kind=ActionKind.VERIFY,
        family=ActionFamily.TARGET,
        prompt="parked car",
        region=(0.0, 0.0, 640.0, 480.0),
        threshold=0.5,
        semantic_key="parked+car",
        target_node_id=node.graph_node_id,
        tiling_mode=TilingMode.OFF,
        beta_by_class={"car": 0.9, "background": 0.1},
    )
    candidate_set = __import__("provenance").CandidateActionSetRecord(
        candidate_set_id=ids.new(EntityKind.CANDIDATE),
        pass_id=pass_id,
        target_node_id=node.graph_node_id,
        actions=(action,),
    )
    cost = CostSnapshotRecord(
        pass_id=pass_id,
        sam3_calls=1,
        qwen_calls=0,
        tile_calls=0,
        verify_calls=1,
        orchestration_calls=1,
        input_tokens=0,
        output_tokens=0,
        runtime_seconds=0.25,
        normalized_cost=1.5,
    )
    pass_record = PassRecord(
        pass_id=pass_id,
        run_id=run_id,
        pass_index=1,
        started_at="2026-07-22T00:00:00+00:00",
        completed_at="2026-07-22T00:00:01+00:00",
        target_node_id=node.graph_node_id,
        graph_before=(node,),
        candidate_action_set=candidate_set,
        information_gain_records=(),
        selected_action_id=action.action_id,
        qwen_calls=(),
        sam3_calls=(),
        tiles=(),
        raw_detections=(),
        dedup_comparisons=(),
        registrations=(),
        observation=None,
        kernel=None,
        belief_update=None,
        stopping_decision=None,
        graph_after=(node,),
        cost_after=cost,
        continuation_reason="node remains unresolved",
    )
    run = RunRecord(
        run_id=run_id,
        status=RunStatus.SUCCEEDED,
        created_at="2026-07-22T00:00:00+00:00",
        started_at="2026-07-22T00:00:00+00:00",
        completed_at="2026-07-22T00:00:01+00:00",
        repository=RepositoryStateRecord(
            commit="deadbeef",
            branch="phase-1-contracts",
            dirty=False,
        ),
        environment=EnvironmentRecord(
            python_version="3.11",
            platform="linux",
            hostname="test-host",
        ),
        resolved_config={"conf": 0.5},
        config_sha256="config-hash",
        random_seeds={"python": 0},
        dataset_sample=sample,
        models=(
            ModelIdentityRecord(
                model_id="sam3-model",
                role="sensor",
                name="SAM3",
                provider="Meta",
            ),
        ),
        id_counters=ids.snapshot(),
        passes=(pass_record,),
        final_graph=(node,),
        final_predictions={"hard_count": 1, "soft_count": 0.8},
        evaluations=(),
        artifacts=(artifact,),
        errors=(),
    )
    return ids, run, action


def test_nested_run_round_trips_through_strict_json():
    _, run, _ = make_fixture_records()
    text = run.to_json()
    loaded = RunRecord.from_json(text)

    assert loaded == run
    assert loaded.dataset_sample.image_artifact.kind is ArtifactKind.IMAGE
    assert loaded.passes[0].candidate_action_set.actions[0].family is ActionFamily.TARGET


def test_record_type_and_schema_version_are_embedded():
    _, run, _ = make_fixture_records()
    payload = run.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["record_type"] == "run"
    assert payload["passes"][0]["record_type"] == "pass"


def test_registry_dispatches_nested_record_type():
    _, _, action = make_fixture_records()
    loaded = record_from_dict(action.to_dict())
    assert loaded == action
    assert "sensing_action" in registered_record_types()


def test_event_envelope_round_trips_polymorphic_payload():
    ids, _, action = make_fixture_records()
    event = EventEnvelope(
        event_id=ids.new(EntityKind.EVENT),
        run_id=ids.run_id,
        sequence_number=1,
        event_kind=EventKind.ACTION,
        payload=action,
    )
    loaded = EventEnvelope.from_json(event.to_json())
    assert loaded == event
    assert isinstance(loaded.payload, SensingActionRecord)


def test_invalid_probability_vector_is_rejected():
    run_id = create_run_id(random_token="invalid")
    ids = IdFactory(run_id)
    with pytest.raises(ContractError, match="sum to 1"):
        GraphNodeSnapshotRecord(
            graph_node_id=ids.new(EntityKind.GRAPH_NODE),
            pass_id=ids.new(EntityKind.PASS),
            box=(0.0, 0.0, 1.0, 1.0),
            status=NodeStatus.UNRESOLVED,
            source_detection_ids=(),
            found_in_passes=(1,),
            prior={"fruit": 0.8, "leaf": 0.8},
            posterior={"fruit": 0.5, "leaf": 0.5},
            temporary_map_class=None,
            final_declaration=None,
            query_count=0,
            exemplar_eligible_positive=False,
            exemplar_eligible_negative=False,
        )


def test_invalid_box_is_rejected():
    run_id = create_run_id(random_token="box")
    ids = IdFactory(run_id)
    with pytest.raises(ContractError, match="x2 >= x1"):
        SensingActionRecord(
            action_id=ids.new(EntityKind.ACTION),
            action_kind=ActionKind.QUERY,
            family=ActionFamily.SYSTEM,
            prompt="car",
            region=(10.0, 10.0, 5.0, 5.0),
            threshold=0.5,
            semantic_key="car",
        )


def test_non_finite_metadata_is_rejected_during_json_serialization():
    _, run, _ = make_fixture_records()
    bad = RunRecord(
        **{
            **{field.name: getattr(run, field.name) for field in fields(run)},
            "metadata": {"bad": float("nan")},
        }
    )
    with pytest.raises(ContractError, match="Non-finite"):
        bad.to_json()

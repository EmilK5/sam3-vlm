from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from provenance import (
    ArtifactKind,
    ArtifactRef,
    CoordinateSpace,
    CostSnapshotRecord,
    DatasetSampleRecord,
    EnvironmentRecord,
    ErrorEventRecord,
    EvaluationResultRecord,
    EventKind,
    EventLogReader,
    EventLogWriter,
    GraphNodeSnapshotRecord,
    IdFactory,
    JsonLineCorruptionError,
    ModelIdentityRecord,
    NodeStatus,
    PassRecord,
    ReportingLevel,
    RepositoryStateRecord,
    RunRecord,
    RunStatus,
    RunStore,
    SensingActionRecord,
    ActionKind,
    ActionFamily,
    TilingMode,
    StorageError,
    create_run_id,
)
from provenance.ids import EntityKind
from provenance.io import repair_truncated_jsonl_tail, sha256_file


def make_initial_run(token: str = "storage") -> tuple[IdFactory, RunRecord]:
    run_id = create_run_id(
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        random_token=token,
    )
    ids = IdFactory(run_id)
    image_artifact = ArtifactRef(
        artifact_id=ids.new(EntityKind.ARTIFACT),
        kind=ArtifactKind.IMAGE,
        relative_path="artifacts/input.png",
        media_type="image/png",
        sha256="input-sha",
        size_bytes=10,
        width=640,
        height=480,
    )
    sample = DatasetSampleRecord(
        image_id=ids.new(EntityKind.IMAGE),
        dataset_name="CARPK",
        dataset_version="official",
        split="test",
        image_artifact=image_artifact,
        image_sha256="input-sha",
        width=640,
        height=480,
        target_concept="car",
        target_class="car",
        confounder_classes=("background",),
        ground_truth_count=1,
        ground_truth_boxes=((1.0, 2.0, 30.0, 40.0),),
    )
    run = RunRecord(
        run_id=run_id,
        status=RunStatus.CREATED,
        created_at="2026-07-22T00:00:00+00:00",
        started_at=None,
        completed_at=None,
        repository=RepositoryStateRecord(
            commit="deadbeef",
            branch="phase-2-storage",
            dirty=False,
        ),
        environment=EnvironmentRecord(
            python_version="3.13",
            platform="linux",
            hostname="test-host",
        ),
        resolved_config={"confidence": 0.5},
        config_sha256="config-sha",
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
        passes=(),
        final_graph=(),
        final_predictions={},
        evaluations=(),
        artifacts=(image_artifact,),
        errors=(),
    )
    return ids, run


def make_graph_and_pass(store: RunStore) -> tuple[GraphNodeSnapshotRecord, PassRecord]:
    pass_id = store.new_id(EntityKind.PASS)
    node = GraphNodeSnapshotRecord(
        graph_node_id=store.new_id(EntityKind.GRAPH_NODE),
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
    cost = CostSnapshotRecord(
        pass_id=pass_id,
        sam3_calls=1,
        qwen_calls=0,
        tile_calls=0,
        verify_calls=1,
        orchestration_calls=1,
        input_tokens=0,
        output_tokens=0,
        runtime_seconds=0.2,
        normalized_cost=1.0,
    )
    pass_record = PassRecord(
        pass_id=pass_id,
        run_id=store.run_id,
        pass_index=1,
        started_at="2026-07-22T00:00:00+00:00",
        completed_at="2026-07-22T00:00:01+00:00",
        target_node_id=node.graph_node_id,
        graph_before=(),
        candidate_action_set=None,
        information_gain_records=(),
        selected_action_id=None,
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
        continuation_reason="continue",
    )
    return node, pass_record


def test_event_log_appends_strict_sequential_envelopes(tmp_path: Path):
    ids, run = make_initial_run("events")
    path = tmp_path / "events.jsonl"
    writer = EventLogWriter(path, run.run_id, ids, fsync=False)

    first = writer.append(run.dataset_sample.image_artifact)
    second = writer.append(run, event_kind=EventKind.RUN)

    assert first.sequence_number == 1
    assert second.sequence_number == 2
    loaded = EventLogReader(path).read_all()
    assert loaded == (first, second)
    assert all(line.strip().startswith("{") for line in path.read_text().splitlines())


def test_event_writer_rejects_payload_from_another_run(tmp_path: Path):
    ids, run = make_initial_run("owner")
    _, other = make_initial_run("other")
    writer = EventLogWriter(tmp_path / "events.jsonl", run.run_id, ids, fsync=False)
    with pytest.raises(Exception, match="does not match"):
        writer.append(other, event_kind=EventKind.RUN)


def test_truncated_final_event_is_repaired_and_sequence_resumes(tmp_path: Path):
    ids, run = make_initial_run("repair")
    path = tmp_path / "events.jsonl"
    writer = EventLogWriter(path, run.run_id, ids, fsync=False)
    first = writer.append(run.dataset_sample.image_artifact)
    with path.open("ab") as handle:
        handle.write(b'{"schema_version": "1.0.0"')

    assert repair_truncated_jsonl_tail(path) is True
    restored = EventLogWriter.restore(path, run.run_id, ids, fsync=False)
    second = restored.append(run, event_kind=EventKind.RUN)

    assert first.sequence_number == 1
    assert second.sequence_number == 2
    assert len(EventLogReader(path).read_all()) == 2


def test_non_tail_corruption_is_never_silently_repaired(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"bad":\n{"also": "bad"}\n')
    with pytest.raises(JsonLineCorruptionError):
        repair_truncated_jsonl_tail(path)


def test_run_store_creates_self_contained_layout_and_checkpoint(tmp_path: Path):
    _, initial = make_initial_run("layout")
    store = RunStore.create(tmp_path, initial, fsync=False)

    assert store.paths.manifest.exists()
    assert store.paths.events.exists() or store.paths.events.parent.exists()
    assert store.paths.config.exists()
    assert store.paths.environment.exists()
    assert store.paths.partial_run.exists()
    assert store.paths.latest_checkpoint.exists()
    assert store.paths.summary.exists()
    assert store.paths.artifacts.is_dir()

    partial = RunRecord.from_json(store.paths.partial_run.read_text())
    assert partial.status is RunStatus.RUNNING
    assert partial.run_id == initial.run_id
    assert partial.metadata["reporting_level"] == "standard"


def test_checkpoint_and_finalization_consolidate_pass_graph_evaluation_and_artifact(
    tmp_path: Path,
):
    _, initial = make_initial_run("consolidate")
    store = RunStore.create(tmp_path, initial, fsync=False)
    node, pass_record = make_graph_and_pass(store)
    store.append(pass_record)

    evaluation = EvaluationResultRecord(
        evaluation_id=store.new_id(EntityKind.EVALUATION),
        run_id=store.run_id,
        image_id=initial.dataset_sample.image_id,
        evaluator_name="count",
        evaluator_version="1",
        metrics={"mae": 0.0},
        counts={"prediction": 1, "ground_truth": 1},
    )
    store.append(evaluation)
    artifact = store.write_artifact_bytes(
        b"mask-bytes",
        kind=ArtifactKind.MASK,
        relative_path="masks/node-1.bin",
        minimum_level=ReportingLevel.STANDARD,
    )

    checkpoint = store.checkpoint(final_predictions={"soft_count": 0.8})
    assert checkpoint.passes == (
        dataclasses.replace(pass_record, graph_before=(), graph_after=()),
    )
    assert checkpoint.final_graph == (node,)
    assert checkpoint.evaluations == (evaluation,)
    assert artifact in checkpoint.artifacts
    assert checkpoint.events == ()

    final = store.finalize_success(final_predictions={"hard_count": 1, "soft_count": 0.8})
    loaded = RunRecord.from_json(store.paths.final_run.read_text())
    assert loaded == final
    assert loaded.status is RunStatus.SUCCEEDED
    assert loaded.metadata["event_count"] == 3
    assert loaded.metadata["last_event_sequence"] == 3
    assert loaded.events == ()
    assert tuple(
        event.sequence_number
        for event in EventLogReader(store.paths.events).read_all()
    ) == (1, 2, 3)
    assert json.loads(store.paths.summary.read_text())["status"] == "succeeded"


def test_artifact_file_is_copied_hashed_and_recorded(tmp_path: Path):
    _, initial = make_initial_run("artifact")
    store = RunStore.create(tmp_path, initial, fsync=False)
    source = tmp_path / "source.txt"
    source.write_text("complete provenance")

    artifact = store.add_artifact_file(
        source,
        kind=ArtifactKind.LOG,
        relative_path="logs/source.txt",
        media_type="text/plain",
    )
    saved = store.paths.root / artifact.relative_path

    assert saved.read_text() == "complete provenance"
    assert artifact.sha256 == sha256_file(saved)
    assert artifact.size_bytes == saved.stat().st_size
    assert EventLogReader(store.paths.events).read_all()[0].payload == artifact


def test_store_reopens_with_restored_ids_and_repairs_partial_tail(tmp_path: Path):
    _, initial = make_initial_run("resume")
    store = RunStore.create(tmp_path, initial, fsync=False)
    first_id = store.new_id(EntityKind.ACTION)
    store.checkpoint()
    with store.paths.events.open("ab") as handle:
        handle.write(b'{"partial":')

    reopened = RunStore.open(store.paths.root, fsync=False)
    second_id = reopened.new_id(EntityKind.ACTION)

    assert first_id.endswith("000001")
    assert second_id.endswith("000002")
    assert EventLogReader(reopened.paths.events).read_all() == ()



def test_reopen_recovers_entity_counters_from_post_checkpoint_events(tmp_path: Path):
    _, initial = make_initial_run("event-counters")
    store = RunStore.create(tmp_path, initial, fsync=False)
    store.checkpoint()
    action = SensingActionRecord(
        action_id=store.new_id(EntityKind.ACTION),
        action_kind=ActionKind.VERIFY,
        family=ActionFamily.TARGET,
        prompt="parked car",
        region=(0.0, 0.0, 640.0, 480.0),
        threshold=0.5,
        semantic_key="parked+car",
        tiling_mode=TilingMode.OFF,
        beta_by_class={"car": 0.9, "background": 0.1},
    )
    store.append(action)

    reopened = RunStore.open(store.paths.root, fsync=False)
    next_action_id = reopened.new_id(EntityKind.ACTION)

    assert action.action_id.endswith("000001")
    assert next_action_id.endswith("000002")


def test_failure_finalization_preserves_exception_and_partial_results(tmp_path: Path):
    _, initial = make_initial_run("failure")
    store = RunStore.create(tmp_path, initial, fsync=False)
    _, pass_record = make_graph_and_pass(store)
    store.append(pass_record)

    try:
        raise RuntimeError("synthetic failure")
    except RuntimeError as exc:
        failed = store.finalize_failure(
            exc,
            component="sam3",
            final_predictions={"partial_count": 1},
        )

    assert failed.status is RunStatus.FAILED
    assert failed.passes == (
        dataclasses.replace(pass_record, graph_before=(), graph_after=()),
    )
    assert failed.final_predictions == {"partial_count": 1}
    assert len(failed.errors) == 1
    assert failed.errors[0].exception_type == "RuntimeError"
    assert "synthetic failure" in (failed.errors[0].traceback or "")
    assert store.paths.final_run.exists()
    with pytest.raises(StorageError):
        store.append(pass_record)


def test_reporting_level_can_omit_full_only_events(tmp_path: Path):
    _, initial = make_initial_run("minimal")
    store = RunStore.create(
        tmp_path,
        initial,
        reporting_level=ReportingLevel.MINIMAL,
        fsync=False,
    )
    result = store.append(
        initial.dataset_sample.image_artifact,
        minimum_level=ReportingLevel.FULL,
    )
    assert result is None
    assert EventLogReader(store.paths.events).read_all() == ()


def test_materialize_declared_input_artifact(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"declared input")
    _, run = make_initial_run("declared")
    from dataclasses import replace
    from provenance.io import sha256_file

    declared = replace(
        run.dataset_sample.image_artifact,
        relative_path="artifacts/input/source.png",
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
    )
    sample = replace(run.dataset_sample, image_artifact=declared, image_sha256=declared.sha256)
    run = replace(run, dataset_sample=sample, artifacts=(declared,))
    store = RunStore.create(tmp_path / "declared", run, fsync=False)
    destination = store.materialize_declared_artifact(declared, source)
    assert destination.read_bytes() == b"declared input"

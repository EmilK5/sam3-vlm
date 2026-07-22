from datetime import datetime, timezone

import pytest

from provenance import ContractError
from provenance.ids import EntityKind, IdFactory, create_run_id, validate_entity_id


def test_run_id_is_deterministic_when_clock_and_token_are_injected():
    run_id = create_run_id(
        now=datetime(2026, 7, 22, 13, 5, 9, 123456, tzinfo=timezone.utc),
        random_token="abc123",
    )
    assert run_id == "run_20260722T130509123456Z_abc123"
    validate_entity_id(run_id, expected_kind=EntityKind.RUN)


def test_factory_generates_namespaced_monotonic_ids():
    run_id = create_run_id(
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        random_token="unit",
    )
    factory = IdFactory(run_id)

    assert factory.new(EntityKind.PASS) == "pass_unit_000001"
    assert factory.new(EntityKind.PASS) == "pass_unit_000002"
    assert factory.new(EntityKind.ACTION) == "action_unit_000001"
    assert factory.snapshot() == {"action": 1, "pass": 2}


def test_factory_can_resume_from_counter_snapshot():
    run_id = create_run_id(
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
        random_token="resume",
    )
    restored = IdFactory.restore(run_id, {"sam3": 4})
    assert restored.new(EntityKind.SAM3_CALL) == "sam3_resume_000005"


def test_run_kind_cannot_be_created_as_child_id():
    run_id = create_run_id(random_token="child")
    with pytest.raises(ContractError):
        IdFactory(run_id).new(EntityKind.RUN)


def test_wrong_prefix_is_rejected():
    with pytest.raises(ContractError):
        validate_entity_id("action_unit_000001", expected_kind=EntityKind.PASS)

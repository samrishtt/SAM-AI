"""Unit tests for CallTelemetry schema and serialization."""

import json
from sam_ai.agent.telemetry import CallTelemetry, get_package_versions, get_git_commit_hash


def test_telemetry_schema_fields():
    """Verify that CallTelemetry contains all required fields with expected types."""
    tel = CallTelemetry.create(
        prompt_tokens=150,
        completion_tokens=420,
        model_text="<think>deliberating</think>{\"action\": \"ACTION1\"}",
        parse_valid=True,
        latency_ms=123.456,
        seed=42,
        prompt_ends_with_think=False,
        think_prefix_added=True,
        decode={"think_budget": 3072, "action_text": "{\"action\": \"ACTION1\"}", "think_ended_by": "close_tag"},
    )

    data = tel.to_dict()
    assert isinstance(data["prompt_tokens"], int)
    assert data["prompt_tokens"] == 150
    assert isinstance(data["completion_tokens"], int)
    assert data["completion_tokens"] == 420
    assert isinstance(data["has_think_close"], bool)
    assert data["has_think_close"] is True
    assert isinstance(data["parse_valid"], bool)
    assert data["parse_valid"] is True
    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] == 123.46
    assert data["seed"] == 42
    assert isinstance(data["repo_commit_hash"], str)
    assert isinstance(data["package_versions"], dict)
    assert data["prompt_ends_with_think"] is False
    assert data["think_prefix_added"] is True
    assert isinstance(data["decode"], dict)
    assert data["decode"]["think_budget"] == 3072

    # Verify JSON serializability
    serialized = json.dumps(data)
    loaded = json.loads(serialized)
    assert loaded["prompt_tokens"] == 150
    assert loaded["decode"]["think_budget"] == 3072


def test_telemetry_think_closing_tag_detection():
    """Verify that has_think_close accurately reflects the presence of </think>."""
    tel_with = CallTelemetry.create(
        prompt_tokens=50,
        completion_tokens=100,
        model_text="some thoughts </think> action",
        parse_valid=True,
        latency_ms=10.0,
    )
    assert tel_with.has_think_close is True

    tel_without = CallTelemetry.create(
        prompt_tokens=50,
        completion_tokens=100,
        model_text="some thoughts without closing tag action",
        parse_valid=False,
        latency_ms=10.0,
    )
    assert tel_without.has_think_close is False


def test_package_versions_schema():
    """Verify package versions introspection dictionary structure."""
    versions = get_package_versions()
    assert isinstance(versions, dict)
    for pkg in ["torch", "transformers", "arc_agi", "arcengine", "peft"]:
        assert pkg in versions
        assert isinstance(versions[pkg], str)

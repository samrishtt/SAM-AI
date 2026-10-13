"""Tests that all 4,103 grammar strings parse and validate into GameAction with valid coordinates."""

import pytest
from sam_ai.agent.think_commit import ActionGrammar
from sam_ai.agent.schema import ActionParser

try:
    from arcengine import GameAction
except ImportError:
    try:
        from arcengine.enums import GameAction
    except ImportError:
        GameAction = None


def test_action_grammar_coverage_all_4103_strings():
    grammar = ActionGrammar.arc3(
        click_template='{"action": "%s", "data": {"x": %d, "y": %d}}'
    )
    all_strings = grammar.strings()
    assert len(all_strings) == 4103, f"Expected 4103 strings, got {len(all_strings)}"

    for s in all_strings:
        parsed = ActionParser.parse_and_validate(s)
        assert parsed.is_valid, f"Grammar string failed to parse: {s} -> {parsed.validation_error}"
        if GameAction is not None:
            assert parsed.action in GameAction.__members__, f"Action {parsed.action} not in GameAction"
        if parsed.action == "ACTION6":
            assert "x" in parsed.data and "y" in parsed.data
            assert 0 <= parsed.data["x"] <= 63
            assert 0 <= parsed.data["y"] <= 63

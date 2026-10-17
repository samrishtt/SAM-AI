"""Unit tests for thinking-prefix rendering and telemetry flags."""

import jinja2
import pytest
from sam_ai.substrate.models import HuggingFaceModelRunner
from sam_ai.agent.telemetry import CallTelemetry


class StubTokenizerWithThink:
    """Chat template that already appends <think>\\n to generation prompt."""
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "assistant: <think>\n"
        "{% endif %}"
    )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        template = jinja2.Template(self.chat_template)
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


class StubTokenizerWithoutThink:
    """Chat template that does not append <think>\\n to generation prompt."""
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "assistant:\n"
        "{% endif %}"
    )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        template = jinja2.Template(self.chat_template)
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


class DummyRunner:
    """Lightweight dummy subclass that tests render_prompt without loading PyTorch weights."""
    render_prompt = HuggingFaceModelRunner.render_prompt

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_fixture_with_think_prefix():
    """When chat template ends with <think>\\n, no prefix is added."""
    runner = DummyRunner(StubTokenizerWithThink())
    formatted, ends_with, added = runner.render_prompt(
        prompt="Solve ARC puzzle",
        system_prompt="Instructions here"
    )

    assert ends_with is True
    assert added is False
    assert formatted.endswith("<think>\n")
    assert not formatted.endswith("<think>\n<think>\n")
    # Instructions must be in user turn, not a separate system role
    assert "system:" not in formatted
    assert "user: Instructions here\n\nSolve ARC puzzle" in formatted


def test_fixture_without_think_prefix():
    """When chat template does not end with <think>\\n, <think>\\n is appended once."""
    runner = DummyRunner(StubTokenizerWithoutThink())
    formatted, ends_with, added = runner.render_prompt(
        prompt="Solve ARC puzzle",
        system_prompt="Instructions here"
    )

    assert ends_with is False
    assert added is True
    assert formatted.endswith("<think>\n")
    assert not formatted.endswith("<think>\n<think>\n")
    # Instructions must be in user turn
    assert "system:" not in formatted
    assert "user: Instructions here\n\nSolve ARC puzzle" in formatted


def test_telemetry_captures_think_flags():
    """Verify that CallTelemetry retains prompt_ends_with_think and think_prefix_added."""
    tel = CallTelemetry.create(
        prompt_tokens=100,
        completion_tokens=50,
        model_text="<think>plan</think>{\"action\":\"ACTION1\"}",
        parse_valid=True,
        latency_ms=45.0,
        prompt_ends_with_think=False,
        think_prefix_added=True,
    )
    d = tel.to_dict()
    assert d["prompt_ends_with_think"] is False
    assert d["think_prefix_added"] is True

"""
Tests for SAM-AI Python ReasoningStreamFilter
=============================================
Validates:
1. Complete suppression of <think>...</think> reasoning traces from streamed output.
2. Safe buffering across split delimiter boundaries (e.g. '</' + 'think>').
3. Multi-split delimiter boundaries.
4. Truncation handling via finalize() - honest fail-closed notice without leakage.
5. Passthrough of direct answers without think tags.
6. Canonical system prompt content and integrity.
"""

import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPACE_DIR))

from model_loader import ReasoningStreamFilter, filter_reasoning_from_text, SAM_AI_SYSTEM_PROMPT


def test_standard_thinking_filter():
    f = ReasoningStreamFilter()
    chunks = [
        "<think>\n",
        "The user is asking 2 + 2.\n",
        "Let us compute 2 + 2 = 4.\n",
        "</think>\n\n",
        "2 + 2 = 4.",
    ]
    emitted = []
    for c in chunks:
        out = f.process_chunk(c)
        if out:
            emitted.append(out)
    final = f.finalize()
    if final:
        emitted.append(final)

    result = "".join(emitted)
    assert "<think>" not in result
    assert "</think>" not in result
    assert "The user is asking" not in result
    assert "Let us compute" not in result
    assert "2 + 2 = 4." in result
    assert f.has_seen_closing is True


def test_split_delimiter_boundary():
    f = ReasoningStreamFilter()
    chunks = [
        "<think>Let me ponder this question deeply",
        ". The answer is 42.",
        "</",
        "think>\n\n",
        "The answer is 42.",
    ]
    emitted = []
    for c in chunks:
        out = f.process_chunk(c)
        if out:
            emitted.append(out)
    final = f.finalize()
    if final:
        emitted.append(final)

    result = "".join(emitted)
    assert "<think>" not in result
    assert "</think>" not in result
    assert "Let me ponder" not in result
    assert result == "The answer is 42."
    assert f.has_seen_closing is True


def test_multi_split_delimiter():
    f = ReasoningStreamFilter()
    chunks = [
        "<think>Internal rationale",
        "<",
        "/",
        "th",
        "ink",
        ">",
        "Final verified output.",
    ]
    emitted = []
    for c in chunks:
        out = f.process_chunk(c)
        if out:
            emitted.append(out)
    final = f.finalize()
    if final:
        emitted.append(final)

    result = "".join(emitted)
    assert "<think>" not in result
    assert "</think>" not in result
    assert "Internal rationale" not in result
    assert "Final verified output." in result


def test_truncated_thinking_fails_closed():
    f = ReasoningStreamFilter()
    chunks = [
        "<think>\n",
        "Step 1: Compute intermediate values...\n",
        "Step 2: Still thinking...",
    ]
    emitted = []
    for c in chunks:
        out = f.process_chunk(c)
        if out:
            emitted.append(out)
    final = f.finalize()
    if final:
        emitted.append(final)

    result = "".join(emitted)
    assert "<think>" not in result
    assert "</think>" not in result
    assert "intermediate values" not in result
    assert "Still thinking" not in result
    assert "interrupted" in result.lower()
    assert f.has_seen_closing is False


def test_direct_answer_passthrough():
    raw_direct = "Hello, I am SAM-AI. How can I help you today?"
    cleaned = filter_reasoning_from_text(raw_direct)
    assert cleaned == raw_direct

    raw_with_think = "<think>\nInternal reasoning.\n</think>\n\nHere is the answer."
    cleaned_think = filter_reasoning_from_text(raw_with_think)
    assert cleaned_think == "Here is the answer."
    assert "<think>" not in cleaned_think
    assert "</think>" not in cleaned_think


def test_canonical_system_prompt():
    assert "You are SAM-AI" in SAM_AI_SYSTEM_PROMPT
    assert "Parallax Intelligence Lab" in SAM_AI_SYSTEM_PROMPT
    assert "DeepSeek-R1-Distill-Qwen-7B" in SAM_AI_SYSTEM_PROMPT
    assert "Core Operating Guidelines:" in SAM_AI_SYSTEM_PROMPT

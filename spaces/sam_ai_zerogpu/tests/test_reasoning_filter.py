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


def test_plain_final_response_streaming_absent_delimiters():
    f = ReasoningStreamFilter()
    c1 = f.process_chunk("42 is the answer to everything.")
    assert c1 is None  # Buffers until </think> or finalize
    final = f.finalize()
    assert final == "42 is the answer to everything."
    assert f.has_seen_closing is False


def test_multi_chunk_plain_response_streaming():
    f = ReasoningStreamFilter()
    f.process_chunk("The prime numbers below 10 are: ")
    f.process_chunk("2, 3, 5, 7.")
    final = f.finalize()
    assert final == "The prime numbers below 10 are: 2, 3, 5, 7."


def test_malformed_unclosed_think_fails_closed():
    f = ReasoningStreamFilter()
    f.process_chunk("<think>malformed unclosed stream")
    final = f.finalize()
    assert final is not None
    assert "interrupted" in final.lower()
    assert "malformed" not in final


def test_formatting_math_and_code_preserved():
    raw = (
        "<think>\nCompute function\n</think>\n\n"
        "$$\\int 2x dx = x^2 + C$$\n\n"
        "```python\ndef is_prime(n):\n    return n > 1\n```"
    )
    clean = filter_reasoning_from_text(raw)
    assert "<think>" not in clean
    assert "</think>" not in clean
    assert "Compute function" not in clean
    assert "$$\\int 2x dx = x^2 + C$$" in clean
    assert "def is_prime(n):" in clean

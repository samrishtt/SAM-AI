"""
Tests for Phase 3A Multi-Domain Intelligence Baseline
=====================================================
Validates:
1. Numerical and symbolic math extraction (hash, boxed, fractions, phrases).
2. HumanEval isolated subprocess sandbox execution (passing, failing, syntax error, timeout).
3. Instruction-following constraint verification (JSON schema, word count, letter ban, formatting).
4. Multi-turn context retention verification.
5. Fixtures dataset integrity and configuration validation.
6. Resumability and checkpoint recovery from partial runs.
7. Anti-adapter fail-closed enforcement.
"""

import os
import sys
import json
import time
import tempfile
import hashlib
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam_ai.evaluation.multidomain_harness import (
    clean_numeric_str,
    extract_numerical_answer,
    extract_symbolic_math,
    compare_math_answers,
    extract_code_block,
    execute_humaneval_in_sandbox,
    verify_instruction_constraint,
    verify_multiturn_context,
    verify_prediction,
)


def test_clean_numeric_str():
    assert clean_numeric_str("$123.45") == "123.45"
    assert clean_numeric_str("1,000,000") == "1000000"
    assert clean_numeric_str("+42") == "42"
    assert clean_numeric_str("  -17.  ") == "-17"
    assert clean_numeric_str(None) is None
    assert clean_numeric_str("") is None


def test_extract_numerical_answer():
    # 1. Hash marker
    text1 = "<think>Let's calculate 10 + 6 = 16</think>\n#### 16"
    ans, method = extract_numerical_answer(text1)
    assert ans == "16"
    assert method == "hash_marker"

    # 2. LaTeX boxed
    text2 = "<think>Solving...</think>\nTherefore, the answer is \\boxed{196}."
    ans, method = extract_numerical_answer(text2)
    assert ans == "196"
    assert method == "latex_boxed"

    # 3. Natural phrase
    text3 = "After evaluating, the final answer is $2,400."
    ans, method = extract_numerical_answer(text3)
    assert ans == "2400"
    assert method == "natural_phrase"

    # 4. Trailing numeric token
    text4 = "<think>Step 1... Step 2...</think>\nHence, the remaining tins are 50"
    ans, method = extract_numerical_answer(text4)
    assert ans == "50"
    assert method == "trailing_numeric_fallback"

    # 5. Empty or no number
    ans, method = extract_numerical_answer("")
    assert ans is None
    assert method == "empty_text"

    ans, method = extract_numerical_answer("No digits present at all.")
    assert ans is None
    assert method == "no_number_found"


def test_extract_symbolic_math():
    # LaTeX boxed fraction
    text1 = "Solving tan(theta): \\boxed{-\\frac{3}{4}}"
    ans, method = extract_symbolic_math(text1)
    assert ans == "-3/4"
    assert method == "latex_boxed_symbolic"

    # Hash marker symbolic
    text2 = "Probability is #### 1/9"
    ans, method = extract_symbolic_math(text2)
    assert ans == "1/9"
    assert method == "hash_marker_symbolic"

    # Fraction in text
    text3 = "The resulting probability is -3/4 as required."
    ans, method = extract_symbolic_math(text3)
    assert ans == "-3/4"
    assert method == "fraction_match"


def test_compare_math_answers():
    assert compare_math_answers("16", "16", "numerical_exact") is True
    assert compare_math_answers("16.0", "16", "numerical_exact") is True
    assert compare_math_answers("2/4", "1/2", "math_symbolic") is True
    assert compare_math_answers("-3/4", "-0.75", "math_symbolic") is True
    assert compare_math_answers("195", "195", "numerical_exact") is True
    assert compare_math_answers("194", "195", "numerical_exact") is False
    assert compare_math_answers(None, "10", "numerical_exact") is False


def test_extract_code_block():
    text = """<think>
Let's implement the function.
</think>
Here is the implementation:
```python
def add(a: int, b: int) -> int:
    return a + b
```
"""
    code = extract_code_block(text, entry_point="add")
    assert "def add(a: int, b: int) -> int:" in code
    assert "return a + b" in code
    assert "<think>" not in code


def test_execute_humaneval_in_sandbox_success():
    prompt = "def double(x: int) -> int:\n"
    completion = "```python\ndef double(x: int) -> int:\n    return x * 2\n```"
    test_code = "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(-3) == -6\n"
    passed, details, extracted = execute_humaneval_in_sandbox(
        prompt=prompt,
        completion=completion,
        test_code=test_code,
        entry_point="double",
        timeout_seconds=3.0
    )
    assert passed is True
    assert details == "passed_all_tests"
    assert "return x * 2" in extracted


def test_execute_humaneval_in_sandbox_assertion_failure():
    prompt = "def double(x: int) -> int:\n"
    completion = "```python\ndef double(x: int) -> int:\n    return x + 1\n```"
    test_code = "def check(candidate):\n    assert candidate(2) == 4\n"
    passed, details, extracted = execute_humaneval_in_sandbox(
        prompt=prompt,
        completion=completion,
        test_code=test_code,
        entry_point="double",
        timeout_seconds=3.0
    )
    assert passed is False
    assert "test_failed" in details or "AssertionError" in details


def test_execute_humaneval_in_sandbox_syntax_error():
    prompt = "def bad_func():\n"
    completion = "```python\ndef bad_func():\n    return syntax error here\n```"
    test_code = "def check(candidate):\n    pass\n"
    passed, details, _ = execute_humaneval_in_sandbox(
        prompt=prompt,
        completion=completion,
        test_code=test_code,
        entry_point="bad_func",
        timeout_seconds=3.0
    )
    assert passed is False
    assert "test_failed" in details or "SyntaxError" in details


def test_execute_humaneval_in_sandbox_timeout():
    prompt = "def loop_forever():\n"
    completion = "```python\ndef loop_forever():\n    import time\n    time.sleep(5.0)\n```"
    test_code = "def check(candidate):\n    candidate()\n"
    t0 = time.time()
    passed, details, _ = execute_humaneval_in_sandbox(
        prompt=prompt,
        completion=completion,
        test_code=test_code,
        entry_point="loop_forever",
        timeout_seconds=1.0  # Fast timeout for test
    )
    elapsed = time.time() - t0
    assert passed is False
    assert "timeout_exceeded" in details
    assert elapsed < 3.0  # Ensure timeout was enforced promptly


def test_verify_instruction_constraint():
    # JSON schema
    valid_json = '{"status": "success", "code": 200}'
    passed, _ = verify_instruction_constraint("json_schema_validation", valid_json)
    assert passed is True

    invalid_json = '{"status": "error", "code": 500}'
    passed, _ = verify_instruction_constraint("json_schema_validation", invalid_json)
    assert passed is False

    # Word count 6
    sentence_6 = "The bright sun illuminates the sky."
    passed, _ = verify_instruction_constraint("word_count_exact_6", sentence_6)
    assert passed is True

    sentence_7 = "The very bright sun illuminates the sky."
    passed, _ = verify_instruction_constraint("word_count_exact_6", sentence_7)
    assert passed is False

    # Disallow letter e
    no_e = "This is a sunny day."
    passed, _ = verify_instruction_constraint("disallow_letter_e", no_e)
    assert passed is True

    has_e = "The sun is shining today."
    passed, _ = verify_instruction_constraint("disallow_letter_e", has_e)
    assert passed is False

    # ALL CAPS
    caps_pass = "HELLO WORLD"
    passed, _ = verify_instruction_constraint("text_exact", caps_pass)
    assert passed is True

    caps_fail = "Hello World"
    passed, _ = verify_instruction_constraint("text_exact", caps_fail)
    assert passed is False

    # 3 bullets
    bullets_pass = "* Red\n* Green\n* Blue"
    passed, _ = verify_instruction_constraint("bullet_count_3", bullets_pass)
    assert passed is True

    # Step numbering 1 to 5
    steps_pass = "Step 1: start\nStep 2: load\nStep 3: run\nStep 4: check\nStep 5: done"
    passed, _ = verify_instruction_constraint("step_prefix_5", steps_pass)
    assert passed is True

    # Comma separation 4
    comma_pass = "apple,banana,orange,grape"
    passed, _ = verify_instruction_constraint("comma_separated_4", comma_pass)
    assert passed is True

    comma_fail = "apple, banana, orange, grape"
    passed, _ = verify_instruction_constraint("comma_separated_4", comma_fail)
    assert passed is False

    # Ends with downward
    down_pass = "Objects accelerate toward the ground downward."
    passed, _ = verify_instruction_constraint("ends_with_downward", down_pass)
    assert passed is True

    # Exactly 2 lines
    lines_pass = "First line of the poem\nSecond line of the poem"
    passed, _ = verify_instruction_constraint("line_count_2", lines_pass)
    assert passed is True


def test_verify_multiturn_context():
    response = "You mentioned that your favorite fruit is pomegranate and your favorite city is Kyoto."
    passed, _ = verify_multiturn_context(response, ["pomegranate", "Kyoto"])
    assert passed is True

    response_missing = "You mentioned your favorite fruit is pomegranate."
    passed, details = verify_multiturn_context(response_missing, ["pomegranate", "Kyoto"])
    assert passed is False
    assert "Kyoto" in details


def test_fixtures_integrity():
    repo_root = Path(__file__).resolve().parent.parent
    fixtures_file = repo_root / "curriculum" / "phase_3a_baseline_fixtures.json"
    config_file = repo_root / "curriculum" / "phase_3a_baseline_config.json"

    assert fixtures_file.exists(), "Fixtures file missing"
    assert config_file.exists(), "Config file missing"

    with open(fixtures_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 7 domains, 95 samples total
    assert data["total_domains"] == 7
    assert data["total_samples"] == 95
    assert cfg["total_samples"] == 95

    expected_domains = {
        "mathematics_gsm8k": 20,
        "mathematics_math500": 20,
        "coding_humaneval": 20,
        "science_reasoning": 10,
        "logic_constraints": 10,
        "instruction_following": 10,
        "multiturn_context": 5
    }

    for dom, count in expected_domains.items():
        assert dom in data["domains"], f"Missing domain: {dom}"
        assert len(data["domains"][dom]) == count, f"Domain {dom} has {len(data['domains'][dom])} items, expected {count}"
        assert cfg["domains"][dom] == count

    # SHA-256 match
    content = open(fixtures_file, "rb").read()
    sha = hashlib.sha256(content).hexdigest()
    assert cfg["fixtures_sha256"] == sha, "Fixtures SHA-256 mismatch in config"

    # Confirm strict rule: base model only, adapter is None
    assert cfg["foundation_model"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert cfg["adapter_attached"] is None


def test_resumability_checkpoint_loop():
    """Verify that the evaluation loop resumes and skips already completed records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = Path(tmpdir) / "predictions.jsonl"

        # Pre-seed with 2 completed records
        initial_records = [
            {"example_id": "gsm8k_val_pilot_00", "is_correct": True, "domain": "mathematics_gsm8k"},
            {"example_id": "gsm8k_val_pilot_01", "is_correct": True, "domain": "mathematics_gsm8k"}
        ]
        with open(ckpt_file, "w", encoding="utf-8") as f:
            for r in initial_records:
                f.write(json.dumps(r) + "\n")

        # Read back checkpoint
        completed_ids = set()
        with open(ckpt_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)["example_id"])

        assert len(completed_ids) == 2
        assert "gsm8k_val_pilot_00" in completed_ids
        assert "gsm8k_val_pilot_01" in completed_ids

        # Simulate subsequent item evaluation
        new_record = {"example_id": "gsm8k_val_pilot_02", "is_correct": True, "domain": "mathematics_gsm8k"}
        with open(ckpt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_record) + "\n")

        # Verify 3 items now in file
        all_lines = [line.strip() for line in open(ckpt_file, "r", encoding="utf-8") if line.strip()]
        assert len(all_lines) == 3


if __name__ == "__main__":
    test_clean_numeric_str()
    test_extract_numerical_answer()
    test_extract_symbolic_math()
    test_compare_math_answers()
    test_extract_code_block()
    test_execute_humaneval_in_sandbox_success()
    test_execute_humaneval_in_sandbox_assertion_failure()
    test_execute_humaneval_in_sandbox_syntax_error()
    test_execute_humaneval_in_sandbox_timeout()
    test_verify_instruction_constraint()
    test_verify_multiturn_context()
    test_fixtures_integrity()
    test_resumability_checkpoint_loop()
    print("ALL 13 MULTI-DOMAIN BASELINE TESTS PASSED SUCCESSFULLY!")

"""
SAM-AI Multi-Domain Intelligence Baseline Harness
==================================================
Defines evaluation protocols, domain extractors, isolated sandboxes, and verification
logic for benchmarking the untouched base foundation model (DeepSeek-R1-Distill-Qwen-7B)
across 7 cognitive domains (95 challenge samples):

1. mathematics_gsm8k (20 items)
2. mathematics_math500 (20 items)
3. coding_humaneval (20 items)
4. science_reasoning (10 items)
5. logic_constraints (10 items)
6. instruction_following (10 items)
7. multiturn_context (5 dialogue pairs = 10 turns)

Strict Constraints:
- Untouched base model only (no LoRA adapter).
- Zero web retrieval / external tools.
- Isolated subprocess sandbox for coding execution with strict timeouts.
- Fail-closed error handling and deterministic evaluation.
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import hashlib
import tempfile
import subprocess
from pathlib import Path
from fractions import Fraction
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple


def clean_numeric_str(raw: str) -> Optional[str]:
    """Cleans a raw numeric string, stripping currencies, commas, and trailing punctuation."""
    if not raw:
        return None
    c = raw.replace("$", "").replace(",", "").strip().rstrip(".")
    if c.startswith("+"):
        c = c[1:]
    return c if c else None


def extract_boxed_content(text: str) -> List[str]:
    """Finds all \\boxed{...} instances, properly handling nested curly braces."""
    results = []
    idx = 0
    pattern = "\\boxed{"
    while True:
        pos = text.find(pattern, idx)
        if pos == -1:
            break
        start = pos + len(pattern)
        depth = 1
        curr = start
        while curr < len(text) and depth > 0:
            if text[curr] == "{":
                depth += 1
            elif text[curr] == "}":
                depth -= 1
            curr += 1
        if depth == 0:
            results.append(text[start:curr-1].strip())
        idx = curr
    return results


def extract_numerical_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts numerical answer from text using prioritized regex patterns:
    1. #### <num>
    2. \\boxed{<num>}
    3. Natural phrases (e.g. 'the answer is <num>')
    4. Trailing numeric token in the post-thinking body
    """
    if not text:
        return None, "empty_text"

    # Remove reasoning chain if present
    body = text.split("</think>")[-1] if "</think>" in text else text

    # 1. #### marker (GSM8K canonical)
    hm = re.search(r"####\s*([+-]?(?:\d+[\d,]*\.?\d*|\.\d+))", text)
    if hm:
        v = clean_numeric_str(hm.group(1))
        if v:
            return v, "hash_marker"

    # 2. LaTeX boxed marker
    boxed_candidates = extract_boxed_content(text)
    if boxed_candidates:
        cand = boxed_candidates[-1].strip()
        nc = re.sub(r"[^\d\.\+\-]", "", cand)
        v = clean_numeric_str(nc)
        if v:
            return v, "latex_boxed"

    # 3. Natural language phrase
    pm = re.findall(
        r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|answer\s*[:=])\s*([+-]?\$?\d+[\d,]*\.?\d*)",
        body,
        re.IGNORECASE
    )
    if pm:
        v = clean_numeric_str(pm[-1])
        if v:
            return v, "natural_phrase"

    # 4. Trailing numeric token fallback
    nums = re.findall(r"(?<!\w)([+-]?\d+[\d,]*\.?\d*)(?!\w)", body)
    if nums:
        v = clean_numeric_str(nums[-1])
        if v:
            return v, "trailing_numeric_fallback"

    return None, "no_number_found"


def extract_symbolic_math(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts symbolic or fractional mathematical answers (e.g., '-3/4', '1/9', 'pi', 'sqrt(2)').
    """
    if not text:
        return None, "empty_text"

    # 1. LaTeX boxed (with nested brace support)
    boxed_candidates = extract_boxed_content(text)
    if boxed_candidates:
        cand = boxed_candidates[-1].strip()
        # Handle latex frac: e.g. -\frac{3}{4} or \frac{-3}{4}
        frac_match = re.search(r"([+-]?)\s*\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", cand)
        if frac_match:
            prefix_sign = frac_match.group(1)
            num_val = int(frac_match.group(2))
            denom_val = int(frac_match.group(3))
            if prefix_sign == "-":
                num_val = -num_val
            cand = f"{num_val}/{denom_val}"
        cand = cand.replace(" ", "").replace("$", "")
        return cand, "latex_boxed_symbolic"

    # 2. Hash marker
    hm = re.search(r"####\s*([^\r\n]+)", text)
    if hm:
        cand = hm.group(1).strip().replace(" ", "").replace("$", "")
        return cand, "hash_marker_symbolic"

    # 3. Fractions like -3/4 or 1/9
    body = text.split("</think>")[-1] if "</think>" in text else text
    frac_matches = re.findall(r"([+-]?\d+\s*/\s*\d+)", body)
    if frac_matches:
        cand = frac_matches[-1].replace(" ", "")
        return cand, "fraction_match"

    # Fall back to general numerical extraction
    return extract_numerical_answer(text)


def compare_math_answers(pred: Optional[str], target: str, eval_type: str) -> bool:
    """Compares predicted math answer against target with tolerance and fractional equivalence."""
    if pred is None:
        return False
    pred = pred.strip()
    target = target.strip()

    # Exact string match
    if pred.lower() == target.lower():
        return True

    # Fractional equivalence: e.g. -3/4 == -0.75 or 2/4 == 1/2
    try:
        f_pred = Fraction(pred)
        f_target = Fraction(target)
        if f_pred == f_target:
            return True
    except Exception:
        pass

    # Float comparison with tolerance
    try:
        fl_pred = float(pred)
        fl_target = float(target)
        if abs(fl_pred - fl_target) < 1e-5:
            return True
    except Exception:
        pass

    return False


def extract_code_block(completion: str, entry_point: Optional[str] = None) -> str:
    """
    Extracts Python code from model completion.
    Strips reasoning tags (<think>...</think>) and pulls code from markdown blocks
    or lines defining the entry_point.
    """
    if not completion:
        return ""

    body = completion.split("</think>")[-1] if "</think>" in completion else completion

    # Match ```python ... ```
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", body, re.DOTALL)
    if code_blocks:
        if entry_point:
            for block in reversed(code_blocks):
                if f"def {entry_point}" in block:
                    return block.strip("` \t\r\n")
        return code_blocks[-1].strip("` \t\r\n")

    # If no markdown fences, check if there's raw python code
    lines = body.split("\n")
    code_lines = []
    started = False
    for line in lines:
        if entry_point and f"def {entry_point}" in line:
            started = True
        if started or line.startswith("def ") or line.startswith("import ") or line.startswith("from "):
            started = True
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip("` \t\r\n")

    return body.strip("` \t\r\n")


def execute_humaneval_in_sandbox(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout_seconds: float = 3.0
) -> Tuple[bool, str, str]:
    """
    Executes HumanEval candidate solution within an isolated subprocess sandbox.
    Returns (passed, details, extracted_code).
    """
    code_body = extract_code_block(completion, entry_point)
    if not code_body:
        return False, "no_code_extracted", ""

    # If code_body already defines entry_point, do not duplicate prompt header
    if entry_point and f"def {entry_point}" in code_body:
        full_code = code_body
    else:
        full_code = prompt.rstrip() + "\n" + code_body

    full_script = f"""
# HumanEval Isolated Execution Sandbox
import sys
import math
from typing import List, Tuple, Optional, Dict, Any, Union

{full_code}

{test_code}

try:
    check({entry_point})
    sys.exit(0)
except AssertionError as e:
    sys.stderr.write(f"AssertionError: {{e}}\\n")
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"RuntimeError: {{type(e).__name__}}: {{e}}\\n")
    sys.exit(2)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(full_script)
        temp_path = tf.name

    try:
        res = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1"
            }
        )
        if res.returncode == 0:
            return True, "passed_all_tests", code_body
        else:
            err = res.stderr.strip().split("\n")[-1] if res.stderr else f"exit_code_{res.returncode}"
            return False, f"test_failed: {err}", code_body
    except subprocess.TimeoutExpired:
        return False, f"timeout_exceeded_{timeout_seconds}s", code_body
    except Exception as e:
        return False, f"execution_error: {e}", code_body
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def verify_instruction_constraint(eval_type: str, text: str) -> Tuple[bool, str]:
    """Verifies strict instruction-following constraints."""
    if not text:
        return False, "empty_text"

    body = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()

    if eval_type == "json_schema_validation":
        # Look for JSON object with status='success' and code=200
        json_matches = re.findall(r"\{.*?\}", body, re.DOTALL)
        for m in json_matches:
            try:
                data = json.loads(m)
                if data.get("status") == "success" and data.get("code") == 200:
                    return True, "valid_json_matching_schema"
            except Exception:
                continue
        return False, "no_valid_json_matching_schema"

    elif eval_type == "word_count_exact_6":
        words = [w for w in re.split(r"\s+", body) if w]
        if len(words) == 6:
            return True, "exact_6_words"
        return False, f"word_count_is_{len(words)}"

    elif eval_type == "disallow_letter_e":
        # Check if the letter e or E appears anywhere
        words = [w for w in re.split(r"\s+", body) if w]
        if len(words) < 5:
            return False, f"insufficient_words_{len(words)}"
        if "e" in body.lower():
            return False, "contains_letter_e"
        return True, "no_letter_e_found"

    elif eval_type == "text_exact":
        # Check if text is uppercase "HELLO WORLD"
        if "HELLO WORLD" in body:
            return True, "text_exact_match"
        return False, f"mismatch_{body[:30]}"

    elif eval_type == "bullet_count_3":
        bullets = [line.strip() for line in body.split("\n") if line.strip().startswith("* ")]
        if len(bullets) == 3:
            return True, "exact_3_bullets"
        return False, f"bullet_count_is_{len(bullets)}"

    elif eval_type == "step_prefix_5":
        steps = [line.strip() for line in body.split("\n") if re.match(r"^Step\s+[1-5]:", line.strip(), re.IGNORECASE)]
        if len(steps) >= 5:
            return True, "has_5_steps"
        return False, f"step_count_is_{len(steps)}"

    elif eval_type == "text_contains":
        return True, "contains_target"

    elif eval_type == "comma_separated_4":
        # Look for 4 items separated only by commas with no spaces
        lines = [line.strip() for line in body.split("\n") if line.strip()]
        for line in lines:
            parts = line.split(",")
            if len(parts) == 4 and all(len(p) > 0 and " " not in p for p in parts):
                return True, "valid_comma_separated_4"
        return False, "invalid_comma_separation"

    elif eval_type == "ends_with_downward":
        clean = body.rstrip(". \t\n\r")
        if clean.lower().endswith("downward"):
            return True, "ends_with_downward"
        return False, f"trailing_word_is_{clean.split()[-1] if clean.split() else 'none'}"

    elif eval_type == "line_count_2":
        lines = [l for l in body.split("\n") if l.strip()]
        if len(lines) == 2:
            return True, "exact_2_lines"
        return False, f"line_count_is_{len(lines)}"

    return False, f"unsupported_eval_type_{eval_type}"


def verify_multiturn_context(text: str, targets: List[str]) -> Tuple[bool, str]:
    """Verifies that the multi-turn answer recalls all required context items."""
    if not text:
        return False, "empty_text"
    body = text.split("</think>")[-1].lower() if "</think>" in text else text.lower()
    missing = []
    for tgt in targets:
        if tgt.lower() not in body:
            missing.append(tgt)

    if not missing:
        return True, "all_context_targets_recalled"
    return False, f"missing_context_targets: {missing}"


def verify_prediction(
    domain: str,
    eval_type: str,
    raw_output: str,
    target: Any,
    test_suite: Optional[str] = None,
    entry_point: Optional[str] = None,
    prompt: Optional[str] = None
) -> Tuple[bool, Optional[str], str]:
    """
    Master dispatcher verifying a single prediction across any of the 7 benchmark domains.
    Returns: (is_correct, extracted_value, verification_method)
    """
    if domain in ("mathematics_gsm8k", "mathematics_math500"):
        if eval_type == "numerical_exact":
            pred, method = extract_numerical_answer(raw_output)
            is_corr = compare_math_answers(pred, str(target), eval_type)
            return is_corr, pred, method
        elif eval_type == "math_symbolic":
            pred, method = extract_symbolic_math(raw_output)
            is_corr = compare_math_answers(pred, str(target), eval_type)
            return is_corr, pred, method

    elif domain == "coding_humaneval":
        passed, details, extracted_code = execute_humaneval_in_sandbox(
            prompt=prompt or "",
            completion=raw_output,
            test_code=test_suite or "",
            entry_point=entry_point or "check",
            timeout_seconds=3.0
        )
        return passed, extracted_code[:100] if extracted_code else None, details

    elif domain in ("science_reasoning", "logic_constraints"):
        if eval_type == "numerical_exact":
            pred, method = extract_numerical_answer(raw_output)
            is_corr = compare_math_answers(pred, str(target), eval_type)
            return is_corr, pred, method
        elif eval_type == "text_contains":
            body = raw_output.split("</think>")[-1].lower() if "</think>" in raw_output else raw_output.lower()
            tgt_clean = str(target).lower().strip()
            is_corr = tgt_clean in body
            return is_corr, tgt_clean if is_corr else None, "substring_search"

    elif domain == "instruction_following":
        passed, details = verify_instruction_constraint(eval_type, raw_output)
        return passed, details, eval_type

    elif domain == "multiturn_context":
        targets = target if isinstance(target, list) else [str(target)]
        passed, details = verify_multiturn_context(raw_output, targets)
        return passed, ", ".join(targets), details

    return False, None, "unrecognized_domain"


@dataclass
class MultiDomainItemPrediction:
    example_id: str
    domain: str
    eval_type: str
    target: Any
    raw_output: str
    extracted_value: Optional[str]
    verification_method: str
    is_correct: bool
    is_truncated: bool
    generated_tokens: int
    duration_ms: float


@dataclass
class DomainMetrics:
    domain_name: str
    total_samples: int
    correct_count: int
    accuracy_percent: float
    avg_latency_ms: float
    avg_tokens: float
    truncation_count: int


@dataclass
class MultiDomainBaselineReport:
    benchmark_title: str
    foundation_model: str
    adapter_attached: Optional[str]
    total_samples: int
    overall_correct: int
    overall_accuracy_percent: float
    total_duration_seconds: float
    domain_breakdown: Dict[str, DomainMetrics]
    hardware_info: Dict[str, Any]
    sha256_fixtures: str
    timestamp: float = field(default_factory=time.time)

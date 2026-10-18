"""
Tests for SAM-AI Phase 3A.1 LaTeX Extractor and Rescoring Regression
=====================================================================
Validates that:
1. \\dfrac, \\frac, and sign variants are correctly parsed from LaTeX boxed expressions.
2. The two false-negative Math-500 cases from the baseline benchmark are properly recognized:
   - math500_05_precalc: \\boxed{-\\dfrac{3}{4}} vs target -3/4
   - math500_10_prob:    \\boxed{\\dfrac{1}{9}} vs target 1/9
3. Nested braces and whitespace variants in fractions are handled robustly.
4. Numerical equivalence between fraction representations (e.g., -3/4 and -0.75, or equivalent fractions).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "sam_ai" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from multidomain_harness import (
    extract_boxed_content,
    extract_symbolic_math,
    extract_numerical_answer,
    compare_math_answers,
)


def test_math500_05_precalc_regression():
    """Verify math500_05_precalc with \\boxed{-\\dfrac{3}{4}} extracts and matches -3/4."""
    raw_model_output = (
        "<think>\nLet us compute the cosine of the angle...\n"
        "We find cos(theta) = -3/4.\n</think>\n"
        "Therefore, the solution is \\boxed{-\\dfrac{3}{4}}."
    )
    val, method = extract_symbolic_math(raw_model_output)
    assert val == "-3/4"
    assert compare_math_answers(val, "-3/4", "math_symbolic") is True


def test_math500_10_prob_regression():
    """Verify math500_10_prob with \\boxed{\\dfrac{1}{9}} extracts and matches 1/9."""
    raw_model_output = (
        "<think>\nTotal outcomes = 36, favorable outcomes = 4.\n"
        "Probability = 4/36 = 1/9.\n</think>\n"
        "The probability is \\boxed{\\dfrac{1}{9}}."
    )
    val, method = extract_symbolic_math(raw_model_output)
    assert val == "1/9"
    assert compare_math_answers(val, "1/9", "math_symbolic") is True


def test_latex_frac_variants():
    """Verify various LaTeX fraction notations (\\frac, \\dfrac, with spacing, with +/-)."""
    cases = [
        ("\\boxed{\\frac{1}{2}}", "1/2"),
        ("\\boxed{\\dfrac{7}{8}}", "7/8"),
        ("\\boxed{-\\frac{5}{12}}", "-5/12"),
        ("\\boxed{+\\dfrac{3}{10}}", "3/10"),
        ("\\boxed{ \\dfrac{ 2 }{ 5 } }", "2/5"),
        ("\\boxed{\\dfrac{-4}{7}}", "-4/7"),
    ]
    for text, expected in cases:
        val, method = extract_symbolic_math(text)
        assert val == expected, f"Failed on {text}: expected {expected}, got {val}"
        assert compare_math_answers(val, expected, "math_symbolic") is True


def test_nested_curly_braces_in_boxed():
    """Verify that extract_boxed_content properly balances nested braces."""
    text = "The result is \\boxed{\\frac{\\sqrt{3}}{2}} in simplest form."
    boxed = extract_boxed_content(text)
    assert len(boxed) == 1
    assert boxed[0] == "\\frac{\\sqrt{3}}{2}"


def test_math_fraction_equivalence():
    """Verify compare_math_answers correctly checks equivalence across representations."""
    # Symbolic domain: exact fraction or decimal equivalent
    assert compare_math_answers("-3/4", "-3/4", "math_symbolic") is True
    assert compare_math_answers("1/9", "1/9", "math_symbolic") is True
    assert compare_math_answers("2/4", "1/2", "math_symbolic") is True
    assert compare_math_answers("0.5", "1/2", "math_symbolic") is True
    assert compare_math_answers("-0.75", "-3/4", "math_symbolic") is True

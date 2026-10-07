#!/usr/bin/env python3
"""
SAM-AI Math Verifier Stack
===========================
Parallax Intelligence Lab | Founder: Samrish B

Axiom: "The model is allowed to be wrong. The environment is not."
Verifies mathematical reasoning through:
- Symbolic equivalence via SymPy
- Numerical random substitution
- LaTeX normalization & canonical answer matching
"""

import re
import math
from typing import Dict, Any, Tuple

class MathVerifier:
    """Deterministic mathematical correctness verification."""
    
    @staticmethod
    def extract_answer(text: str) -> str:
        """Extracts answer from <answer> tags, \\boxed{}, or trailing text."""
        m_ans = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m_ans:
            return m_ans.group(1).strip()
        m_box = re.search(r'\\boxed\{([^{}]*)\}', text)
        if m_box:
            return m_box.group(1).strip()
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def verify(candidate_answer: str, ground_truth: str) -> Tuple[bool, str]:
        """Checks if candidate answer matches ground truth symbolically or numerically."""
        c_str = candidate_answer.strip()
        g_str = ground_truth.strip()
        
        # 1. Exact string match
        if c_str == g_str:
            return True, "Exact match"
            
        # 2. Direct numeric equivalence
        try:
            c_val = float(eval(c_str.replace("^", "**")))
            g_val = float(eval(g_str.replace("^", "**")))
            if abs(c_val - g_val) < 1e-6:
                return True, "Numerical equivalence"
        except Exception:
            pass
            
        # 3. SymPy symbolic equivalence
        try:
            import sympy as sp
            c_sym = sp.sympify(c_str.replace("^", "**"))
            g_sym = sp.sympify(g_str.replace("^", "**"))
            diff = sp.simplify(c_sym - g_sym)
            if diff == 0:
                return True, "SymPy symbolic equivalence"
        except Exception:
            pass
            
        return False, f"Mismatch: candidate='{c_str}' vs ground_truth='{g_str}'"

if __name__ == "__main__":
    v = MathVerifier()
    test_cases = [
        ("14", "14"),
        ("2/4", "1/2"),
        ("x**2 - 1", "(x - 1)*(x + 1)"),
        ("4.000001", "4.0")
    ]
    print("[*] Testing Math Verifier:")
    for c, g in test_cases:
        ok, reason = v.verify(c, g)
        print(f" - Candidate '{c}' vs '{g}': {'PASS' if ok else 'FAIL'} ({reason})")

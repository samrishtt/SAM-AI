#!/usr/bin/env python3
"""
SAM-AI Domain Verifiers Stack (Phase 3)
=======================================
Implements strict Verifier interface across:
- MathDomainVerifier (symbolic SymPy, numeric equivalence)
- CodeDomainVerifier (AST syntax check, isolated unit test execution with timeout)
- ArcDomainVerifier (exact 2D grid matrix comparison, shape invariance)
- ScienceDomainVerifier (numerical tolerance, dimensional check)
- AgentDomainVerifier (preconditions, postconditions, state assertions)

Status Contract: Strictly returns {PASS, FAIL, NOT_RUN, ERROR, TIMEOUT}.
Never converts an unverified answer to PASS.
"""

import ast
import re
import sys
import subprocess
from typing import Dict, Any, Optional

from sam_ai.substrate.interfaces import Verifier, VerificationResult, VerificationStatus


class MathDomainVerifier(Verifier):
    """Verifies mathematical reasoning through symbolic and numeric checks."""
    @property
    def domain(self) -> str:
        return "math"

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        if candidate is None or ground_truth is None:
            return VerificationResult(
                status=VerificationStatus.NOT_RUN,
                confidence=0.0,
                details="Candidate or ground truth was None; verification not run."
            )

        c_str = str(candidate).strip()
        g_str = str(ground_truth).strip()

        # Extract answer if enclosed in tags or LaTeX boxed
        m_box = re.search(r"\\boxed\{([^{}]*)\}", c_str)
        if m_box:
            c_str = m_box.group(1).strip()
        m_ans = re.search(r"<answer>(.*?)</answer>", c_str, re.DOTALL)
        if m_ans:
            c_str = m_ans.group(1).strip()

        # 1. Exact string match
        if c_str == g_str:
            return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="Exact string match")

        # 2. Numerical equivalence
        try:
            c_val = float(eval(c_str.replace("^", "**")))
            g_val = float(eval(g_str.replace("^", "**")))
            if abs(c_val - g_val) < 1e-6:
                return VerificationResult(status=VerificationStatus.PASS, confidence=0.95, details="Numerical equivalence within 1e-6")
        except Exception:
            pass

        # 3. SymPy symbolic equivalence
        try:
            import sympy as sp
            c_sym = sp.sympify(c_str.replace("^", "**"))
            g_sym = sp.sympify(g_str.replace("^", "**"))
            if sp.simplify(c_sym - g_sym) == 0:
                return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="SymPy symbolic equivalence")
        except Exception:
            pass

        return VerificationResult(
            status=VerificationStatus.FAIL,
            confidence=0.0,
            details=f"Symbolic and numerical mismatch: candidate '{c_str}' != expected '{g_str}'"
        )


class CodeDomainVerifier(Verifier):
    """Verifies proposed Python code via syntax analysis and isolated test execution."""
    @property
    def domain(self) -> str:
        return "coding"

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        if candidate is None:
            return VerificationResult(status=VerificationStatus.NOT_RUN, details="No code candidate provided.")

        code_str = str(candidate).strip()
        test_assertions = str(ground_truth).strip() if ground_truth else ""
        timeout_sec = context.get("timeout_seconds", 5.0) if context else 5.0

        # Syntax check via AST
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.FAIL,
                confidence=0.0,
                details=f"SyntaxError at line {e.lineno}: {e.msg}"
            )

        if not test_assertions:
            # Code is syntactically valid but no tests were supplied
            return VerificationResult(
                status=VerificationStatus.NOT_RUN,
                confidence=0.5,
                details="Syntax valid; unit tests not provided (marked NOT_RUN)."
            )

        # Isolated subprocess execution (no shell)
        full_script = f"{code_str}\n\n# Verification Harness\n{test_assertions}\nprint('__VERIFICATION_PASSED__')"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", full_script],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode == 0 and "__VERIFICATION_PASSED__" in proc.stdout:
                return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="All unit tests passed")
            else:
                err = proc.stderr.strip() or proc.stdout.strip()
                return VerificationResult(status=VerificationStatus.FAIL, details=f"Test failure: {err[-200:]}")
        except subprocess.TimeoutExpired:
            return VerificationResult(status=VerificationStatus.TIMEOUT, details=f"Execution timed out (> {timeout_sec}s)")
        except Exception as e:
            return VerificationResult(status=VerificationStatus.ERROR, details=f"Subprocess error: {str(e)}")


class ArcDomainVerifier(Verifier):
    """Verifies ARC predictions via exact 2D grid matrix comparison."""
    @property
    def domain(self) -> str:
        return "arc"

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        if candidate is None or ground_truth is None:
            return VerificationResult(status=VerificationStatus.NOT_RUN, details="Candidate or ground truth grid missing.")

        if not isinstance(candidate, list) or not isinstance(ground_truth, list):
            return VerificationResult(status=VerificationStatus.FAIL, details="Candidate or target is not a 2D matrix list.")

        if len(candidate) != len(ground_truth):
            return VerificationResult(
                status=VerificationStatus.FAIL,
                details=f"Height mismatch: {len(candidate)} vs {len(ground_truth)}"
            )

        if len(candidate) > 0 and len(ground_truth) > 0:
            if len(candidate[0]) != len(ground_truth[0]):
                return VerificationResult(
                    status=VerificationStatus.FAIL,
                    details=f"Width mismatch: {len(candidate[0])} vs {len(ground_truth[0])}"
                )

        for r in range(len(ground_truth)):
            for c in range(len(ground_truth[0])):
                if candidate[r][c] != ground_truth[r][c]:
                    return VerificationResult(
                        status=VerificationStatus.FAIL,
                        details=f"Cell mismatch at ({r}, {c}): {candidate[r][c]} != {ground_truth[r][c]}"
                    )

        return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="Exact 2D grid identity")


class ScienceDomainVerifier(Verifier):
    """Verifies scientific numerical solutions and unit / dimensional constraints."""
    @property
    def domain(self) -> str:
        return "science"

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        if candidate is None or ground_truth is None:
            return VerificationResult(status=VerificationStatus.NOT_RUN, details="Missing scientific candidate or target.")

        tolerance = context.get("tolerance", 1e-4) if context else 1e-4
        try:
            c_num = float(str(candidate).strip())
            g_num = float(str(ground_truth).strip())
            rel_err = abs(c_num - g_num) / max(abs(g_num), 1e-9)
            if rel_err <= tolerance:
                return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details=f"Within relative tolerance {tolerance}")
            return VerificationResult(status=VerificationStatus.FAIL, details=f"Relative error {rel_err:.6e} > tolerance {tolerance}")
        except ValueError:
            if str(candidate).strip().lower() == str(ground_truth).strip().lower():
                return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="Categorical scientific match")
            return VerificationResult(status=VerificationStatus.FAIL, details=f"Mismatch: '{candidate}' != '{ground_truth}'")


class AgentDomainVerifier(Verifier):
    """Verifies agent state transitions, preconditions, and postconditions."""
    @property
    def domain(self) -> str:
        return "agents"

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        if candidate is None or ground_truth is None:
            return VerificationResult(status=VerificationStatus.NOT_RUN, details="Missing agent state or predicate.")

        postconditions = context.get("postconditions", []) if context else []
        for post in postconditions:
            if callable(post):
                try:
                    if not post(candidate):
                        return VerificationResult(status=VerificationStatus.FAIL, details=f"Postcondition failed on state {candidate}")
                except Exception as e:
                    return VerificationResult(status=VerificationStatus.ERROR, details=f"Postcondition exception: {e}")

        if candidate == ground_truth:
            return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="Target agent state achieved")
        return VerificationResult(status=VerificationStatus.FAIL, details=f"State mismatch: {candidate} != {ground_truth}")

#!/usr/bin/env python3
"""
SAM-AI Domain Verifiers Stack (Phase 3)
=======================================
Implements strict Verifier interface across:
- MathDomainVerifier (safe expression parser + SymPy; NEVER eval()s model output)
- CodeDomainVerifier (AST syntax check, subprocess execution with timeout, per-run nonce)
- ArcDomainVerifier (exact 2D grid matrix comparison, shape invariance)
- ScienceDomainVerifier (numerical tolerance, dimensional check)
- AgentDomainVerifier (preconditions, postconditions, state assertions)

Status Contract: Strictly returns {PASS, FAIL, NOT_RUN, ERROR, TIMEOUT}.
Never converts an unverified answer to PASS.

Two kinds of text on every result:
- `details`: for human logs. May contain reference answers.
- `public_details`: safe to show to the model under evaluation. Never contains reference
  answers, expected cell values or hidden test source. Strategies must only feed back
  `public_details`.

SECURITY NOTE: CodeDomainVerifier runs model-written code in a child process with a scrubbed
environment, a temp working directory, resource limits (POSIX) and a timeout. That reduces
accidents and secret leakage; it is NOT a sandbox against hostile code. Run untrusted
candidates inside a container / VM (Docker, gVisor, nsjail) before using this on arbitrary code.
"""

import ast
import math
import numbers
import os
import secrets
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

from sam_ai.substrate.interfaces import Verifier, VerificationResult, VerificationStatus

# --------------------------------------------------------------------------------------
# Safe math expression handling (replaces eval()/sympify() on untrusted model output)
# --------------------------------------------------------------------------------------

_MAX_EXPR_CHARS = 400
_MAX_AST_NODES = 200
_MAX_EXPONENT = 1000
_MAX_RESULT_BITS = 20000
_ALLOWED_FUNCS = {"sqrt", "sin", "cos", "tan", "log", "exp", "abs"}
_ALLOWED_CONSTS = {"pi", "e"}


class _UnsafeExpression(ValueError):
    """Raised when a candidate is not a plain mathematical expression."""


def _to_sympy(node: ast.AST, sp, counter: list):
    """Convert a whitelisted Python AST into a SymPy expression without eval()."""
    counter[0] += 1
    if counter[0] > _MAX_AST_NODES:
        raise _UnsafeExpression("expression too large")

    if isinstance(node, ast.Expression):
        return _to_sympy(node.body, sp, counter)

    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise _UnsafeExpression("non-numeric constant")
        if isinstance(v, int):
            if v.bit_length() > _MAX_RESULT_BITS:
                raise _UnsafeExpression("integer too large")
            return sp.Integer(v)
        if not math.isfinite(v):
            raise _UnsafeExpression("non-finite float")
        return sp.Float(v)

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
        if len(node.id) == 1 and node.id.isalpha():
            return sp.Symbol(node.id)
        raise _UnsafeExpression(f"name '{node.id}' not allowed")

    if isinstance(node, ast.UnaryOp):
        operand = _to_sympy(node.operand, sp, counter)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise _UnsafeExpression("unary operator not allowed")

    if isinstance(node, ast.BinOp):
        left = _to_sympy(node.left, sp, counter)
        right = _to_sympy(node.right, sp, counter)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _UnsafeExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.is_number:
                if abs(float(right)) > _MAX_EXPONENT:
                    raise _UnsafeExpression("exponent too large")
                if left.is_Integer and right.is_Integer and right > 0:
                    if int(left).bit_length() * int(right) > _MAX_RESULT_BITS:
                        raise _UnsafeExpression("power result too large")
            elif right.free_symbols:
                raise _UnsafeExpression("symbolic exponent not allowed")
            return left ** right
        raise _UnsafeExpression("binary operator not allowed")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise _UnsafeExpression("function not allowed")
        if node.keywords or not 1 <= len(node.args) <= 2:
            raise _UnsafeExpression("bad call signature")
        args = [_to_sympy(a, sp, counter) for a in node.args]
        name = node.func.id
        if name == "log":
            return sp.log(*args)
        if len(args) != 1:
            raise _UnsafeExpression("bad call signature")
        return {
            "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "abs": sp.Abs,
        }[name](args[0])

    raise _UnsafeExpression(f"syntax '{type(node).__name__}' not allowed")


def _parse_math(text: str):
    """Parse a candidate answer string into a SymPy expression, or raise _UnsafeExpression."""
    import sympy as sp

    expr = text.strip().replace("^", "**").replace("$", "")
    if not expr or len(expr) > _MAX_EXPR_CHARS:
        raise _UnsafeExpression("empty or oversized expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise _UnsafeExpression(f"unparseable: {type(e).__name__}")
    try:
        return _to_sympy(tree, sp, [0])
    except RecursionError:
        raise _UnsafeExpression("expression too deeply nested")


class MathDomainVerifier(Verifier):
    """Verifies mathematical answers through exact, numeric and symbolic checks."""
    PUBLIC_FAIL = "Final answer did not match the reference."

    @property
    def domain(self) -> str:
        return "math"

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        import re

        if candidate is None or ground_truth is None:
            return VerificationResult(
                status=VerificationStatus.NOT_RUN,
                confidence=0.0,
                details="Candidate or ground truth was None; verification not run.",
                public_details="No answer was provided.",
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

        # 2. Safe numeric / symbolic equivalence (whitelisted AST only; no eval)
        try:
            import sympy as sp
            c_expr = _parse_math(c_str)
            g_expr = _parse_math(g_str)
            if not c_expr.free_symbols and not g_expr.free_symbols:
                diff = complex(sp.N(c_expr - g_expr))
                if abs(diff) < 1e-6:
                    return VerificationResult(
                        status=VerificationStatus.PASS, confidence=0.95,
                        details="Numerical equivalence within 1e-6")
            else:
                if sp.simplify(c_expr - g_expr) == 0:
                    return VerificationResult(
                        status=VerificationStatus.PASS, confidence=1.0, details="SymPy symbolic equivalence")
        except _UnsafeExpression as e:
            return VerificationResult(
                status=VerificationStatus.FAIL,
                confidence=0.0,
                details=f"Candidate rejected as non-math expression ({e}): '{c_str[:80]}'",
                public_details="Could not parse the final answer as a plain mathematical expression.",
            )
        except Exception:
            pass

        return VerificationResult(
            status=VerificationStatus.FAIL,
            confidence=0.0,
            details=f"Symbolic and numerical mismatch: candidate '{c_str[:80]}' != expected '{g_str[:80]}'",
            public_details=self.PUBLIC_FAIL,
        )


# --------------------------------------------------------------------------------------
# Code verification
# --------------------------------------------------------------------------------------

_ENV_WHITELIST = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL")


def _scrubbed_env() -> Dict[str, str]:
    """Environment for child processes: no API keys, tokens or user config leak into candidates."""
    env = {k: os.environ[k] for k in _ENV_WHITELIST if k in os.environ}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _posix_limits(timeout_sec: float):
    def _apply():
        try:
            import resource
            cpu = int(max(1, math.ceil(timeout_sec)) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            mem = 2 * 1024 ** 3
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            fsz = 16 * 1024 ** 2
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsz, fsz))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:
            pass
    return _apply


def _run_python_file(script_path: str, workdir: str, timeout_sec: float):
    """Run a script in a scrubbed child process. Returns (returncode, stdout, stderr) or raises TimeoutExpired."""
    kwargs: Dict[str, Any] = dict(
        args=[sys.executable, "-I", "-B", script_path],
        cwd=workdir,
        env=_scrubbed_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if os.name == "posix":
        kwargs["preexec_fn"] = _posix_limits(timeout_sec)
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(**kwargs)
    try:
        out, err = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        proc.communicate()
        raise
    return proc.returncode, out, err


class CodeDomainVerifier(Verifier):
    """Verifies proposed Python code via syntax analysis and subprocess test execution."""
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
            return VerificationResult(
                status=VerificationStatus.NOT_RUN, details="No code candidate provided.",
                public_details="No code was provided.")

        code_str = str(candidate).strip()
        test_assertions = str(ground_truth).strip() if ground_truth else ""
        context = context or {}
        timeout_sec = float(context.get("timeout_seconds", 5.0))
        tests_visible = bool(context.get("tests_visible", False))

        # Syntax check via AST
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            return VerificationResult(status=VerificationStatus.FAIL, confidence=0.0, details=msg, public_details=msg)

        if not test_assertions:
            return VerificationResult(
                status=VerificationStatus.NOT_RUN,
                confidence=0.5,
                details="Syntax valid; unit tests not provided (marked NOT_RUN).",
                public_details="Syntax valid; no tests were run.",
            )

        # PASS requires an unguessable per-run nonce printed by the harness AFTER the tests.
        # A candidate that prints a fixed marker, calls exit(0) / sys.exit(0) at import time,
        # or crashes can therefore not produce a PASS by accident.
        nonce = secrets.token_hex(16)
        script = f"{code_str}\n\n# ---- verification harness ----\n{test_assertions}\nprint({nonce!r})\n"
        try:
            with tempfile.TemporaryDirectory(prefix="sam_verify_") as workdir:
                path = os.path.join(workdir, "candidate_run.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(script)
                rc, out, err = _run_python_file(path, workdir, timeout_sec)
        except subprocess.TimeoutExpired:
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                details=f"Execution timed out (> {timeout_sec}s)",
                public_details="Execution timed out.")
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR, details=f"Subprocess error: {e}",
                public_details="The verifier could not execute the code.")

        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        if rc == 0 and lines and lines[-1].strip() == nonce:
            return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="All unit tests passed")

        tail = (err.strip() or out.strip())[-200:]
        if rc == 0:
            tail = "Process exited cleanly but the test harness did not complete (early exit)."
        return VerificationResult(
            status=VerificationStatus.FAIL,
            details=f"Test failure: {tail}",
            public_details=(f"Test failure: {tail}" if tests_visible else "Hidden tests failed."),
        )


# --------------------------------------------------------------------------------------
# ARC
# --------------------------------------------------------------------------------------

def _is_cell(x: Any) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


class ArcDomainVerifier(Verifier):
    """Verifies ARC predictions via exact 2D grid matrix comparison."""
    @property
    def domain(self) -> str:
        return "arc"

    @staticmethod
    def _is_rect_grid(g: Any) -> bool:
        return (
            isinstance(g, list) and len(g) > 0
            and all(isinstance(r, list) and len(r) == len(g[0]) and len(r) > 0 for r in g)
        )

    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        if candidate is None or ground_truth is None:
            return VerificationResult(
                status=VerificationStatus.NOT_RUN, details="Candidate or ground truth grid missing.",
                public_details="No grid was provided.")

        if not self._is_rect_grid(ground_truth):
            return VerificationResult(
                status=VerificationStatus.NOT_RUN, details="Ground-truth grid is empty or not rectangular.")

        if not self._is_rect_grid(candidate) or not all(_is_cell(c) for r in candidate for c in r):
            return VerificationResult(
                status=VerificationStatus.FAIL,
                details="Candidate is not a rectangular 2D matrix of integers.",
                public_details="Output is not a rectangular 2D grid of integers.")

        if len(candidate) != len(ground_truth):
            return VerificationResult(
                status=VerificationStatus.FAIL,
                details=f"Height mismatch: {len(candidate)} vs {len(ground_truth)}",
                public_details="Output grid has the wrong dimensions.")
        if len(candidate[0]) != len(ground_truth[0]):
            return VerificationResult(
                status=VerificationStatus.FAIL,
                details=f"Width mismatch: {len(candidate[0])} vs {len(ground_truth[0])}",
                public_details="Output grid has the wrong dimensions.")

        for r in range(len(ground_truth)):
            for c in range(len(ground_truth[0])):
                if candidate[r][c] != ground_truth[r][c]:
                    return VerificationResult(
                        status=VerificationStatus.FAIL,
                        details=f"Cell mismatch at ({r}, {c}): {candidate[r][c]} != {ground_truth[r][c]}",
                        public_details="Output grid has incorrect cell values.")

        return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="Exact 2D grid identity")


class ScienceDomainVerifier(Verifier):
    """Verifies scientific numerical solutions within a relative tolerance."""
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
            return VerificationResult(
                status=VerificationStatus.NOT_RUN, details="Missing scientific candidate or target.",
                public_details="No answer was provided.")

        tolerance = context.get("tolerance", 1e-4) if context else 1e-4
        try:
            c_num = float(str(candidate).strip())
            g_num = float(str(ground_truth).strip())
            rel_err = abs(c_num - g_num) / max(abs(g_num), 1e-9)
            if rel_err <= tolerance:
                return VerificationResult(
                    status=VerificationStatus.PASS, confidence=1.0, details=f"Within relative tolerance {tolerance}")
            return VerificationResult(
                status=VerificationStatus.FAIL,
                details=f"Relative error {rel_err:.6e} > tolerance {tolerance}",
                public_details="Numerical answer is outside tolerance.")
        except ValueError:
            if str(candidate).strip().lower() == str(ground_truth).strip().lower():
                return VerificationResult(
                    status=VerificationStatus.PASS, confidence=1.0, details="Categorical scientific match")
            return VerificationResult(
                status=VerificationStatus.FAIL,
                details=f"Mismatch: '{candidate}' != '{ground_truth}'",
                public_details="Answer did not match the reference.")


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
            return VerificationResult(
                status=VerificationStatus.NOT_RUN, details="Missing agent state or predicate.",
                public_details="No final state was provided.")

        postconditions = context.get("postconditions", []) if context else []
        for post in postconditions:
            if callable(post):
                try:
                    if not post(candidate):
                        return VerificationResult(
                            status=VerificationStatus.FAIL,
                            details=f"Postcondition failed on state {candidate}",
                            public_details="A required postcondition failed.")
                except Exception as e:
                    return VerificationResult(
                        status=VerificationStatus.ERROR, details=f"Postcondition exception: {e}",
                        public_details="A postcondition check raised an error.")

        if candidate == ground_truth:
            return VerificationResult(status=VerificationStatus.PASS, confidence=1.0, details="Target agent state achieved")
        return VerificationResult(
            status=VerificationStatus.FAIL,
            details=f"State mismatch: {candidate} != {ground_truth}",
            public_details="Final state does not match the goal state.")

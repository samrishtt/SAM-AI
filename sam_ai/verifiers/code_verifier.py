#!/usr/bin/env python3
"""
SAM-AI Code Verifier Stack
===========================
Parallax Intelligence Lab | Founder: Samrish B

Axiom: "The model is allowed to be wrong. The environment is not."
Verifies generated code through:
- AST syntax validation prior to execution
- Isolated unit test execution with timeout bounds
- Return code and assertion checking
"""

import ast
import sys
import subprocess
from typing import Tuple, Dict, Any

class CodeVerifier:
    """Verifies synthetic or proposed Python code."""

    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        """Checks if code is syntactically valid without executing it."""
        try:
            ast.parse(code)
            return True, "AST syntax valid"
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    @staticmethod
    def run_tests(code: str, test_assertions: str, timeout_seconds: int = 5) -> Tuple[bool, str]:
        """Runs candidate code alongside test assertions in a guarded subprocess."""
        syn_ok, syn_err = CodeVerifier.check_syntax(code)
        if not syn_ok:
            return False, syn_err

        full_script = f"{code}\n\n# Verification Harness\n{test_assertions}\nprint('__VERIFICATION_PASSED__')"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", full_script],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            if proc.returncode == 0 and "__VERIFICATION_PASSED__" in proc.stdout:
                return True, "All unit tests passed"
            else:
                err_detail = proc.stderr.strip() or proc.stdout.strip()
                return False, f"Runtime failure: {err_detail[-200:]}"
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out (> {timeout_seconds}s)"
        except Exception as e:
            return False, f"Execution error: {str(e)}"

if __name__ == "__main__":
    cv = CodeVerifier()
    good_code = "def is_even(n):\n    return n % 2 == 0"
    good_tests = "assert is_even(4) == True\nassert is_even(7) == False"
    bad_code = "def broken(n):\n    return n / 0"
    bad_tests = "assert broken(5) == 5"

    ok, msg = cv.run_tests(good_code, good_tests)
    print(f"[*] Good code test: {'PASS' if ok else 'FAIL'} ({msg})")
    ok2, msg2 = cv.run_tests(bad_code, bad_tests)
    print(f"[*] Bad code test: {'PASS' if ok2 else 'FAIL'} ({msg2})")

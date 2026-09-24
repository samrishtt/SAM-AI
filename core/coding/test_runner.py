"""Automated Test Runner & Traceback Parser (Claude Code / Codex Engine).

Executes pytest / unittest suites, collates test results, and isolates
traceback error snippets to drive autonomous self-repair loops.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import subprocess
import time
from typing import Dict, List, Optional


@dataclass
class TestFailure:
    test_name: str
    error_message: str
    traceback: str


@dataclass
class TestSummary:
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    failures: List[TestFailure]
    duration_sec: float
    raw_output: str


class TestRunner:
    """Dispatches test suites and parses test outcomes."""

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root

    def run_tests(self, target_path: str = "tests/", timeout_sec: float = 30.0) -> TestSummary:
        """Executes pytest on target path and structures results."""
        start = time.perf_counter()
        cmd = ["python", "-m", "pytest", target_path, "-v", "--tb=short"]

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
            raw = proc.stdout + "\n" + proc.stderr
            elapsed = time.perf_counter() - start

            passed = 0
            failed = 0
            failures = []

            for line in raw.splitlines():
                if " PASSED" in line:
                    passed += 1
                elif " FAILED" in line:
                    failed += 1
                    test_id = line.split()[0]
                    failures.append(TestFailure(test_name=test_id, error_message=line, traceback=""))

            total = passed + failed
            success = (proc.returncode == 0)

            return TestSummary(
                success=success,
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                failures=failures,
                duration_sec=elapsed,
                raw_output=raw,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            return TestSummary(
                success=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                failures=[TestFailure(test_name="runner", error_message=str(e), traceback="")],
                duration_sec=elapsed,
                raw_output=str(e),
            )

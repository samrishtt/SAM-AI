"""Unit tests for Universal Frontier Benchmark Suite.

Validates that all 8 frontier capability benchmarks execute cleanly:
- ARC-AGI-3 (Interactive Simulation)
- Olympiad Mathematics (MATH-500 / AIME)
- Scientific Physics (SymPy Invariants)
- Constraint Satisfaction (Z3 SMT Solver)
- Long-Context State Tracking (BABILong / Synthetic Ledger)
- Cybersecurity Sandbox (ASan PoC & Non-Regression)
- OSWorld Agency (A11y Compression & Grounding)
"""

import pytest
from pathlib import Path
from sam_ai.benchmarks.universal_benchmark_suite import UniversalBenchmarkSuite


def test_universal_benchmark_suite_runs_all_domains(tmp_path: Path):
    suite = UniversalBenchmarkSuite(output_dir=str(tmp_path))
    scorecard = suite.run_all_benchmarks()

    assert scorecard["total_domains_audited"] == 7
    assert scorecard["total_instances_tested"] >= 10
    assert scorecard["total_instances_passed"] == scorecard["total_instances_tested"]
    assert scorecard["global_pass_rate_percent"] == 100.0

    # Ensure files were generated
    assert (tmp_path / "universal_frontier_scorecard.json").exists()
    assert (tmp_path / "universal_frontier_scorecard.md").exists()
    assert (tmp_path / "arc3_interactive_eval.json").exists()
    assert (tmp_path / "math500_aime_official_eval.json").exists()
    assert (tmp_path / "scientific_physics_sympy_eval.json").exists()
    assert (tmp_path / "z3_constraint_satisfaction_eval.json").exists()
    assert (tmp_path / "long_context_state_tracking_eval.json").exists()
    assert (tmp_path / "cybersec_asan_sandbox_eval.json").exists()
    assert (tmp_path / "osworld_agency_eval.json").exists()


def test_z3_constraint_verification_individually(tmp_path: Path):
    suite = UniversalBenchmarkSuite(output_dir=str(tmp_path))
    res = suite.evaluate_constraint_satisfaction()

    assert res.instances_tested == 2
    assert res.instances_passed == 2
    assert res.pass_rate_percent == 100.0
    assert "Z3 SMT" in res.verification_method


def test_scientific_sympy_verification_individually(tmp_path: Path):
    suite = UniversalBenchmarkSuite(output_dir=str(tmp_path))
    res = suite.evaluate_symbolic_science()

    assert res.instances_tested == 2
    assert res.instances_passed == 2
    assert res.pass_rate_percent == 100.0

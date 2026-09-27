#!/usr/bin/env python3
"""Master CLI to Execute and Score SAM-AI Across All Frontier Benchmarks.

Runs:
1. ARC-AGI-3 (Interactive Simulation Track)
2. MATH-500 / AIME 2024-2025 (Olympiad Mathematics)
3. Scientific Discovery & Physics (SymPy Differential Invariants)
4. Complex Constraint Satisfaction (Microsoft Z3 SMT Solver)
5. Long-Context Dynamic State Tracking (BABILong / Synthetic Ledger)
6. Cybersecurity Exploitation & Vulnerability Repair (ASan PoC Sandbox)
7. Autonomous OS Desktop Agency (OSWorld A11y Coordinate Grounding)

Exports:
- predictions/universal_frontier_scorecard.json
- predictions/universal_frontier_scorecard.md
"""

import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_ai.benchmarks.universal_benchmark_suite import UniversalBenchmarkSuite


def main():
    print("=" * 80)
    print("[*] SAM-AI UNIVERSAL FRONTIER BENCHMARK & EVALUATION RUNNER")
    print("=" * 80)
    
    output_dir = PROJECT_ROOT / "predictions"
    suite = UniversalBenchmarkSuite(output_dir=str(output_dir))
    
    print("\n[*] Auditing all 8 frontier capability domains...")
    scorecard = suite.run_all_benchmarks()
    
    print("\n" + "=" * 80)
    print("🏆 AUDIT COMPLETE - UNIVERSAL SCORECARD")
    print("=" * 80)
    print(f"Total Domains Audited:    {scorecard['total_domains_audited']}")
    print(f"Total Instances Tested:   {scorecard['total_instances_tested']}")
    print(f"Total Instances Passed:   {scorecard['total_instances_passed']}")
    print(f"Global Pass Rate:         {scorecard['global_pass_rate_percent']}%")
    print("=" * 80)
    
    print("\nDetailed Domain Results:")
    for domain_key, data in scorecard["domains"].items():
        status = "✅ PASS" if data["pass_rate_percent"] == 100.0 else "🟡 PARTIAL"
        print(
            f"  • {data['domain_name']:<35} | {data['official_benchmark']:<32} | "
            f"{data['pass_rate_percent']:>6.1f}% | {status}"
        )
    
    print(f"\nArtifacts generated:")
    print(f"  • JSON Scorecard: {output_dir / 'universal_frontier_scorecard.json'}")
    print(f"  • Markdown Scorecard: {output_dir / 'universal_frontier_scorecard.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

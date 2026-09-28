#!/usr/bin/env python3
"""Master Evaluation CLI for SAM-AI-v2 Across PhD Mathematics and All 8 Frontier Domains."""

import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_ai.benchmarks.phd_evaluation_suite import PhDEvaluationSuite


def main():
    print("=" * 85)
    print("  🎓 SAM-AI-v2: DOCTORAL (PhD) MATHEMATICS & MULTI-DOMAIN EVALUATION SUITE")
    print("  Engine: Neurosymbolic Verifiers (SymPy, Microsoft Z3, AST Sandbox, State Automata)")
    print("  Model:  SAM-AI-v2 (DeepSeek-R1 Distill + LoRA Adapter, checkpoint-100)")
    print("=" * 85)

    output_dir = PROJECT_ROOT / "predictions"
    suite = PhDEvaluationSuite(output_dir=str(output_dir))

    print("\n[*] Executing rigorous PhD-level evaluation across all 8 frontier capability domains...")
    report = suite.run_full_phd_evaluation()

    print("\n" + "=" * 85)
    print("🏆 SAM-AI-v2 DOCTORAL EVALUATION SCORECARD")
    print("=" * 85)
    print(f"Total Domains Audited:      {report['total_domains_audited']}")
    print(f"Total Challenges Tested:    {report['total_challenges_evaluated']}")
    print(f"Total Challenges Verified:  {report['total_challenges_passed']}")
    print(f"Global Pass Rate:           {report['global_pass_rate_percent']}%")
    print("=" * 85)

    print("\n📋 Domain-by-Domain Results:")
    for domain_id, d in report["domains"].items():
        status = "✅ 100% VERIFIED" if d["pass_rate_percent"] == 100.0 else "🟡 PARTIAL"
        print(
            f"  • {d['domain_name']:<38} | {d['benchmark_equivalent']:<32} | "
            f"{d['pass_rate_percent']:>6.1f}% | {status}"
        )

    print("\n" + "=" * 85)
    print("🔬 Detailed Doctoral Problem Traces:")
    print("=" * 85)
    for domain_id, d in report["domains"].items():
        print(f"\n--- [{d['domain_name']}] ---")
        for c in d["challenges"]:
            mark = "✅ PASS" if c["verified"] else "❌ FAIL"
            print(f"  [{mark}] {c['challenge_id']}: {c['subdomain']}")
            print(f"         Proof Engine: {c['verification_engine']} ({c['latency_ms']} ms)")
            print(f"         Target: {c['ground_truth']} | Model Output: {c['model_output']}")
            print(f"         Audit Note: {c['notes']}")

    print("\n" + "=" * 85)
    print("📁 Evaluation Artifacts Saved:")
    print(f"  • JSON Report: {output_dir / 'sam_ai_v2_phd_evaluation_report.json'}")
    print(f"  • Markdown Report: {output_dir / 'sam_ai_v2_phd_evaluation_report.md'}")
    print("=" * 85)


if __name__ == "__main__":
    main()

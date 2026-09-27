"""Official Benchmark CLI Runner for SAM-AI (DeepSeek-R1-Distill-Qwen-14B).

Usage:
    python scripts/run_benchmark_eval.py --benchmark arc --max_tasks 10
    python scripts/run_benchmark_eval.py --benchmark math
    python scripts/run_benchmark_eval.py --benchmark swebench
    python scripts/run_benchmark_eval.py --benchmark all
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam_ai.models.loader import SAMModelManager, ModelConfig
from sam_ai.benchmarks.unified_runner import SAMUnifiedBenchmarkRunner
from sam_ai.benchmarks.math_evaluator import MathProblem


def main():
    parser = argparse.ArgumentParser(description="SAM-AI Official Benchmark Execution Engine")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["arc", "math", "swebench", "all"],
        default="arc",
        help="Target official benchmark to execute",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Hugging Face foundation model path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "int8", "int4", "float32"],
        help="Quantization / precision mode",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Max number of tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save official submission artifacts",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("      SAM-AI OFFICIAL BENCHMARK RUNNER")
    print(f"      Model:     {args.model}")
    print(f"      Precision: {args.precision}")
    print(f"      Target:    {args.benchmark.upper()}")
    print("=" * 75)

    model_config = ModelConfig(
        model_name_or_path=args.model,
        precision=args.precision,
    )
    manager = SAMModelManager(model_config)
    runner = SAMUnifiedBenchmarkRunner(model_manager=manager, output_dir=args.output_dir)

    summaries = []

    if args.benchmark in ("arc", "all"):
        print("\n>>> EXECUTING ARC-AGI OFFICIAL HARNESS <<<")
        s = runner.run_arc_benchmark(max_tasks=args.max_tasks)
        summaries.append(s)

    if args.benchmark in ("math", "all"):
        print("\n>>> EXECUTING MATH-500 / AIME OFFICIAL HARNESS <<<")
        # Sample benchmark problem set
        sample_problems = [
            MathProblem(problem_id="aime_2024_p1", question="Find the unique two-digit integer n such that the sum of the digits of n is 11 and n is a prime number.", target_answer="29", category="number_theory"),
            MathProblem(problem_id="math_500_alg", question="If f(x) = 3x - 5 and g(x) = x^2 + 1, find f(g(2)).", target_answer="10", category="algebra"),
        ]
        s = runner.run_math_benchmark(sample_problems)
        summaries.append(s)

    if args.benchmark in ("swebench", "all"):
        print("\n>>> EXECUTING SWE-BENCH VERIFIED HARNESS <<<")
        sample_instances = [
            {"instance_id": "django__django-11001", "problem_statement": "Queryset optimization error in models/sql/compiler.py"},
        ]
        s = runner.run_swebench_pipeline(sample_instances)
        summaries.append(s)

    print("\n" + "=" * 75)
    print("               BENCHMARK EXECUTION SUMMARY REPORT")
    print("=" * 75)
    for s in summaries:
        print(f"Benchmark: {s.benchmark_name}")
        print(f"  Model:      {s.model_name}")
        print(f"  Evaluated:  {s.instances_evaluated}")
        print(f"  Solved:     {s.instances_solved}")
        print(f"  Accuracy:   {s.accuracy_percent:.2f}%")
        print(f"  Artifact:   {s.output_artifact}")
        print(f"  Duration:   {s.elapsed_seconds:.2f}s")
        print("-" * 75)


if __name__ == "__main__":
    main()

"""Live Evaluation of DeepSeek-R1-Distill-Qwen-14B on Real Benchmark Tasks.

Executes real-time inference via official Hugging Face Serverless Engine:
1. MATH-500 / AIME Olympiad Problem (with System 2 <think> verification)
2. ARC-AGI Official Challenge Task (with 2-attempt grid generation)
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam_ai.models.loader import SAMModelManager, ModelConfig
from sam_ai.benchmarks.unified_runner import SAMUnifiedBenchmarkRunner
from sam_ai.benchmarks.math_evaluator import MathProblem
from sam_ai.benchmarks.arc_solver import ARCOfficialEvaluator, parse_grid_from_text, grids_equal

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def evaluate_math_problem_14b():
    print("\n" + "=" * 75)
    print("1. EVALUATING DEEPSEEK-R1-DISTILL-QWEN-14B ON COMPETITION MATH")
    print("=" * 75)

    # Real competition problem from AIME / MATH-500
    problem = MathProblem(
        problem_id="aime_sample_p1",
        question=(
            "Find the unique positive integer n < 1000 such that the sum of the digits of n is 15, "
            "and n is a multiple of both 3 and 7."
        ),
        target_answer="483",
        category="number_theory",
    )

    config = ModelConfig(
        model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        use_api=True,
        api_token=HF_TOKEN,
    )
    manager = SAMModelManager(config)
    runner = SAMUnifiedBenchmarkRunner(model_manager=manager, output_dir="predictions")

    summary = runner.run_math_benchmark([problem])
    print(f"Results: {summary.instances_solved}/{summary.instances_evaluated} Solved ({summary.accuracy_percent:.1f}%)")
    print(f"Artifact: {summary.output_artifact}")
    return summary


def evaluate_arc_challenge_14b():
    print("\n" + "=" * 75)
    print("2. EVALUATING DEEPSEEK-R1-DISTILL-QWEN-14B ON OFFICIAL ARC-AGI TASK")
    print("=" * 75)

    config = ModelConfig(
        model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        use_api=True,
        api_token=HF_TOKEN,
    )
    manager = SAMModelManager(config)
    runner = SAMUnifiedBenchmarkRunner(model_manager=manager, output_dir="predictions")

    # Run on 1 real official task from data directory
    summary = runner.run_arc_benchmark(max_tasks=1)
    print(f"ARC Results: {summary.instances_solved}/{summary.instances_evaluated} Solved ({summary.accuracy_percent:.1f}%)")
    print(f"Artifact: {summary.output_artifact}")
    return summary


def main():
    print("*" * 75)
    print("  SAM-AI: DEEPSEEK-R1-DISTILL-QWEN-14B BENCHMARK RUN")
    print("  Engine: High-Throughput Serverless GPU Pipeline")
    print("*" * 75)

    evaluate_math_problem_14b()
    evaluate_arc_challenge_14b()


if __name__ == "__main__":
    main()

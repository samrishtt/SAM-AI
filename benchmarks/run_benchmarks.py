"""Ablation Experiment Runner and Scientific Evaluation Harness.

Executes controlled empirical ablations comparing:
1. System 1 Baseline (Greedy intuitive proposal, no search)
2. System 2 Pure Search (PUCT MCTS, without memory consolidation)
3. Full Micro-AGI Architecture (System 2 PUCT MCTS + CLS Memory Consolidation)

Generates publication-quality LaTeX and Markdown comparison tables.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure micro_agi root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engine import CognitiveEngine, CognitiveCycleResult
from benchmarks.arc_agi_suite import get_standard_arc_tasks, ARCEvaluator, ARCTask
from benchmarks.symbolic_deduction import get_standard_deduction_problems, DeductionEvaluator, DeductionProblem


def run_arc_experiments(engine: CognitiveEngine) -> Tuple[float, float, float]:
    """Runs ARC tasks on engine. Returns (accuracy, avg_nodes, avg_time_ms)."""
    tasks = get_standard_arc_tasks()
    evaluator = ARCEvaluator()
    solved = 0
    total_nodes = 0
    total_time_ms = 0.0

    for task in tasks:
        context = {
            "task_id": task.id,
            "train_pairs": str(task.train_pairs),
            "test_input": str(task.test_pair.input_grid),
        }
        res: CognitiveCycleResult = engine.solve_task(
            goal=f"Synthesize transformation rule for ARC task {task.name}",
            initial_context=context,
            allowed_actions=task.candidate_hypotheses,
            max_steps=5,
            domain="arc",
        )
        if evaluator.evaluate_task(str(res.solution or ""), task):
            solved += 1

        total_nodes += sum(t.search_nodes for t in res.traces)
        total_time_ms += res.elapsed_time_sec * 1000.0

    acc = (solved / len(tasks)) * 100.0
    avg_nodes = total_nodes / len(tasks)
    avg_time = total_time_ms / len(tasks)
    return acc, avg_nodes, avg_time


def run_deduction_experiments(engine: CognitiveEngine) -> Tuple[float, float, float]:
    """Runs Deduction problems on engine. Returns (accuracy, avg_nodes, avg_time_ms)."""
    problems = get_standard_deduction_problems()
    evaluator = DeductionEvaluator()
    solved = 0
    total_nodes = 0
    total_time_ms = 0.0

    for prob in problems:
        context = {
            "problem_id": prob.id,
            "premises": " | ".join(prob.premises),
            "query": prob.query,
        }
        res: CognitiveCycleResult = engine.solve_task(
            goal=f"Deduce conclusion for: {prob.query}",
            initial_context=context,
            allowed_actions=prob.allowed_reasoning_steps,
            max_steps=5,
            domain="deduction",
        )
        if evaluator.evaluate(str(res.solution or ""), prob):
            solved += 1

        total_nodes += sum(t.search_nodes for t in res.traces)
        total_time_ms += res.elapsed_time_sec * 1000.0

    acc = (solved / len(problems)) * 100.0
    avg_nodes = total_nodes / len(problems)
    avg_time = total_time_ms / len(problems)
    return acc, avg_nodes, avg_time


def run_full_ablation_suite() -> str:
    """Executes the full matrix of architectural ablations."""
    print("=" * 78)
    print(" MICRO-AGI EMPIRICAL ABLATION EXPERIMENTS")
    print(" Benchmarking Dual-Process Search & Complementary Learning Systems")
    print("=" * 78)

    configurations = [
        ("System 1 Baseline (Greedy Prior)", False, 1, False),
        ("System 2 Pure Search (MCTS, N=15)", True, 15, False),
        ("Full Micro-AGI (MCTS N=30 + CLS Memory)", True, 30, True),
    ]

    results_table = []

    for name, s2_enabled, sims, mem_consol in configurations:
        print(f"\n[RUNNING] Configuration: {name}...")
        engine = CognitiveEngine(
            system2_enabled=s2_enabled,
            mcts_simulations=sims,
            auto_consolidate=mem_consol,
        )

        arc_acc, arc_nodes, arc_time = run_arc_experiments(engine)
        ded_acc, ded_nodes, ded_time = run_deduction_experiments(engine)

        overall_acc = (arc_acc + ded_acc) / 2.0
        results_table.append({
            "name": name,
            "arc_acc": arc_acc,
            "ded_acc": ded_acc,
            "overall_acc": overall_acc,
            "avg_nodes": (arc_nodes + ded_nodes) / 2.0,
            "avg_time_ms": (arc_time + ded_time) / 2.0,
        })

    # Render Markdown / ASCII Table
    output_lines = [
        "\n### Empirical Ablation Results Matrix",
        "",
        "| Architecture Configuration | ARC-AGI Acc (%) | Deduction Acc (%) | Overall Acc (%) | Avg Tree Nodes | Latency (ms) |",
        "|:---------------------------|:---------------:|:-----------------:|:---------------:|:--------------:|:------------:|",
    ]

    for row in results_table:
        line = (
            f"| {row['name']:<26} | "
            f"{row['arc_acc']:>14.1f}% | "
            f"{row['ded_acc']:>16.1f}% | "
            f"{row['overall_acc']:>14.1f}% | "
            f"{row['avg_nodes']:>14.1f} | "
            f"{row['avg_time_ms']:>11.2f} |"
        )
        output_lines.append(line)

    markdown_report = "\n".join(output_lines)
    print(markdown_report)
    print("\n" + "=" * 78)
    return markdown_report


if __name__ == "__main__":
    report = run_full_ablation_suite()
    with open("BENCHMARK_RESULTS.md", "w") as f:
        f.write("# Empirical Benchmark Results: Micro-AGI\n\n" + report + "\n")
    print("Benchmark results saved to BENCHMARK_RESULTS.md")

"""Unified Benchmark Pipeline Wiring DeepSeek-R1-Distill-Qwen-14B to Official Harnesses.

Orchestrates official evaluation across:
1. ARC-AGI / ARC Prize (Official 2-attempt visual logic harness)
2. MATH-500 / AIME 2024 (Olympiad mathematical reasoning)
3. SWE-bench Verified (Official repository patch generation)

Connects:
SAMModelManager -> Tokenizer -> Prompt Construction -> Model Inference / MCTS Search
-> Response Extraction -> Verification & Official Artifact Formatting.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch

from sam_ai.models.loader import SAMModelManager, ModelConfig
from sam_ai.benchmarks.arc_solver import ARCOfficialEvaluator, ARCTask, parse_grid_from_text
from sam_ai.benchmarks.math_evaluator import MathOfficialEvaluator, MathProblem, extract_boxed_answer
from sam_ai.leaderboard.swebench_runner import SWEBenchLeaderboardRunner


@dataclass
class BenchmarkRunSummary:
    benchmark_name: str
    model_name: str
    instances_evaluated: int
    instances_solved: int
    accuracy_percent: float
    output_artifact: str
    elapsed_seconds: float


class SAMUnifiedBenchmarkRunner:
    """
    Connects DeepSeek-R1-Distill-Qwen-14B to all official evaluation harnesses.
    """

    def __init__(
        self,
        model_manager: Optional[SAMModelManager] = None,
        output_dir: str = "predictions",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager or SAMModelManager(
            ModelConfig(model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        )

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        custom_generator: Optional[Callable[[str], str]] = None,
    ) -> str:
        """
        Generates completion using the configured model manager or custom generator callable.
        """
        if custom_generator is not None:
            return custom_generator(prompt)

        return self.model_manager.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def run_arc_benchmark(
        self,
        max_tasks: Optional[int] = None,
        generator_fn: Optional[Callable[[str], str]] = None,
    ) -> BenchmarkRunSummary:
        """
        Executes the official ARC Prize evaluation on official challenge tasks.
        Uses 2 attempts per task strictly matching official competition rules.
        """
        start_time = time.perf_counter()
        evaluator = ARCOfficialEvaluator()
        tasks = evaluator.load_all_tasks()
        if max_tasks is not None:
            tasks = tasks[:max_tasks]

        print(f"[*] Running ARC-AGI Benchmark over {len(tasks)} tasks using {self.model_manager.config.model_name_or_path}...")
        predictions: Dict[str, List[Dict[str, List[List[int]]]]] = {}

        for task in tasks:
            task_attempts: List[Dict[str, List[List[int]]]] = []
            for test_idx in range(len(task.test_inputs)):
                prompt = evaluator.construct_prompt(task, test_idx=test_idx)

                # Attempt 1 (Standard temperature)
                raw_out_1 = self.generate_completion(
                    prompt,
                    temperature=0.6,
                    custom_generator=generator_fn,
                )
                grid_1 = parse_grid_from_text(raw_out_1) or [[0]]

                # Attempt 2 (Higher exploration temperature)
                raw_out_2 = self.generate_completion(
                    prompt,
                    temperature=0.85,
                    custom_generator=generator_fn,
                )
                grid_2 = parse_grid_from_text(raw_out_2) or grid_1

                task_attempts.append({
                    "attempt_1": grid_1,
                    "attempt_2": grid_2,
                })

            predictions[task.task_id] = task_attempts

        # Grade predictions with official 2-attempt rule
        eval_report = evaluator.evaluate_predictions(tasks, predictions)
        artifact_file = evaluator.format_submission(
            predictions,
            str(self.output_dir / "arc_prize_submission.json")
        )

        elapsed = time.perf_counter() - start_time
        return BenchmarkRunSummary(
            benchmark_name="ARC-AGI (Official ARC Prize)",
            model_name=self.model_manager.config.model_name_or_path,
            instances_evaluated=eval_report["total_test_instances"],
            instances_solved=eval_report["solved_instances"],
            accuracy_percent=eval_report["accuracy_percent"],
            output_artifact=str(artifact_file),
            elapsed_seconds=elapsed,
        )

    def run_math_benchmark(
        self,
        problems: List[MathProblem],
        generator_fn: Optional[Callable[[str], str]] = None,
    ) -> BenchmarkRunSummary:
        """
        Executes MATH-500 / AIME competition mathematics benchmark.
        """
        start_time = time.perf_counter()
        evaluator = MathOfficialEvaluator()
        print(f"[*] Running Math Benchmark over {len(problems)} problems...")

        model_outputs = {}
        for p in problems:
            prompt = evaluator.construct_prompt(p)
            out = self.generate_completion(
                prompt,
                max_new_tokens=1024,
                temperature=0.6,
                custom_generator=generator_fn,
            )
            model_outputs[p.problem_id] = out

        eval_report = evaluator.evaluate_predictions(problems, model_outputs)
        out_file = self.output_dir / "math_benchmark_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        elapsed = time.perf_counter() - start_time
        return BenchmarkRunSummary(
            benchmark_name="MATH-500 / AIME Competition Math",
            model_name=self.model_manager.config.model_name_or_path,
            instances_evaluated=eval_report["total_problems"],
            instances_solved=eval_report["solved_problems"],
            accuracy_percent=eval_report["accuracy_percent"],
            output_artifact=str(out_file),
            elapsed_seconds=elapsed,
        )

    def run_swebench_pipeline(
        self,
        instances: List[Dict[str, Any]],
        generator_fn: Optional[Callable[[str], str]] = None,
    ) -> BenchmarkRunSummary:
        """
        Generates official SWE-bench Verified submission file (all_preds.jsonl).
        """
        start_time = time.perf_counter()
        runner = SWEBenchLeaderboardRunner(output_dir=str(self.output_dir))
        predictions = []

        for inst in instances:
            inst_id = inst["instance_id"]
            problem_stmt = inst.get("problem_statement", "")
            prompt = (
                f"Repository Issue for {inst_id}:\n{problem_stmt}\n\n"
                "Generate the git diff patch fixing this issue inside a ```diff code block."
            )
            raw_patch = self.generate_completion(
                prompt,
                max_new_tokens=1024,
                custom_generator=generator_fn,
            )
            predictions.append(
                runner.format_prediction(
                    instance_id=inst_id,
                    patch_content=raw_patch,
                    model_id=self.model_manager.config.model_name_or_path,
                )
            )

        artifact_file = runner.write_predictions_jsonl(predictions)
        elapsed = time.perf_counter() - start_time

        return BenchmarkRunSummary(
            benchmark_name="SWE-bench Verified (Official Submissions)",
            model_name=self.model_manager.config.model_name_or_path,
            instances_evaluated=len(instances),
            instances_solved=0,  # Solved count determined by Princeton Docker runner
            accuracy_percent=0.0,
            output_artifact=str(artifact_file),
            elapsed_seconds=elapsed,
        )

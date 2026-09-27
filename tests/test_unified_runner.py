"""Unit and integration tests for SAMUnifiedBenchmarkRunner and MathOfficialEvaluator."""

import json
from pathlib import Path
import pytest

from sam_ai.benchmarks.unified_runner import SAMUnifiedBenchmarkRunner
from sam_ai.benchmarks.math_evaluator import (
    MathProblem,
    MathOfficialEvaluator,
    extract_boxed_answer,
    math_answers_equal,
)
from sam_ai.models.loader import SAMModelManager, ModelConfig


def test_boxed_answer_extraction():
    # Simple boxed
    text1 = "Therefore, the value of x is \\boxed{42}."
    assert extract_boxed_answer(text1) == "42"

    # Nested braces
    text2 = "Thus, the fraction is \\boxed{\\frac{7}{12}}."
    assert extract_boxed_answer(text2) == "\\frac{7}{12}"

    # AIME integer formatting
    text3 = "The final three-digit integer is \\boxed{045}."
    assert extract_boxed_answer(text3) == "045"


def test_math_answer_equivalence():
    assert math_answers_equal("42", "42")
    assert math_answers_equal("045", "45")  # AIME leading zero
    assert math_answers_equal("$1/2$", "1/2")
    assert math_answers_equal("0.5", "1/2")
    assert not math_answers_equal("12", "15")


def test_unified_runner_arc_pipeline_end_to_end(tmp_path):
    runner = SAMUnifiedBenchmarkRunner(
        model_manager=SAMModelManager(ModelConfig(model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")),
        output_dir=str(tmp_path),
    )

    # Mock generator simulating model output
    def mock_generator(prompt: str) -> str:
        return "<think> The grid expands </think>\n```json\n[[8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4]]\n```"

    summary = runner.run_arc_benchmark(max_tasks=2, generator_fn=mock_generator)
    assert summary.instances_evaluated == 2
    assert summary.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    assert Path(summary.output_artifact).exists()

    with open(summary.output_artifact, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2


def test_unified_runner_math_pipeline_end_to_end(tmp_path):
    runner = SAMUnifiedBenchmarkRunner(
        model_manager=SAMModelManager(ModelConfig(model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")),
        output_dir=str(tmp_path),
    )

    problems = [
        MathProblem(problem_id="p1", question="Find x: 2x + 10 = 20", target_answer="5"),
        MathProblem(problem_id="p2", question="Find 3^3", target_answer="27"),
    ]

    def mock_generator(prompt: str) -> str:
        if "2x + 10" in prompt:
            return "<think> 2x = 10 -> x = 5 </think> \\boxed{5}"
        return "<think> 3*3*3 = 27 </think> \\boxed{27}"

    summary = runner.run_math_benchmark(problems=problems, generator_fn=mock_generator)
    assert summary.instances_evaluated == 2
    assert summary.instances_solved == 2
    assert summary.accuracy_percent == 100.0
    assert Path(summary.output_artifact).exists()


def test_unified_runner_swebench_pipeline_end_to_end(tmp_path):
    runner = SAMUnifiedBenchmarkRunner(
        model_manager=SAMModelManager(ModelConfig(model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")),
        output_dir=str(tmp_path),
    )

    instances = [
        {"instance_id": "django__django-11001", "problem_statement": "Fix query optimizer bug"},
    ]

    def mock_generator(prompt: str) -> str:
        return "diff --git a/django/db.py b/django/db.py\n--- a/django/db.py\n+++ b/django/db.py\n@@ -1 +1 @@\n-old\n+new"

    summary = runner.run_swebench_pipeline(instances=instances, generator_fn=mock_generator)
    assert summary.instances_evaluated == 1
    assert Path(summary.output_artifact).exists()

    with open(summary.output_artifact, "r", encoding="utf-8") as f:
        line = f.readline()
        pred = json.loads(line)
        assert pred["instance_id"] == "django__django-11001"
        assert "diff --git" in pred["model_patch"]
        assert pred["model_name_or_path"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

"""Canonical Evaluation & Controlled Comparison Pipeline for SAM-AI.

Supports model-neutral evaluation across 6 distinct capability groups:
1. Conversation & instruction following
2. Mathematics & scientific reasoning with checkable answers
3. Coding tasks verified by execution and unit tests
4. Structured output & action formatting (JSON action schema)
5. Multi-step planning & tool selection
6. ARC visual logic puzzle transformation

Enforces:
- Exact telemetry per task: task_id, model_id, revision, generation_config, raw_output, reasoning_trace,
  verifier_status, duration_ms, failure_reason.
- Controlled comparison between:
  * Unmodified Qwen/Qwen3.8-27B
  * Candidate LoRA adapter checkpoint
  * Legacy DeepSeek-R1-Distill-Qwen-14B baseline
"""

from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from sam_ai.substrate.interfaces import (
    ModelRunner,
    ModelOutput,
    BenchmarkTask,
    Verifier,
    VerificationResult,
    VerificationStatus,
    FailureCategory,
)
from sam_ai.substrate.verifiers import (
    MathDomainVerifier,
    CodeDomainVerifier,
    ArcDomainVerifier,
    ScienceDomainVerifier,
    AgentDomainVerifier,
)
from sam_ai.models.loader import PRIMARY_MODEL_ID, BASELINE_MODEL_ID


@dataclass
class TaskEvaluationRecord:
    """Individual task evaluation telemetry record."""
    task_id: str
    group_name: str
    model_id: str
    model_revision: Optional[str]
    generation_config: Dict[str, Any]
    raw_output: str
    reasoning_trace: Optional[str]
    verifier_status: str  # PASS, FAIL, ERROR, TIMEOUT, NOT_RUN
    verifier_details: str
    duration_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupEvaluationSummary:
    """Summary metrics for a specific task group."""
    group_name: str
    model_id: str
    total_tasks: int
    pass_count: int
    fail_count: int
    error_count: int
    timeout_count: int
    accuracy_percent: float
    avg_latency_ms: float
    total_tokens: int
    records: List[TaskEvaluationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["records"] = [r.to_dict() for r in self.records]
        return d


@dataclass
class FullEvaluationReport:
    """Comprehensive evaluation report across all capability task groups."""
    model_id: str
    model_revision: Optional[str]
    timestamp: str
    total_tasks: int
    total_passed: int
    overall_accuracy_percent: float
    group_summaries: Dict[str, GroupEvaluationSummary] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "timestamp": self.timestamp,
            "total_tasks": self.total_tasks,
            "total_passed": self.total_passed,
            "overall_accuracy_percent": round(self.overall_accuracy_percent, 2),
            "group_summaries": {k: v.to_dict() for k, v in self.group_summaries.items()},
            "metadata": self.metadata,
        }


# ==============================================================================
# Standardized Built-In Task Sets
# ==============================================================================
def get_standard_task_suites() -> Dict[str, List[Dict[str, Any]]]:
    """Returns verified task suites across the 6 core capability groups."""
    return {
        "conversation_instruction": [
            {
                "task_id": "conv_01_json_keys",
                "prompt": "Provide a summary of photosythesis. Format your output strictly as a JSON object with keys 'summary' and 'key_reactants'.",
                "verifier_type": "json_schema",
                "expected_keys": ["summary", "key_reactants"],
            },
            {
                "task_id": "conv_02_constraint_length",
                "prompt": "Explain gradient descent in exactly 3 bullet points, each under 20 words.",
                "verifier_type": "bullet_count",
                "expected_bullets": 3,
            },
        ],
        "mathematics_scientific": [
            {
                "task_id": "math_01_quadratic",
                "prompt": "Solve for x: 3*x - 15 = 0. State your final answer as \\boxed{x}.",
                "verifier_type": "math",
                "ground_truth": "5",
            },
            {
                "task_id": "math_02_modular",
                "prompt": "What is the remainder when 45 is divided by 7? Output \\boxed{remainder}.",
                "verifier_type": "math",
                "ground_truth": "3",
            },
            {
                "task_id": "sci_01_kinetic_energy",
                "prompt": "Calculate the kinetic energy in Joules of a 2 kg object moving at 3 m/s. Formula: 0.5 * m * v^2. Output \\boxed{energy}.",
                "verifier_type": "science",
                "ground_truth": 9.0,
            },
        ],
        "coding_verified": [
            {
                "task_id": "code_01_palindrome",
                "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if s is a palindrome ignoring case and spaces.",
                "verifier_type": "code",
                "test_code": (
                    "assert is_palindrome('Race car') == True\n"
                    "assert is_palindrome('hello') == False\n"
                    "assert is_palindrome('') == True\n"
                ),
            },
            {
                "task_id": "code_02_factorial",
                "prompt": "Write a Python function `factorial(n: int) -> int`.",
                "verifier_type": "code",
                "test_code": (
                    "assert factorial(0) == 1\n"
                    "assert factorial(5) == 120\n"
                    "assert factorial(3) == 6\n"
                ),
            },
        ],
        "structured_action_format": [
            {
                "task_id": "action_01_move_right",
                "prompt": "You are navigating a grid. Output a valid action JSON to move right. Legal actions are ACTION1 (up), ACTION2 (down), ACTION3 (left), ACTION4 (right). Output: {'action': '<NAME>'}",
                "verifier_type": "action_schema",
                "expected_action": "ACTION4",
            },
            {
                "task_id": "action_02_click_cell",
                "prompt": "Click coordinate (12, 24). Output strict JSON: {'action': 'ACTION6', 'data': {'x': 12, 'y': 24}}",
                "verifier_type": "action_schema",
                "expected_action": "ACTION6",
                "expected_data": {"x": 12, "y": 24},
            },
        ],
        "planning_tool_selection": [
            {
                "task_id": "plan_01_tool_ordering",
                "prompt": "You need to analyze an unknown code file. Step 1: list files. Step 2: read file. Step 3: run tests. Output JSON: {'plan': ['list_dir', 'read_file', 'run_tests']}",
                "verifier_type": "json_schema",
                "expected_keys": ["plan"],
            },
        ],
        "arc_puzzle_transformation": [
            {
                "task_id": "arc_01_color_inversion",
                "prompt": "Invert this 2x2 grid where 0 becomes 1 and 1 becomes 0: [[0, 1], [1, 0]]. Output JSON 2D grid.",
                "verifier_type": "arc_grid",
                "expected_grid": [[1, 0], [0, 1]],
            },
        ],
    }


# ==============================================================================
# Universal Evaluation Runner
# ==============================================================================
class CapabilityEvaluationRunner:
    """Executes capability evaluations with explicit verifiers and detailed telemetry."""

    def __init__(
        self,
        model_runner: ModelRunner,
        task_suites: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        output_dir: str = "evidence/evaluations",
    ):
        self.model_runner = model_runner
        self.task_suites = task_suites or get_standard_task_suites()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Domain verifiers
        self.math_verifier = MathDomainVerifier()
        self.code_verifier = CodeDomainVerifier()
        self.arc_verifier = ArcDomainVerifier()
        self.science_verifier = ScienceDomainVerifier()

    def _verify_task(self, task: Dict[str, Any], model_output: ModelOutput) -> Tuple[VerificationStatus, str]:
        """Runs the appropriate domain verifier on the model output."""
        v_type = task.get("verifier_type", "json_schema")
        text = model_output.text.strip()

        if v_type == "math":
            candidate = text
            if "\\boxed{" in text:
                candidate = text.split("\\boxed{", 1)[1].split("}", 1)[0]
            res = self.math_verifier.verify(candidate, task["ground_truth"])
            return res.status, res.details

        elif v_type == "science":
            candidate = text
            if "\\boxed{" in text:
                candidate = text.split("\\boxed{", 1)[1].split("}", 1)[0]
            res = self.science_verifier.verify(candidate, task["ground_truth"])
            return res.status, res.details

        elif v_type == "code":
            # Extract python code block if present
            code = text
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0].strip()
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0].strip()

            res = self.code_verifier.verify(code, task["test_code"])
            return res.status, res.details

        elif v_type == "action_schema":
            try:
                # Extract json substring
                json_str = text
                if "{" in json_str and "}" in json_str:
                    s_idx = json_str.find("{")
                    e_idx = json_str.rfind("}")
                    json_str = json_str[s_idx:e_idx + 1]
                data = json.loads(json_str)
                expected_action = task.get("expected_action")
                if expected_action and data.get("action") != expected_action:
                    return VerificationStatus.FAIL, f"Action '{data.get('action')}' != expected '{expected_action}'"
                expected_data = task.get("expected_data")
                if expected_data and data.get("data") != expected_data:
                    return VerificationStatus.FAIL, f"Data '{data.get('data')}' != expected '{expected_data}'"
                return VerificationStatus.PASS, "Valid action schema satisfied."
            except Exception as e:
                return VerificationStatus.FAIL, f"JSON parse failure: {e}"

        elif v_type == "arc_grid":
            try:
                json_str = text
                if "[" in json_str and "]" in json_str:
                    s_idx = json_str.find("[")
                    e_idx = json_str.rfind("]")
                    json_str = json_str[s_idx:e_idx + 1]
                grid = json.loads(json_str)
                if grid == task["expected_grid"]:
                    return VerificationStatus.PASS, "Grid matches expected output exactly."
                return VerificationStatus.FAIL, f"Grid {grid} != {task['expected_grid']}"
            except Exception as e:
                return VerificationStatus.FAIL, f"Grid parse failure: {e}"

        elif v_type == "json_schema":
            try:
                json_str = text
                if "{" in json_str and "}" in json_str:
                    s_idx = json_str.find("{")
                    e_idx = json_str.rfind("}")
                    json_str = json_str[s_idx:e_idx + 1]
                data = json.loads(json_str)
                for k in task.get("expected_keys", []):
                    if k not in data:
                        return VerificationStatus.FAIL, f"Missing required key '{k}' in output JSON."
                return VerificationStatus.PASS, "All required JSON keys present."
            except Exception as e:
                return VerificationStatus.FAIL, f"Failed to parse JSON: {e}"

        elif v_type == "bullet_count":
            lines = [line.strip() for line in text.split("\n") if line.strip().startswith(("-", "*", "1.", "2.", "3."))]
            if len(lines) == task.get("expected_bullets", 3):
                return VerificationStatus.PASS, f"Found exactly {len(lines)} bullets as required."
            return VerificationStatus.FAIL, f"Found {len(lines)} bullets, expected {task.get('expected_bullets')}"

        return VerificationStatus.NOT_RUN, "Unknown verifier type"

    def run_evaluation(
        self,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
    ) -> FullEvaluationReport:
        """Executes full evaluation suite across all groups."""
        gen_cfg = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }

        total_tasks = 0
        total_passed = 0
        group_summaries: Dict[str, GroupEvaluationSummary] = {}

        for group_name, tasks in self.task_suites.items():
            records: List[TaskEvaluationRecord] = []
            pass_cnt = 0
            fail_cnt = 0
            err_cnt = 0
            timeout_cnt = 0
            total_lat = 0.0
            total_toks = 0

            for task in tasks:
                t0 = time.perf_counter()
                try:
                    output = self.model_runner.generate(
                        prompt=task["prompt"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    v_status, v_details = self._verify_task(task, output)
                except Exception as e:
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    output = ModelOutput(text="")
                    v_status = VerificationStatus.ERROR
                    v_details = str(e)

                rec = TaskEvaluationRecord(
                    task_id=task["task_id"],
                    group_name=group_name,
                    model_id=self.model_runner.model_name,
                    model_revision=None,
                    generation_config=gen_cfg,
                    raw_output=output.text,
                    reasoning_trace=output.reasoning_trace,
                    verifier_status=v_status.value if hasattr(v_status, "value") else str(v_status),
                    verifier_details=v_details,
                    duration_ms=round(lat_ms, 2),
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    failure_reason=None if v_status == VerificationStatus.PASS else v_details,
                )
                records.append(rec)

                if v_status == VerificationStatus.PASS:
                    pass_cnt += 1
                elif v_status == VerificationStatus.FAIL:
                    fail_cnt += 1
                elif v_status == VerificationStatus.TIMEOUT:
                    timeout_cnt += 1
                else:
                    err_cnt += 1

                total_lat += lat_ms
                total_toks += output.prompt_tokens + output.completion_tokens

            grp_total = len(tasks)
            grp_acc = (pass_cnt / grp_total * 100.0) if grp_total > 0 else 0.0
            avg_lat = (total_lat / grp_total) if grp_total > 0 else 0.0

            group_summaries[group_name] = GroupEvaluationSummary(
                group_name=group_name,
                model_id=self.model_runner.model_name,
                total_tasks=grp_total,
                pass_count=pass_cnt,
                fail_count=fail_cnt,
                error_count=err_cnt,
                timeout_count=timeout_cnt,
                accuracy_percent=round(grp_acc, 2),
                avg_latency_ms=round(avg_lat, 2),
                total_tokens=total_toks,
                records=records,
            )

            total_tasks += grp_total
            total_passed += pass_cnt

        overall_acc = (total_passed / total_tasks * 100.0) if total_tasks > 0 else 0.0
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        report = FullEvaluationReport(
            model_id=self.model_runner.model_name,
            model_revision=None,
            timestamp=timestamp_str,
            total_tasks=total_tasks,
            total_passed=total_passed,
            overall_accuracy_percent=overall_acc,
            group_summaries=group_summaries,
            metadata={"evaluation_pipeline": "SAM-AI-CapabilityPipeline-v1"},
        )

        # Save report artifact
        clean_model_name = self.model_runner.model_name.replace("/", "_").replace(":", "_")
        artifact_file = self.output_dir / f"eval_report_{clean_model_name}_{int(time.time())}.json"
        with open(artifact_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report


# ==============================================================================
# Controlled Comparison Suite
# ==============================================================================
class ControlledComparisonSuite:
    """Executes controlled comparisons between base, candidate, and legacy models."""

    def __init__(
        self,
        primary_runner: ModelRunner,
        baseline_runner: Optional[ModelRunner] = None,
        candidate_adapted_runner: Optional[ModelRunner] = None,
        task_suites: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        output_dir: str = "evidence/comparisons",
    ):
        self.primary_runner = primary_runner
        self.baseline_runner = baseline_runner
        self.candidate_adapted_runner = candidate_adapted_runner
        self.task_suites = task_suites or get_standard_task_suites()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_comparison(self) -> Dict[str, Any]:
        """Runs controlled evaluation across configured runners on identical task sets."""
        results = {}

        # 1. Primary Model Evaluation (Qwen/Qwen3.8-27B)
        p_evaluator = CapabilityEvaluationRunner(
            self.primary_runner,
            task_suites=self.task_suites,
            output_dir=str(self.output_dir),
        )
        p_report = p_evaluator.run_evaluation()
        results["primary_model"] = p_report.to_dict()

        # 2. Candidate Adapted Model (if provided)
        if self.candidate_adapted_runner:
            c_evaluator = CapabilityEvaluationRunner(
                self.candidate_adapted_runner,
                task_suites=self.task_suites,
                output_dir=str(self.output_dir),
            )
            c_report = c_evaluator.run_evaluation()
            results["candidate_adapted"] = c_report.to_dict()

            # Compute regressions and improvements
            improvements = []
            regressions = []
            for g_name, p_grp in p_report.group_summaries.items():
                c_grp = c_report.group_summaries.get(g_name)
                if not c_grp:
                    continue
                p_task_map = {r.task_id: r.verifier_status for r in p_grp.records}
                c_task_map = {r.task_id: r.verifier_status for r in c_grp.records}
                for t_id, p_stat in p_task_map.items():
                    c_stat = c_task_map.get(t_id)
                    if p_stat != "PASS" and c_stat == "PASS":
                        improvements.append(t_id)
                    elif p_stat == "PASS" and c_stat != "PASS":
                        regressions.append(t_id)

            results["delta"] = {
                "accuracy_change": round(c_report.overall_accuracy_percent - p_report.overall_accuracy_percent, 2),
                "improved_tasks": improvements,
                "regressed_tasks": regressions,
                "adapter_retained": len(improvements) > len(regressions) and len(regressions) == 0,
            }

        # 3. Legacy Baseline Model (if provided)
        if self.baseline_runner:
            b_evaluator = CapabilityEvaluationRunner(
                self.baseline_runner,
                task_suites=self.task_suites,
                output_dir=str(self.output_dir),
            )
            b_report = b_evaluator.run_evaluation()
            results["baseline_model"] = b_report.to_dict()

        comparison_file = self.output_dir / f"controlled_comparison_{int(time.time())}.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

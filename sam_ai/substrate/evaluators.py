#!/usr/bin/env python3
"""
SAM-AI Domain Evaluator (Phase 5)
=================================
Executes real held-out evaluation datasets across reasoning, coding, math, science, and agents.

Contracts:
- Zero simulated benchmarks (no mock model outputs, no success = True)
- Purely reports empirical verifier outcomes: PASS, FAIL, NOT_RUN, ERROR, TIMEOUT
- Computes:
  - Accuracy (pass rate)
  - Verification rate
  - Average latency
  - Total tokens & compute cost
  - Failure taxonomy distribution
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from sam_ai.substrate.interfaces import (
    BenchmarkTask,
    VerificationStatus,
    FailureCategory,
)
from sam_ai.substrate.engine import CognitiveEngine, ExecutionRecord


@dataclass
class EvaluationReport:
    """Summary metrics of an evaluation run on a benchmark suite."""
    domain: str
    benchmark_name: str
    model_name: str
    strategy_name: str
    total_tasks: int
    pass_count: int
    fail_count: int
    error_count: int
    timeout_count: int
    not_run_count: int
    accuracy: float
    verification_rate: float
    avg_latency_ms: float
    total_tokens: int
    failure_distribution: Dict[str, int] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "strategy_name": self.strategy_name,
            "total_tasks": self.total_tasks,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "not_run_count": self.not_run_count,
            "accuracy": round(self.accuracy, 4),
            "verification_rate": round(self.verification_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "failure_distribution": self.failure_distribution,
            "timestamp": self.timestamp,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Benchmark Evaluation Report: {self.benchmark_name}",
            f"- **Domain**: `{self.domain}`",
            f"- **Model**: `{self.model_name}`",
            f"- **Strategy**: `{self.strategy_name}`",
            f"- **Total Tasks**: {self.total_tasks}",
            f"- **Pass**: {self.pass_count} ({self.accuracy * 100:.2f}%)",
            f"- **Fail**: {self.fail_count}",
            f"- **Error**: {self.error_count}",
            f"- **Timeout**: {self.timeout_count}",
            f"- **Not Run**: {self.not_run_count}",
            f"- **Verification Rate**: {self.verification_rate * 100:.2f}%",
            f"- **Avg Latency**: {self.avg_latency_ms:.2f} ms",
            f"- **Total Tokens**: {self.total_tokens}",
            "",
            "### Failure Category Distribution",
        ]
        if self.failure_distribution:
            for cat, cnt in sorted(self.failure_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"- **{cat}**: {cnt}")
        else:
            lines.append("- *(No failures recorded)*")
        return "\n".join(lines)


class BenchmarkEvaluator:
    """Runs a suite of BenchmarkTasks through CognitiveEngine and generates structured reports."""

    def __init__(self, engine: CognitiveEngine):
        self.engine = engine

    def run_suite(
        self,
        suite_name: str,
        tasks: List[BenchmarkTask],
        search_budget: int = 1,
        save_report: bool = True,
        output_dir: str = "eval_reports",
    ) -> EvaluationReport:
        if not tasks:
            raise ValueError("Task suite is empty.")

        domain = tasks[0].domain
        pass_cnt = 0
        fail_cnt = 0
        err_cnt = 0
        timeout_cnt = 0
        not_run_cnt = 0
        total_tokens = 0
        total_latency = 0.0
        failure_dist: Dict[str, int] = {}
        records_summary = []

        for task in tasks:
            exec_rec = self.engine.solve(task=task, search_budget=search_budget)
            status = exec_rec.verification.status

            if status == VerificationStatus.PASS:
                pass_cnt += 1
            elif status == VerificationStatus.FAIL:
                fail_cnt += 1
            elif status == VerificationStatus.ERROR:
                err_cnt += 1
            elif status == VerificationStatus.TIMEOUT:
                timeout_cnt += 1
            elif status == VerificationStatus.NOT_RUN:
                not_run_cnt += 1

            if exec_rec.failure:
                cat_val = exec_rec.failure.failure_category.value
                failure_dist[cat_val] = failure_dist.get(cat_val, 0) + 1

            total_tokens += exec_rec.total_tokens
            total_latency += exec_rec.latency_ms

            records_summary.append({
                "task_id": task.task_id,
                "status": status.value,
                "strategy": exec_rec.strategy_name,
                "latency_ms": exec_rec.latency_ms,
                "tokens": exec_rec.total_tokens,
                "details": exec_rec.verification.details,
            })

        total_tasks = len(tasks)
        accuracy = pass_cnt / total_tasks if total_tasks > 0 else 0.0
        v_rate = (pass_cnt + fail_cnt) / total_tasks if total_tasks > 0 else 0.0
        avg_latency = total_latency / total_tasks if total_tasks > 0 else 0.0

        report = EvaluationReport(
            domain=domain,
            benchmark_name=suite_name,
            model_name=self.engine.model.model_name,
            strategy_name=tasks[0].metadata.get("strategy_override", "adaptive_heuristic"),
            total_tasks=total_tasks,
            pass_count=pass_cnt,
            fail_count=fail_cnt,
            error_count=err_cnt,
            timeout_count=timeout_cnt,
            not_run_count=not_run_cnt,
            accuracy=accuracy,
            verification_rate=v_rate,
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens,
            failure_distribution=failure_dist,
            records=records_summary,
        )

        if save_report:
            os.makedirs(output_dir, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            fname_base = f"{domain}_{suite_name.replace(' ', '_').lower()}_{timestamp_str}"
            json_path = os.path.join(output_dir, f"{fname_base}.json")
            md_path = os.path.join(output_dir, f"{fname_base}.md")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())

        return report

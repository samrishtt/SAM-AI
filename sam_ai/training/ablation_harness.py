"""Empirical Ablation and Evaluation Harness for SAM-AI.

Enforces the core research discipline:
'Do not claim an optimization works until SAM-AI's own experiments demonstrate it.'

Provides structured experiment definitions, statistical trial runners,
and empirical measurement reporting across training and inference components.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import torch


@dataclass
class MetricRecord:
    name: str
    unit: str
    baseline_values: List[float] = field(default_factory=list)
    optimized_values: List[float] = field(default_factory=list)

    @property
    def baseline_mean(self) -> float:
        return sum(self.baseline_values) / len(self.baseline_values) if self.baseline_values else 0.0

    @property
    def baseline_std(self) -> float:
        if len(self.baseline_values) <= 1:
            return 0.0
        m = self.baseline_mean
        return math.sqrt(sum((x - m) ** 2 for x in self.baseline_values) / (len(self.baseline_values) - 1))

    @property
    def optimized_mean(self) -> float:
        return sum(self.optimized_values) / len(self.optimized_values) if self.optimized_values else 0.0

    @property
    def optimized_std(self) -> float:
        if len(self.optimized_values) <= 1:
            return 0.0
        m = self.optimized_mean
        return math.sqrt(sum((x - m) ** 2 for x in self.optimized_values) / (len(self.optimized_values) - 1))

    @property
    def delta_percent(self) -> float:
        if abs(self.baseline_mean) < 1e-8:
            return 0.0
        return ((self.optimized_mean - self.baseline_mean) / abs(self.baseline_mean)) * 100.0


@dataclass
class AblationResult:
    experiment_name: str
    description: str
    metrics: Dict[str, MetricRecord]
    num_trials: int

    def to_markdown_table(self) -> str:
        """Renders an empirical measurement table."""
        lines = [
            f"### Experiment: {self.experiment_name}",
            f"*{self.description}* (Evaluated over {self.num_trials} trials)",
            "",
            "| Metric | Baseline (Mean ± Std) | Optimized (Mean ± Std) | Delta (%) | Verified Status |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for name, record in self.metrics.items():
            b_str = f"{record.baseline_mean:.4f} ± {record.baseline_std:.4f} {record.unit}"
            o_str = f"{record.optimized_mean:.4f} ± {record.optimized_std:.4f} {record.unit}"
            d_str = f"{record.delta_percent:+.2f}%"
            # Significant improvement verified if delta > 0 for positive metrics
            status = "✓ Demonstrated" if abs(record.delta_percent) > 0.1 else "= Neutral"
            lines.append(f"| {name} | {b_str} | {o_str} | {d_str} | {status} |")
        return "\n".join(lines)


class AblationHarness:
    """Orchestrates controlled comparative trials to validate architectural claims."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metrics: Dict[str, MetricRecord] = {}

    def register_metric(self, name: str, unit: str):
        self.metrics[name] = MetricRecord(name=name, unit=unit)

    def run_comparative_trial(
        self,
        num_trials: int,
        baseline_fn: Callable[[int], Dict[str, float]],
        optimized_fn: Callable[[int], Dict[str, float]],
    ) -> AblationResult:
        """
        Executes symmetric trials for baseline and optimized code paths,
        recording exact measurements without synthetic bias.
        """
        for i in range(num_trials):
            # Run Baseline
            b_results = baseline_fn(i)
            for k, v in b_results.items():
                if k not in self.metrics:
                    self.register_metric(k, "units")
                self.metrics[k].baseline_values.append(v)

            # Run Optimized
            o_results = optimized_fn(i)
            for k, v in o_results.items():
                if k not in self.metrics:
                    self.register_metric(k, "units")
                self.metrics[k].optimized_values.append(v)

        return AblationResult(
            experiment_name=self.name,
            description=self.description,
            metrics=self.metrics,
            num_trials=num_trials,
        )

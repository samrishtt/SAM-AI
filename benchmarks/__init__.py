"""Benchmark Suites for Micro-AGI.

Exposes:
- get_standard_arc_tasks, ARCEvaluator, ARCTask, ARCPair
- get_standard_deduction_problems, DeductionEvaluator, DeductionProblem
"""

from benchmarks.arc_agi_suite import get_standard_arc_tasks, ARCEvaluator, ARCTask, ARCPair
from benchmarks.symbolic_deduction import get_standard_deduction_problems, DeductionEvaluator, DeductionProblem

__all__ = [
    "get_standard_arc_tasks",
    "ARCEvaluator",
    "ARCTask",
    "ARCPair",
    "get_standard_deduction_problems",
    "DeductionEvaluator",
    "DeductionProblem",
]

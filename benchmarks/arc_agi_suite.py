"""ARC-AGI Abstraction Benchmark Suite.

Implements François Chollet's Abstraction and Reasoning Corpus (ARC-AGI) grid
puzzles for testing few-shot inductive program synthesis and spatial reasoning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


@dataclass
class ARCPair:
    input_grid: List[List[int]]
    output_grid: List[List[int]]


@dataclass
class ARCTask:
    id: str
    name: str
    train_pairs: List[ARCPair]
    test_pair: ARCPair
    target_rule: str
    candidate_hypotheses: List[str]


def get_standard_arc_tasks() -> List[ARCTask]:
    """Returns a curated set of formal ARC-AGI grid transformation challenges."""
    tasks = []

    # Task 1: Horizontal Reflection Symmetry
    t1_train = [
        ARCPair([[1, 2], [0, 0]], [[2, 1], [0, 0]]),
        ARCPair([[3, 4], [5, 6]], [[4, 3], [6, 5]]),
    ]
    t1_test = ARCPair([[7, 8], [9, 0]], [[8, 7], [0, 9]])
    tasks.append(
        ARCTask(
            id="arc_001_h_symmetry",
            name="Horizontal Reflection Symmetry",
            train_pairs=t1_train,
            test_pair=t1_test,
            target_rule="apply_flip_horizontal()",
            candidate_hypotheses=[
                "apply_rot90()",
                "apply_flip_horizontal()",
                "apply_flip_vertical()",
                "apply_recolor(1, 2)",
                "conclude_solution(rule='apply_flip_horizontal()')",
            ],
        )
    )

    # Task 2: 90-Degree Orthogonal Rotation
    t2_train = [
        ARCPair([[1, 0], [0, 2]], [[0, 1], [2, 0]]),
        ARCPair([[3, 0], [4, 0]], [[4, 3], [0, 0]]),
    ]
    t2_test = ARCPair([[5, 6], [0, 7]], [[0, 5], [7, 6]])
    tasks.append(
        ARCTask(
            id="arc_002_rotation",
            name="90-Degree Clockwise Rotation",
            train_pairs=t2_train,
            test_pair=t2_test,
            target_rule="apply_rot90()",
            candidate_hypotheses=[
                "apply_flip_horizontal()",
                "apply_rot90()",
                "apply_rot180()",
                "conclude_solution(rule='apply_rot90()')",
            ],
        )
    )

    # Task 3: Color Inversion / Substitution
    t3_train = [
        ARCPair([[1, 1], [2, 2]], [[2, 2], [1, 1]]),
        ARCPair([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
    ]
    t3_test = ARCPair([[1, 2], [2, 1]], [[2, 1], [1, 2]])
    tasks.append(
        ARCTask(
            id="arc_003_recolor",
            name="Binary Color Swap",
            train_pairs=t3_train,
            test_pair=t3_test,
            target_rule="apply_color_swap(1, 2)",
            candidate_hypotheses=[
                "apply_rot90()",
                "apply_flip_vertical()",
                "apply_color_swap(1, 2)",
                "conclude_solution(rule='apply_color_swap(1, 2)')",
            ],
        )
    )

    # Task 4: Gravity Drop (Push non-zero pixels down)
    t4_train = [
        ARCPair([[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        ARCPair([[0, 2], [0, 0]], [[0, 0], [0, 2]]),
    ]
    t4_test = ARCPair([[3, 0], [0, 0]], [[0, 0], [3, 0]])
    tasks.append(
        ARCTask(
            id="arc_004_gravity",
            name="Downwards Gravity Translation",
            train_pairs=t4_train,
            test_pair=t4_test,
            target_rule="apply_gravity_down()",
            candidate_hypotheses=[
                "apply_rot180()",
                "apply_gravity_down()",
                "apply_flip_vertical()",
                "conclude_solution(rule='apply_gravity_down()')",
            ],
        )
    )

    return tasks


class ARCEvaluator:
    """Evaluates agent performance on ARC-AGI tasks."""

    def evaluate_task(self, agent_solution: str, task: ARCTask) -> bool:
        """Determines if the synthesized solution matches the ground-truth target rule."""
        if not agent_solution:
            return False
        clean_sol = agent_solution.strip().lower()
        clean_target = task.target_rule.strip().lower()
        return clean_target in clean_sol or task.name.lower() in clean_sol

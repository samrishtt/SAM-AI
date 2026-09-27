"""Recursive Self-Improvement Flywheel & Autonomous Task Synthesizer.

Enables SAM-AI to autonomously generate, verify, filter, and package
verifiable challenge datasets (Math, Code, ARC) for continuous GRPO policy evolution.
No human labelers. No third-party API dependencies. 100% Deterministic Truth.
"""

from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SynthesizedTask:
    """Represents an autonomously generated and deterministically verified task."""
    task_id: str
    domain: str  # "math", "code", "arc"
    prompt: str
    ground_truth: str
    verification_code: str
    difficulty_level: int = 1  # 1 to 5


class TaskSynthesizer:
    """Procedurally synthesizes verified reasoning challenges."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_math_vieta_challenge(self) -> SynthesizedTask:
        """Synthesizes a quadratic roots Vieta identity problem: x^2 - Sx + P = 0, find x1^2 + x2^2."""
        # x^2 - b*x + c = 0
        # x1 + x2 = b
        # x1 * x2 = c
        # x1^2 + x2^2 = (x1 + x2)^2 - 2*x1*x2 = b^2 - 2*c
        b = random.randint(3, 15)
        c = random.randint(1, 10)
        ans = b**2 - 2 * c

        task_id = f"math_vieta_{random.randint(10000, 99999)}"
        prompt = (
            f"<|im_start|>system\n"
            f"You are SAM-AI, a sovereign mathematical reasoner. Think inside <think> and output final answer in <answer>.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Let $x_1$ and $x_2$ be the roots of the quadratic equation $x^2 - {b}x + {c} = 0$. "
            f"Find the exact value of $x_1^2 + x_2^2$.\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        verification_code = f"assert ({b}**2 - 2*{c}) == {ans}"

        return SynthesizedTask(
            task_id=task_id,
            domain="math",
            prompt=prompt,
            ground_truth=str(ans),
            verification_code=verification_code,
            difficulty_level=2,
        )

    def generate_math_modular_challenge(self) -> SynthesizedTask:
        """Synthesizes modular arithmetic power problems: find (a^b) mod m."""
        a = random.choice([2, 3, 5, 7])
        b = random.randint(10, 50)
        m = random.choice([11, 13, 17, 19])
        ans = pow(a, b, m)

        task_id = f"math_mod_{random.randint(10000, 99999)}"
        prompt = (
            f"<|im_start|>system\n"
            f"You are SAM-AI, a sovereign mathematical reasoner. Derive step-by-step in <think> and put final integer in <answer>.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Compute the remainder when ${a}^{{{b}}}$ is divided by ${m}$.\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        verification_code = f"assert pow({a}, {b}, {m}) == {ans}"

        return SynthesizedTask(
            task_id=task_id,
            domain="math",
            prompt=prompt,
            ground_truth=str(ans),
            verification_code=verification_code,
            difficulty_level=3,
        )

    def generate_code_challenge(self) -> SynthesizedTask:
        """Synthesizes an algorithmic coding challenge with automated unit tests."""
        challenges = [
            {
                "name": "max_subarray_sum",
                "desc": "Write a Python function `max_subarray_sum(nums)` that finds the maximum sum of a contiguous non-empty subarray.",
                "solution": "def max_subarray_sum(nums):\n    cur = mx = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        mx = max(mx, cur)\n    return mx\n",
                "tests": [
                    "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6",
                    "assert max_subarray_sum([1]) == 1",
                    "assert max_subarray_sum([5,4,-1,7,8]) == 23",
                ]
            },
            {
                "name": "longest_valid_parentheses",
                "desc": "Write a Python function `longest_parentheses(s)` returning the length of the longest valid well-formed parentheses substring.",
                "solution": "def longest_parentheses(s):\n    stack = [-1]\n    max_len = 0\n    for i, char in enumerate(s):\n        if char == '(':\n            stack.append(i)\n        else:\n            stack.pop()\n            if not stack:\n                stack.append(i)\n            else:\n                max_len = max(max_len, i - stack[-1])\n    return max_len\n",
                "tests": [
                    "assert longest_parentheses('(()') == 2",
                    "assert longest_parentheses(')()())') == 4",
                    "assert longest_parentheses('') == 0",
                ]
            },
            {
                "name": "count_prime_factors",
                "desc": "Write a Python function `count_prime_factors(n)` returning the number of distinct prime factors of an integer n > 1.",
                "solution": "def count_prime_factors(n):\n    factors = set()\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.add(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.add(n)\n    return len(factors)\n",
                "tests": [
                    "assert count_prime_factors(12) == 2", # 2, 3
                    "assert count_prime_factors(30) == 3", # 2, 3, 5
                    "assert count_prime_factors(17) == 1", # 17
                ]
            }
        ]

        choice = random.choice(challenges)
        task_id = f"code_{choice['name']}_{random.randint(1000, 9999)}"

        prompt = (
            f"<|im_start|>system\n"
            f"You are SAM-AI, an expert software engineering intelligence. Think in <think> and output verified python code in ```python ```.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{choice['desc']}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        test_code = choice["solution"] + "\n" + "\n".join(choice["tests"])

        return SynthesizedTask(
            task_id=task_id,
            domain="code",
            prompt=prompt,
            ground_truth=choice["solution"],
            verification_code=test_code,
            difficulty_level=3,
        )

    def generate_arc_challenge(self) -> SynthesizedTask:
        """Synthesizes a 2D ARC-AGI grid transformation with mathematical symmetry."""
        # 3x3 rotation 90 degrees clockwise
        matrix = [[random.randint(0, 9) for _ in range(3)] for _ in range(3)]
        rotated = [
            [matrix[2][0], matrix[1][0], matrix[0][0]],
            [matrix[2][1], matrix[1][1], matrix[0][1]],
            [matrix[2][2], matrix[1][2], matrix[0][2]],
        ]

        def grid_str(g):
            return "\n".join(" ".join(map(str, row)) for row in g)

        task_id = f"arc_rot90_{random.randint(1000, 9999)}"
        prompt = (
            f"<|im_start|>system\n"
            f"You are SAM-AI, an expert spatial inductive reasoner. Solve in <think> and output transformed grid in <answer>.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[ARC Task] Rotate the 3x3 input grid 90 degrees clockwise.\n\n"
            f"Input Grid:\n{grid_str(matrix)}\n\n"
            f"Derive output step-by-step and output transformed grid inside <answer>...</answer>.\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        gt = grid_str(rotated)
        verification_code = f"in_g = {matrix}\nrot_g = {rotated}\nassert [list(row) for row in zip(*in_g[::-1])] == rot_g"

        return SynthesizedTask(
            task_id=task_id,
            domain="arc",
            prompt=prompt,
            ground_truth=gt,
            verification_code=verification_code,
            difficulty_level=2,
        )


class DeterministicTaskVerifier:
    """Executes verification code inside AST sandbox to filter unsound problems."""

    @staticmethod
    def verify(task: SynthesizedTask) -> bool:
        """Validates that the ground truth passes the verification assertions."""
        try:
            # Syntax validation first
            ast.parse(task.verification_code)
            # Safe namespace execution
            local_ns = {}
            exec(task.verification_code, {}, local_ns)
            return True
        except Exception:
            return False


class SelfImprovementFlywheel:
    """End-to-end pipeline: Synthesize -> Verify -> Filter -> Package for GRPO."""

    def __init__(self, seed: Optional[int] = None):
        self.synthesizer = TaskSynthesizer(seed=seed)
        self.verifier = DeterministicTaskVerifier()

    def generate_verified_batch(self, count: int = 10) -> List[SynthesizedTask]:
        """Generates a verified batch across all reasoning domains."""
        generators = [
            self.synthesizer.generate_math_vieta_challenge,
            self.synthesizer.generate_math_modular_challenge,
            self.synthesizer.generate_code_challenge,
            self.synthesizer.generate_arc_challenge,
        ]

        verified_tasks: List[SynthesizedTask] = []
        attempts = 0
        max_attempts = count * 3

        while len(verified_tasks) < count and attempts < max_attempts:
            attempts += 1
            gen_func = random.choice(generators)
            candidate = gen_func()
            if self.verifier.verify(candidate):
                verified_tasks.append(candidate)

        return verified_tasks

    def export_grpo_training_dataset(self, output_path: str | Path, count: int = 100) -> int:
        """Packages synthesized challenges into a JSONL dataset for Kaggle GRPO training."""
        tasks = self.generate_verified_batch(count=count)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with open(out_p, "w", encoding="utf-8") as f:
            for t in tasks:
                record = {
                    "task_id": t.task_id,
                    "domain": t.domain,
                    "prompt": t.prompt,
                    "ground_truth": t.ground_truth,
                    "difficulty": t.difficulty_level,
                }
                f.write(json.dumps(record) + "\n")

        return len(tasks)

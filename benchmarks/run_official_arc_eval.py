"""Official ARC-AGI Benchmark Evaluation Harness (ARC Prize Specification).

Evaluates SAM-AI against real, official ARC-AGI evaluation tasks directly from
the official benchmark repository (fchollet/ARC-AGI).

Rules:
1. 2 Attempts allowed per test pair (attempt_1, attempt_2).
2. Exact 2D grid match required (all dimensions and cell colors 0-9 must match).
3. If a task has multiple test pairs, ALL test pairs must be solved correctly.
4. Zero partial credit.
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data" / "official_arc_eval"


def grids_equal(g1: Any, g2: Any) -> bool:
    """Strict cell-by-cell exact match check according to ARC Prize rules."""
    try:
        a1 = np.array(g1)
        a2 = np.array(g2)
        return a1.shape == a2.shape and bool(np.array_equal(a1, a2))
    except Exception:
        return False


class OfficialARCEvaluator:
    """Evaluates inductive reasoning architectures against official ARC tasks."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.task_files = sorted(list(self.data_dir.glob("*.json")))

    def load_task(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_inductive_search(
        self,
        task: Dict[str, Any],
        max_depth: int = 2,
    ) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """Hypothesis search over spatial primitives verified on train pairs.
        
        Returns a list of (attempt_1, attempt_2) for each test pair.
        """
        train_pairs = task["train"]
        test_pairs = task["test"]

        # Primitives DSL
        def apply_op(g: np.ndarray, op: str, **kwargs) -> np.ndarray:
            if op == "identity":
                return g.copy()
            elif op == "rot90":
                return np.rot90(g, k=-1)
            elif op == "rot180":
                return np.rot90(g, k=2)
            elif op == "rot270":
                return np.rot90(g, k=1)
            elif op == "flip_h":
                return np.fliplr(g)
            elif op == "flip_v":
                return np.flipud(g)
            elif op == "crop_nonzero":
                nz = np.nonzero(g)
                if len(nz[0]) > 0:
                    r_min, r_max = np.min(nz[0]), np.max(nz[0])
                    c_min, c_max = np.min(nz[1]), np.max(nz[1])
                    return g[r_min:r_max+1, c_min:c_max+1]
                return g
            elif op == "color_swap":
                from_c = kwargs.get("from_c", 0)
                to_c = kwargs.get("to_c", 1)
                res = g.copy()
                res[g == from_c] = to_c
                return res
            return g

        candidate_ops = [
            ("identity", {}),
            ("rot90", {}),
            ("rot180", {}),
            ("rot270", {}),
            ("flip_h", {}),
            ("flip_v", {}),
            ("crop_nonzero", {}),
        ]
        
        # Color substitution candidates based on colors appearing in train inputs
        unique_colors = set()
        for p in train_pairs:
            unique_colors.update(np.unique(np.array(p["input"])).tolist())
        for c1 in list(unique_colors)[:4]:
            for c2 in list(unique_colors)[:4]:
                if c1 != c2:
                    candidate_ops.append(("color_swap", {"from_c": c1, "to_c": c2}))

        # Score hypotheses on train demonstration pairs
        scored_hypotheses = []
        for op, kw in candidate_ops:
            matches = 0
            for p in train_pairs:
                in_g = np.array(p["input"])
                out_g = np.array(p["output"])
                pred = apply_op(in_g, op, **kw)
                if pred.shape == out_g.shape and np.array_equal(pred, out_g):
                    matches += 1
            score = matches / len(train_pairs)
            scored_hypotheses.append((score, op, kw))

        scored_hypotheses.sort(key=lambda x: x[0], reverse=True)
        top1 = scored_hypotheses[0]
        top2 = scored_hypotheses[1] if len(scored_hypotheses) > 1 else top1

        # Predict for each test pair
        predictions = []
        for p in test_pairs:
            test_in = np.array(p["input"])
            att1 = apply_op(test_in, top1[1], **top1[2]).tolist()
            att2 = apply_op(test_in, top2[1], **top2[2]).tolist()
            predictions.append((att1, att2))

        return predictions

    def evaluate_all(self, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Runs the official evaluation benchmark and computes Pass@1 and Pass@2."""
        tasks_to_eval = self.task_files[:max_tasks] if max_tasks else self.task_files
        print("\n" + "=" * 70)
        print(f"  RUNNING OFFICIAL ARC-AGI BENCHMARK ({len(tasks_to_eval)} TASKS)")
        print(f"  Source: fchollet/ARC-AGI/data/evaluation/")
        print("=" * 70)

        total_tasks = len(tasks_to_eval)
        solved_pass1 = 0
        solved_pass2 = 0
        task_results = []

        start_time = time.perf_counter()

        for idx, task_file in enumerate(tasks_to_eval, 1):
            task_id = task_file.stem
            task_data = self.load_task(task_file)
            test_pairs = task_data["test"]
            predictions = self.run_inductive_search(task_data)

            # A task is solved if ALL test pairs in it are solved
            task_pass1 = True
            task_pass2 = True

            for p_idx, pair in enumerate(test_pairs):
                gt_output = pair["output"]
                att1, att2 = predictions[p_idx]

                p_pass1 = grids_equal(att1, gt_output)
                p_pass2 = p_pass1 or grids_equal(att2, gt_output)

                if not p_pass1:
                    task_pass1 = False
                if not p_pass2:
                    task_pass2 = False

            if task_pass1:
                solved_pass1 += 1
            if task_pass2:
                solved_pass2 += 1

            status = "SOLVED (Pass@1)" if task_pass1 else ("SOLVED (Pass@2)" if task_pass2 else "UNRESOLVED")
            task_results.append({
                "task_id": task_id,
                "status": status,
                "pass1": task_pass1,
                "pass2": task_pass2,
                "num_train": len(task_data["train"]),
                "num_test": len(task_data["test"]),
            })
            print(f"  [{idx:02d}/{total_tasks:02d}] Task {task_id} -> {status}")

        total_time = time.perf_counter() - start_time
        pass1_rate = (solved_pass1 / total_tasks) * 100.0
        pass2_rate = (solved_pass2 / total_tasks) * 100.0

        print("\n" + "=" * 70)
        print("  OFFICIAL ARC-AGI BENCHMARK RESULTS")
        print("=" * 70)
        print(f"  Total Official Tasks Evaluated:  {total_tasks}")
        print(f"  Official Solved (Pass@1):        {solved_pass1} / {total_tasks} ({pass1_rate:.1f}%)")
        print(f"  Official Solved (Pass@2):        {solved_pass2} / {total_tasks} ({pass2_rate:.1f}%)")
        print(f"  Total Evaluation Time:           {total_time:.2f} s ({total_time / total_tasks * 1000.0:.1f} ms/task)")
        print("=" * 70 + "\n")

        return {
            "total_tasks": total_tasks,
            "solved_pass1": solved_pass1,
            "solved_pass2": solved_pass2,
            "pass1_rate": pass1_rate,
            "pass2_rate": pass2_rate,
            "task_results": task_results,
            "total_time_sec": total_time,
        }


if __name__ == "__main__":
    evaluator = OfficialARCEvaluator()
    evaluator.evaluate_all()

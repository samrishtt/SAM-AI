"""Unit and integration tests for SAM-AI Official ARC Benchmark Evaluator."""

import pytest
from pathlib import Path
from sam_ai.benchmarks.arc_solver import (
    ARCTask,
    ARCOfficialEvaluator,
    parse_grid_from_text,
    grids_equal,
    format_grid,
)


def test_parse_grid_from_json_and_markdown():
    text_json = 'Here is the grid:\n[[1, 2], [3, 4]]\nDone.'
    grid1 = parse_grid_from_text(text_json)
    assert grid1 == [[1, 2], [3, 4]]

    text_md = '```json\n[[0, 8, 0], [8, 0, 8]]\n```'
    grid2 = parse_grid_from_text(text_md)
    assert grid2 == [[0, 8, 0], [8, 0, 8]]


def test_grid_equality():
    g1 = [[1, 2], [3, 4]]
    g2 = [[1, 2], [3, 4]]
    g3 = [[1, 2], [3, 5]]
    g4 = [[1, 2, 0], [3, 4, 0]]

    assert grids_equal(g1, g2)
    assert not grids_equal(g1, g3)
    assert not grids_equal(g1, g4)
    assert not grids_equal(g1, None)


def test_official_two_attempt_scoring():
    task = ARCTask(
        task_id="demo_task",
        train_pairs=[{"input": [[1]], "output": [[2]]}],
        test_inputs=[[[3]]],
        test_outputs=[[[4]]],
    )
    evaluator = ARCOfficialEvaluator()

    # Case 1: First attempt wrong, second attempt correct -> Marked Solved
    preds_win_on_2nd = {
        "demo_task": [{"attempt_1": [[9]], "attempt_2": [[4]]}]
    }
    res1 = evaluator.evaluate_predictions([task], preds_win_on_2nd)
    assert res1["solved_instances"] == 1
    assert res1["accuracy_percent"] == 100.0

    # Case 2: Both attempts wrong -> Marked Failed
    preds_fail = {
        "demo_task": [{"attempt_1": [[9]], "attempt_2": [[8]]}]
    }
    res2 = evaluator.evaluate_predictions([task], preds_fail)
    assert res2["solved_instances"] == 0
    assert res2["accuracy_percent"] == 0.0


def test_load_real_arc_evaluation_data():
    evaluator = ARCOfficialEvaluator(data_dir="benchmarks/data/official_arc_eval")
    tasks = evaluator.load_all_tasks()

    assert len(tasks) == 25, f"Expected 25 official tasks, found {len(tasks)}"
    sample_task = tasks[0]
    assert len(sample_task.train_pairs) >= 1
    assert len(sample_task.test_inputs) >= 1

    prompt = evaluator.construct_prompt(sample_task)
    assert "Example 1 Input" in prompt
    assert "Test Input" in prompt

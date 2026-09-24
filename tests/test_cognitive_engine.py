"""Integration tests for the unified CognitiveEngine."""

import pytest
from core.engine import CognitiveEngine, CognitiveCycleResult


def test_cognitive_engine_system2_solve():
    engine = CognitiveEngine(system2_enabled=True, mcts_simulations=15, auto_consolidate=True)

    goal = "Deduce positive regulation of kinase"
    context = {"target": "Kinase_X", "activator": "Factor_Y"}
    allowed_actions = [
        "inspect_receptors()",
        "simulate_pathway()",
        "conclude_solution(result='positive_regulation')",
    ]

    res: CognitiveCycleResult = engine.solve_task(
        goal=goal,
        initial_context=context,
        allowed_actions=allowed_actions,
        max_steps=4,
    )

    assert res.success
    assert res.reward > 0.0
    assert len(res.traces) > 0
    assert len(engine.episodic_store) == 1
    # Check that System 2 mode was logged
    assert any("System 2" in t.system_mode for t in res.traces)


def test_cognitive_engine_system1_ablation():
    engine = CognitiveEngine(system2_enabled=False)

    goal = "Quick intuition heuristic"
    allowed_actions = [
        "conclude_heuristic_prior()",
        "explore_alternatives()",
    ]

    res = engine.solve_task(
        goal=goal,
        allowed_actions=allowed_actions,
        max_steps=2,
    )

    assert len(res.traces) > 0
    assert all("System 1" in t.system_mode for t in res.traces)

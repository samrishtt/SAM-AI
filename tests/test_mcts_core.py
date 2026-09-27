"""Unit tests for SAM-AI PUCT MCTS Core and PRM Pruning."""

import pytest
from sam_ai.reasoning.mcts_core import MCTSNode, PUCTSearchEngine


def test_puct_formula_prior_vs_q_value():
    """
    Verifies that PUCT uses policy_prior strictly in exploration U,
    and prm_score strictly in Q-value.
    """
    engine = PUCTSearchEngine(c_puct=2.0, lambda_prm=0.5, prune_threshold=0.2)
    parent = MCTSNode(state_text="Root")

    # Child A: Low prior, high PRM
    child_a = MCTSNode(
        state_text="Root -> A",
        parent=parent,
        policy_prior=0.1,
        prm_score=0.9,
        visits=0,
    )
    # Child B: High prior, moderate PRM
    child_b = MCTSNode(
        state_text="Root -> B",
        parent=parent,
        policy_prior=0.9,
        prm_score=0.6,
        visits=0,
    )
    parent.children = [child_a, child_b]

    # At visits == 0, Q(s, a) = prm_score
    # U(s, a) = c_puct * prior * (sqrt(total_parent_visits) / (1 + child.visits))
    # total_parent_visits = 0 -> max(1, 0) = 1
    # For A: Q = 0.9, U = 2.0 * 0.1 * (1 / 1) = 0.2 -> Score = 1.1
    # For B: Q = 0.6, U = 2.0 * 0.9 * (1 / 1) = 1.8 -> Score = 2.4
    selected = engine.select_child(parent)
    assert selected == child_b, "Child B should be selected due to higher policy prior in PUCT bonus"


def test_step_prm_early_pruning():
    """Verifies that branches below prune_threshold are completely excluded."""
    engine = PUCTSearchEngine(prune_threshold=0.40)
    root = MCTSNode(state_text="Problem: Solve x + 5 = 10")

    candidates = [
        ("Step 1: Subtract 5 from both sides", 0.7, 0.95),  # Keep
        ("Step 1: Multiply both sides by 0", 0.1, 0.05),     # Prune (< 0.40)
        ("Step 1: Guess x = 100", 0.2, 0.20),                # Prune (< 0.40)
    ]

    expanded = engine.expand(root, candidates, is_terminal_fn=lambda t: False)
    assert len(expanded) == 1
    assert len(root.children) == 1
    assert "Subtract 5" in root.children[0].action_text


def test_mcts_backpropagation():
    """Verifies visit counts and value accumulation through ancestral chain."""
    engine = PUCTSearchEngine()
    root = MCTSNode(state_text="Root")
    child = MCTSNode(state_text="Child", parent=root)
    grandchild = MCTSNode(state_text="Grandchild", parent=child)

    root.children = [child]
    child.children = [grandchild]

    engine.backpropagate(grandchild, outcome_reward=1.0)
    assert grandchild.visits == 1 and grandchild.value_sum == 1.0
    assert child.visits == 1 and child.value_sum == 1.0
    assert root.visits == 1 and root.value_sum == 1.0


def test_mcts_search_convergence():
    """Verifies that MCTS converges towards the high-reward reasoning path."""
    engine = PUCTSearchEngine(c_puct=1.0, prune_threshold=0.2)

    def generator(state: str):
        if "Correct" in state:
            return []
        return [
            ("Choose Correct Path", 0.5),
            ("Choose Bad Path", 0.5),
        ]

    def prm(texts):
        return [0.9 if "Correct" in t else 0.1 for t in texts]

    def terminal(text: str) -> bool:
        return "Path" in text

    def evaluator(text: str) -> float:
        return 1.0 if "Correct" in text else 0.0

    best_path = engine.search(
        root_prompt="Start",
        generator_fn=generator,
        prm_fn=prm,
        is_terminal_fn=terminal,
        evaluator_fn=evaluator,
        iterations=15,
    )
    assert "Choose Correct Path" in best_path

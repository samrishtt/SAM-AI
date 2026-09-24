"""Unit tests for System 1 prior and System 2 PUCT MCTS."""

import pytest
import numpy as np
from core.world_model.latent_simulator import WorldModel, WorldState
from core.reasoning.policy_prior import PolicyPrior
from core.reasoning.verifier import ProcessRewardModel
from core.reasoning.mcts import MCTSSearch, MCTSNode
from core.memory.working_memory import WorkingMemory


def test_policy_prior_softmax():
    pp = PolicyPrior(temperature=1.0)
    wm = WorkingMemory()
    candidates = pp.propose_candidates(
        goal="Synthesize sorting algorithm",
        working_memory=wm,
        allowed_actions=["quick_sort()", "merge_sort()", "bubble_sort()"],
    )

    assert len(candidates) == 3
    total_prob = sum(c.prior_prob for c in candidates)
    assert np.isclose(total_prob, 1.0, atol=1e-4)
    assert all(c.prior_prob > 0.0 for c in candidates)


def test_process_reward_model_checks():
    prm = ProcessRewardModel()
    state = WorldState(state_id="s0")

    # Empty action check
    res_empty = prm.verify_step(state, "", "Test goal")
    assert not res_empty.is_valid
    assert res_empty.fatal_error

    # Valid action check
    res_valid = prm.verify_step(state, "conclude_solution(rule='rot90')", "Solve ARC task")
    assert res_valid.is_valid
    assert res_valid.step_value >= 0.5


def test_mcts_search_convergence():
    wm = WorldModel()
    pp = PolicyPrior()
    prm = ProcessRewardModel()
    mcts = MCTSSearch(world_model=wm, policy_prior=pp, verifier=prm, c_puct=1.414, max_depth=5)

    working_mem = WorkingMemory()
    initial_state = WorldState(state_id="root")

    actions = [
        "explore_subproblem_A()",
        "explore_subproblem_B()",
        "conclude_solution()",
    ]

    result = mcts.search(
        initial_state=initial_state,
        goal="Solve formal problem",
        working_memory=working_mem,
        num_simulations=20,
        allowed_actions=actions,
    )

    assert result.simulations_run == 20
    assert result.nodes_expanded > 1
    assert result.best_action in actions
    # The high-value goal action should be preferred
    assert result.best_action == "conclude_solution()"
    assert result.best_value > 0.0

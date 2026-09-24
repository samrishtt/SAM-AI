"""Adaptive Test-Time Compute Engine (HyperMCTS).

Exceeds static reasoning_effort presets by computing entropy-guided test-time
budgeting: dynamically scales Monte Carlo Tree Search simulations proportional
to intermediate uncertainty and branch entropy H(P(a | s)).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.world_model.latent_simulator import WorldModel, WorldState, PredictedTransition
from core.reasoning.policy_prior import PolicyPrior, ActionCandidate
from core.reasoning.verifier import ProcessRewardModel, VerificationResult
from core.reasoning.mcts import MCTSNode, MCTSSearchResult
from core.memory.working_memory import WorkingMemory


@dataclass
class AdaptiveComputeBudget:
    entropy: float
    recommended_simulations: int
    reasoning_tier: str  # "low", "medium", "high", "xhigh", "hyper"
    uncertainty_score: float


class AdaptiveReasoningEngine:
    """Entropy-guided test-time compute scaling engine."""

    def __init__(
        self,
        world_model: WorldModel,
        policy_prior: PolicyPrior,
        verifier: ProcessRewardModel,
        base_simulations: int = 15,
        max_simulations: int = 120,
        c_puct: float = 1.414,
    ):
        self.world_model = world_model
        self.policy_prior = policy_prior
        self.verifier = verifier
        self.base_simulations = base_simulations
        self.max_simulations = max_simulations
        self.c_puct = c_puct

    def compute_entropy_budget(self, candidates: List[ActionCandidate]) -> AdaptiveComputeBudget:
        """Calculates Shannon entropy over candidate prior distribution to dynamically size search budget."""
        if not candidates:
            return AdaptiveComputeBudget(0.0, self.base_simulations, "low", 0.0)

        probs = np.array([c.prior_prob for c in candidates], dtype=np.float32)
        probs = np.clip(probs, 1e-9, 1.0)
        probs = probs / np.sum(probs)

        # Shannon entropy: H = -sum(p * log2(p))
        entropy = -float(np.sum(probs * np.log2(probs)))
        max_entropy = math.log2(max(2, len(candidates)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Scale simulations dynamically
        scaled_sims = int(self.base_simulations + (self.max_simulations - self.base_simulations) * normalized_entropy)

        if scaled_sims < 25:
            tier = "low"
        elif scaled_sims < 50:
            tier = "medium"
        elif scaled_sims < 80:
            tier = "high"
        elif scaled_sims < 100:
            tier = "xhigh"
        else:
            tier = "hyper"

        return AdaptiveComputeBudget(
            entropy=entropy,
            recommended_simulations=scaled_sims,
            reasoning_tier=tier,
            uncertainty_score=normalized_entropy,
        )

    def search_adaptive(
        self,
        initial_state: WorldState,
        goal: str,
        working_memory: WorkingMemory,
        allowed_actions: Optional[List[str]] = None,
        force_tier: Optional[str] = None,
        domain: str = "general",
    ) -> Tuple[MCTSSearchResult, AdaptiveComputeBudget]:
        """Runs MCTS with dynamically allocated compute based on Shannon entropy."""
        candidates = self.policy_prior.propose_candidates(
            goal=goal,
            working_memory=working_memory,
            allowed_actions=allowed_actions,
            max_candidates=5,
        )

        budget = self.compute_entropy_budget(candidates)

        if force_tier:
            tier_map = {"low": 15, "medium": 35, "high": 70, "xhigh": 95, "hyper": 120}
            sims = tier_map.get(force_tier.lower(), budget.recommended_simulations)
        else:
            sims = budget.recommended_simulations

        # Execute PUCT Search with dynamic simulation count
        root = MCTSNode(state=initial_state)
        nodes_expanded = 1

        # Root expansion
        for cand in candidates:
            trans: PredictedTransition = self.world_model.simulate_step(root.state, cand.action)
            verif: VerificationResult = self.verifier.verify_step(root.state, cand.action, goal, domain=domain)

            is_terminal = verif.step_value >= 0.75
            child = MCTSNode(
                state=trans.next_state,
                parent=root,
                action_from_parent=cand.action,
                prior_prob=cand.prior_prob,
                total_value=verif.step_value + trans.predicted_reward,
                visit_count=1,
                is_terminal=is_terminal,
                terminal_reward=1.0 if is_terminal else 0.0,
            )
            root.children[cand.action] = child
            nodes_expanded += 1

        # Tree exploration loop
        for _ in range(sims):
            node = root
            path = [node]
            depth = 0

            # Selection
            while node.children and not node.is_terminal and depth < 8:
                best_action = None
                best_child = None
                best_score = -float("inf")
                total_visits = sum(c.visit_count for c in node.children.values())
                sqrt_total = math.sqrt(max(1, total_visits))

                for act, ch in node.children.items():
                    u = self.c_puct * ch.prior_prob * (sqrt_total / (1 + ch.visit_count))
                    q = ch.q_value
                    if q + u > best_score:
                        best_score = q + u
                        best_action = act
                        best_child = ch

                node = best_child
                path.append(node)
                depth += 1

            # Value evaluation & backpropagation
            val = node.terminal_reward if node.is_terminal else node.q_value
            for b_node in path:
                b_node.visit_count += 1
                b_node.total_value += val

        # Best trajectory extraction
        best_act = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        best_traj = [best_act]
        best_val = root.children[best_act].q_value

        res = MCTSSearchResult(
            best_action=best_act,
            best_trajectory=best_traj,
            best_value=best_val,
            simulations_run=sims,
            nodes_expanded=nodes_expanded,
            tree_depth_reached=depth,
            root_action_distribution={
                act: ch.visit_count / max(1, sum(c.visit_count for c in root.children.values()))
                for act, ch in root.children.items()
            },
        )

        return res, budget

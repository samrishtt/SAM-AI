"""System 2: PUCT Monte Carlo Tree Search (MCTS) Reasoning Engine.

Implements deliberate inference-time search over candidate reasoning paths
using Polynomial Upper Confidence Trees (PUCT), guided by System 1 prior
probabilities P(a | s), Process Reward Model step values, and internal
world-model rollouts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.world_model.latent_simulator import WorldModel, WorldState, PredictedTransition
from core.reasoning.policy_prior import PolicyPrior, ActionCandidate
from core.reasoning.verifier import ProcessRewardModel, VerificationResult
from core.memory.working_memory import WorkingMemory


@dataclass
class MCTSNode:
    state: WorldState
    parent: Optional[MCTSNode] = None
    action_from_parent: Optional[str] = None
    prior_prob: float = 1.0
    visit_count: int = 0
    total_value: float = 0.0
    children: Dict[str, MCTSNode] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_reward: float = 0.0

    @property
    def q_value(self) -> float:
        """Returns empirical mean action-value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class MCTSSearchResult:
    best_action: str
    best_trajectory: List[str]
    best_value: float
    simulations_run: int
    nodes_expanded: int
    tree_depth_reached: int
    root_action_distribution: Dict[str, float]


class MCTSSearch:
    """Deliberate System 2 inference-time tree search."""

    def __init__(
        self,
        world_model: WorldModel,
        policy_prior: PolicyPrior,
        verifier: ProcessRewardModel,
        c_puct: float = 1.414,
        max_depth: int = 8,
    ):
        self.world_model = world_model
        self.policy_prior = policy_prior
        self.verifier = verifier
        self.c_puct = c_puct
        self.max_depth = max_depth
        self._nodes_expanded = 0

    def search(
        self,
        initial_state: WorldState,
        goal: str,
        working_memory: WorkingMemory,
        num_simulations: int = 30,
        allowed_actions: Optional[List[str]] = None,
        domain: str = "general",
    ) -> MCTSSearchResult:
        """Executes PUCT MCTS simulations from the root state."""
        self._nodes_expanded = 1
        root = MCTSNode(state=initial_state)

        # Expand root immediately
        self._expand_node(root, goal, working_memory, allowed_actions, domain)

        max_depth_observed = 0

        for sim_idx in range(num_simulations):
            node = root
            search_path: List[MCTSNode] = [node]
            depth = 0

            # 1. Selection phase: traverse tree using PUCT
            while node.children and not node.is_terminal and depth < self.max_depth:
                action, next_node = self._select_child_puct(node)
                node = next_node
                search_path.append(node)
                depth += 1

            max_depth_observed = max(max_depth_observed, depth)

            # 2. Expansion phase (if not terminal and not at max depth)
            value = 0.0
            if not node.is_terminal and depth < self.max_depth:
                self._expand_node(node, goal, working_memory, allowed_actions, domain)
                # If expansion created children, choose one to evaluate
                if node.children:
                    _, first_child = next(iter(node.children.items()))
                    value = self._evaluate_rollout(first_child, goal, domain)
                else:
                    value = node.q_value
            elif node.is_terminal:
                value = node.terminal_reward
            else:
                value = node.q_value

            # 3. Backpropagation phase
            for back_node in search_path:
                back_node.visit_count += 1
                back_node.total_value += value

        # Compute best action & trajectory from root
        best_action, best_traj, best_val = self._extract_best_trajectory(root)

        # Action distribution at root
        total_visits = sum(c.visit_count for c in root.children.values()) or 1
        root_dist = {
            act: child.visit_count / total_visits
            for act, child in root.children.items()
        }

        return MCTSSearchResult(
            best_action=best_action,
            best_trajectory=best_traj,
            best_value=best_val,
            simulations_run=num_simulations,
            nodes_expanded=self._nodes_expanded,
            tree_depth_reached=max_depth_observed,
            root_action_distribution=root_dist,
        )

    def _select_child_puct(self, node: MCTSNode) -> Tuple[str, MCTSNode]:
        """Selects child maximizing PUCT objective: Q(s, a) + U(s, a)."""
        total_parent_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(max(1, total_parent_visits))

        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # Exploration bonus U(s, a)
            u_score = self.c_puct * child.prior_prob * (sqrt_total / (1 + child.visit_count))
            # Exploitation Q(s, a)
            q_score = child.q_value
            total_score = q_score + u_score

            if total_score > best_score:
                best_score = total_score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(
        self,
        node: MCTSNode,
        goal: str,
        working_memory: WorkingMemory,
        allowed_actions: Optional[List[str]],
        domain: str,
    ) -> None:
        """Generates candidates via System 1 prior, filters with PRM, and adds child nodes."""
        candidates: List[ActionCandidate] = self.policy_prior.propose_candidates(
            goal=goal,
            working_memory=working_memory,
            allowed_actions=allowed_actions,
            max_candidates=4,
        )

        for cand in candidates:
            # 1. Simulate transition in world model
            trans: PredictedTransition = self.world_model.simulate_step(node.state, cand.action)

            # 2. Verify step with PRM
            verif: VerificationResult = self.verifier.verify_step(
                state=node.state,
                action=cand.action,
                goal=goal,
                domain=domain,
            )

            # If fatal error or unsafe transition, prune or assign negative terminal
            if verif.fatal_error or not trans.is_safe:
                child = MCTSNode(
                    state=trans.next_state,
                    parent=node,
                    action_from_parent=cand.action,
                    prior_prob=cand.prior_prob,
                    is_terminal=True,
                    terminal_reward=-1.0,
                )
            else:
                is_goal_reached = verif.step_value >= 0.75
                child = MCTSNode(
                    state=trans.next_state,
                    parent=node,
                    action_from_parent=cand.action,
                    prior_prob=cand.prior_prob,
                    total_value=verif.step_value + trans.predicted_reward,
                    visit_count=1,
                    is_terminal=is_goal_reached,
                    terminal_reward=1.0 if is_goal_reached else 0.0,
                )

            node.children[cand.action] = child
            self._nodes_expanded += 1

    def _evaluate_rollout(self, node: MCTSNode, goal: str, domain: str) -> float:
        """Evaluates node via PRM heuristic value."""
        if node.is_terminal:
            return node.terminal_reward
        return node.q_value

    def _extract_best_trajectory(self, root: MCTSNode) -> Tuple[str, List[str], float]:
        """Traces the greedy path of most visited nodes from root."""
        if not root.children:
            return "no_action", [], 0.0

        # Choose root action with highest visit count
        best_root_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]

        curr = root
        trajectory: List[str] = []
        while curr.children:
            best_act, next_node = max(curr.children.items(), key=lambda item: item[1].visit_count)
            trajectory.append(best_act)
            if next_node.is_terminal:
                break
            curr = next_node

        best_val = root.children[best_root_action].q_value
        return best_root_action, trajectory, best_val

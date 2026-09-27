"""Monte Carlo Tree Search with PUCT and Process Reward Model (PRM) Guidance.

Mathematically correct implementation of PUCT:
a* = argmax_a [ Q(s, a) + U(s, a) ]
U(s, a) = c_puct * P(a|s) * (sqrt(sum_b N(s, b)) / (1 + N(s, a)))
Q(s, a) = (1 - lambda) * Q_rollout(s, a) + lambda * R_prm(s, a)

P(a|s) is strictly the policy prior (generation probability under pi_theta).
R_prm(s, a) informs the state-action value Q, NOT the exploration prior U.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class MCTSNode:
    state_text: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    action_text: str = ""
    policy_prior: float = 1.0        # P(a|s): generation probability under pi_theta
    prm_score: float = 0.5           # R_prm(s, a): Process Reward Model score
    visits: int = 0                  # N(s, a)
    value_sum: float = 0.0           # Cumulative rollout return
    is_terminal: bool = False

    @property
    def q_rollout(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def compute_q_value(self, lambda_prm: float = 0.5) -> float:
        """Combines empirical rollout returns with PRM step verification score."""
        if self.visits == 0:
            return self.prm_score
        return (1.0 - lambda_prm) * self.q_rollout + lambda_prm * self.prm_score


class PUCTSearchEngine:
    """
    Standard AlphaZero-style PUCT search engine for reasoning steps.
    """

    def __init__(
        self,
        c_puct: float = 1.414,
        lambda_prm: float = 0.5,
        prune_threshold: float = 0.30,
    ):
        self.c_puct = c_puct
        self.lambda_prm = lambda_prm
        self.prune_threshold = prune_threshold

    def compute_u_score(self, child: MCTSNode, total_parent_visits: int) -> float:
        """
        Calculates PUCT exploration bonus:
        U(s, a) = c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s, a))
        """
        exploration = math.sqrt(max(1, total_parent_visits)) / (1 + child.visits)
        return self.c_puct * child.policy_prior * exploration

    def select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Selects the child maximizing Q(s, a) + U(s, a)."""
        if not node.children:
            return None

        total_parent_visits = sum(c.visits for c in node.children)
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            q_val = child.compute_q_value(self.lambda_prm)
            u_val = self.compute_u_score(child, total_parent_visits)
            puct_score = q_val + u_val

            if puct_score > best_score:
                best_score = puct_score
                best_child = child

        return best_child

    def expand(
        self,
        node: MCTSNode,
        candidates: List[Tuple[str, float, float]],  # List of (step_text, policy_prior, prm_score)
        is_terminal_fn: Callable[[str], bool],
    ) -> List[MCTSNode]:
        """
        Expands node with valid candidate steps, pruning branches below threshold.
        """
        added_children: List[MCTSNode] = []

        for step_text, prior, prm in candidates:
            # Step-level PRM early pruning
            if prm < self.prune_threshold:
                continue

            full_text = f"{node.state_text}\n{step_text}".strip()
            child = MCTSNode(
                state_text=full_text,
                parent=node,
                action_text=step_text,
                policy_prior=prior,
                prm_score=prm,
                is_terminal=is_terminal_fn(full_text),
            )
            node.children.append(child)
            added_children.append(child)

        return added_children

    def backpropagate(self, node: MCTSNode, outcome_reward: float):
        """Propagates rollout outcome up to the root."""
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += outcome_reward
            curr = curr.parent

    def search(
        self,
        root_prompt: str,
        generator_fn: Callable[[str], List[Tuple[str, float]]],  # Returns [(step, log_prob_or_prob)]
        prm_fn: Callable[[List[str]], List[float]],             # Batch PRM scoring
        is_terminal_fn: Callable[[str], bool],
        evaluator_fn: Callable[[str], float],                   # Rollout terminal reward
        iterations: int = 20,
    ) -> str:
        """Executes MCTS search over reasoning steps."""
        root = MCTSNode(state_text=root_prompt)

        # Initial expansion of root
        raw_candidates = generator_fn(root.state_text)
        if not raw_candidates:
            return root_prompt

        candidate_texts = [f"{root.state_text}\n{c[0]}" for c in raw_candidates]
        prms = prm_fn(candidate_texts)
        full_candidates = [
            (c[0], c[1], p) for c, p in zip(raw_candidates, prms)
        ]
        self.expand(root, full_candidates, is_terminal_fn)

        for _ in range(iterations):
            # 1. Selection
            curr = root
            while curr.children and not curr.is_terminal:
                next_node = self.select_child(curr)
                if next_node is None:
                    break
                curr = next_node

            # 2. Expansion
            if not curr.is_terminal and curr.visits > 0:
                raw_cands = generator_fn(curr.state_text)
                if raw_cands:
                    c_texts = [f"{curr.state_text}\n{c[0]}" for c in raw_cands]
                    c_prms = prm_fn(c_texts)
                    f_cands = [(c[0], c[1], p) for c, p in zip(raw_cands, c_prms)]
                    new_children = self.expand(curr, f_cands, is_terminal_fn)
                    if new_children:
                        # Select best newly expanded child via PUCT rather than arbitrary index 0
                        selected = self.select_child(curr)
                        if selected is not None:
                            curr = selected
                        else:
                            curr = new_children[0]

            # 3. Rollout / Evaluation
            reward = evaluator_fn(curr.state_text) if curr.is_terminal else curr.prm_score

            # 4. Backpropagation
            self.backpropagate(curr, reward)

        # Select most visited non-root child
        if not root.children:
            return root.state_text

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.state_text

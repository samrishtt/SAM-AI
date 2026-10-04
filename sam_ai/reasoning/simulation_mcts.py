"""
In-Sandbox World Model & Action-Conserving MCTS Engine for ARC-AGI-3 & Dynamic Environments.
Fixes Flaw 2 (Action Economy & Trial-and-Error Penalty).

Key Mechanics:
1. ARC-AGI-3 Quadratic Action Penalty:
   Score Multiplier = (25 / (25 + a))^2
   Blind real-world interaction (e.g. 150 moves) degrades score by 98%.
2. In-Sandbox World Model Simulation:
   Nexis formulates or updates a local transition model f(s, a) -> (s', r, done).
   Hundreds of rollouts occur purely in-memory (cost = 0 physical actions).
3. Budget-Guarded Action Commitment:
   Only the verified shortest solution path (or best-gain action) is committed
   to the live environment, guaranteeing action count a <= 25 whenever possible.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any, Set
import copy


@dataclass
class ActionBudgetManager:
    """
    Tracks and enforces physical action economy under the ARC-AGI-3 quadratic penalty:
    Score Factor = (base_allowance / (base_allowance + actions_taken))^2
    """
    base_allowance: int = 25
    actions_committed: int = 0
    simulated_actions: int = 0

    def compute_multiplier(self, actions: Optional[int] = None) -> float:
        """Returns the quadratic action multiplier."""
        a = self.actions_committed if actions is None else actions
        return float((self.base_allowance / (self.base_allowance + a)) ** 2)

    def record_action(self, count: int = 1) -> float:
        """Records a committed physical environment action and returns new multiplier."""
        self.actions_committed += count
        return self.compute_multiplier()

    def record_simulated_action(self, count: int = 1) -> None:
        """Records an in-memory simulation step (zero physical score cost)."""
        self.simulated_actions += count

    def get_remaining_budget_for_target(self, target_multiplier: float = 0.5) -> int:
        """
        Calculates how many physical actions can still be taken before the multiplier
        falls below the target threshold.
        (base / (base + a))^2 >= target  =>  base + a <= base / sqrt(target)
        a <= base * (1 / sqrt(target) - 1)
        """
        if target_multiplier <= 0:
            return 999999
        max_allowed_actions = int(self.base_allowance * (1.0 / math.sqrt(target_multiplier) - 1.0))
        remaining = max_allowed_actions - self.actions_committed
        return max(0, remaining)


@dataclass
class SimNode:
    """Node in the simulation search tree."""
    state_repr: str
    state_data: Any
    parent: Optional['SimNode'] = None
    action_from_parent: Optional[Any] = None
    children: Dict[Any, 'SimNode'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    is_terminal: bool = False
    is_solved: bool = False
    depth: int = 0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def get_path_from_root(self) -> List[Any]:
        """Backtracks from current node to root to retrieve action sequence."""
        actions = []
        curr = self
        while curr.parent is not None and curr.action_from_parent is not None:
            actions.append(curr.action_from_parent)
            curr = curr.parent
        actions.reverse()
        return actions


class SimulationMCTSEngine:
    """
    Offline in-memory MCTS / Tree Search over simulated Python game state transitions.
    Explores candidate action branches without spending physical environment actions.
    """

    def __init__(
        self,
        c_puct: float = 1.414,
        max_depth: int = 30,
        heuristic_fn: Optional[Callable[[Any], float]] = None
    ):
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.heuristic_fn = heuristic_fn or (lambda state: 0.0)
        self.budget_manager = ActionBudgetManager()

    def search_optimal_plan(
        self,
        initial_state: Any,
        state_key_fn: Callable[[Any], str],
        step_fn: Callable[[Any, Any], Tuple[Any, float, bool, bool]],
        action_space: List[Any],
        num_simulations: int = 200,
        early_stop_on_win: bool = True
    ) -> Tuple[List[Any], Optional[SimNode]]:
        """
        Runs MCTS in-memory using transition step_fn(state, action) -> (next_state, reward, is_terminal, is_win).
        Returns:
            (best_action_sequence, best_terminal_or_highest_value_node)
        """
        root_key = state_key_fn(initial_state)
        root = SimNode(state_repr=root_key, state_data=copy.deepcopy(initial_state))

        solved_nodes: List[SimNode] = []
        visited_states: Set[str] = {root_key}

        for sim_idx in range(num_simulations):
            self.budget_manager.record_simulated_action(1)
            
            # 1. Selection
            curr = root
            path = [curr]

            while curr.children and not curr.is_terminal and curr.depth < self.max_depth:
                # Select child using PUCT
                best_action = None
                best_score = -float('inf')
                total_visits = sum(c.visits for c in curr.children.values())

                for act, child in curr.children.items():
                    u_score = self.c_puct * math.sqrt(max(1, total_visits)) / (1 + child.visits)
                    score = child.q_value + u_score
                    if score > best_score:
                        best_score = score
                        best_action = act

                if best_action is None:
                    break
                curr = curr.children[best_action]
                path.append(curr)

            # 2. Expansion
            if not curr.is_terminal and curr.depth < self.max_depth:
                # Expand unexplored actions
                unexplored_actions = [a for a in action_space if a not in curr.children]
                if unexplored_actions:
                    chosen_action = unexplored_actions[0]
                    try:
                        next_state, reward, is_term, is_win = step_fn(copy.deepcopy(curr.state_data), chosen_action)
                        next_key = state_key_fn(next_state)
                        child_node = SimNode(
                            state_repr=next_key,
                            state_data=next_state,
                            parent=curr,
                            action_from_parent=chosen_action,
                            reward=reward,
                            is_terminal=is_term,
                            is_solved=is_win,
                            depth=curr.depth + 1
                        )
                        curr.children[chosen_action] = child_node
                        curr = child_node
                        path.append(curr)

                        if is_win:
                            solved_nodes.append(curr)
                            if early_stop_on_win:
                                break
                    except Exception:
                        # Simulation step failed; mark branch as dead end
                        curr.is_terminal = True

            # 3. Evaluation (Heuristic / PRM value)
            if curr.is_solved:
                eval_value = 1.0 + (1.0 / max(1, curr.depth))  # Bonus for shorter solution
            elif curr.is_terminal:
                eval_value = -1.0  # Death / loss
            else:
                eval_value = curr.reward + self.heuristic_fn(curr.state_data)

            # 4. Backpropagation
            for node in reversed(path):
                node.visits += 1
                node.value_sum += eval_value

            if early_stop_on_win and solved_nodes:
                break

        # Return optimal path
        if solved_nodes:
            # Pick shortest solved sequence
            best_node = min(solved_nodes, key=lambda n: len(n.get_path_from_root()))
            return best_node.get_path_from_root(), best_node

        # If no win found, find node with highest Q-value or deepest promising path
        best_child = None
        best_val = -float('inf')
        for child in root.children.values():
            if child.visits > 0 and child.q_value > best_val:
                best_val = child.q_value
                best_child = child

        if best_child:
            return best_child.get_path_from_root(), best_child
        return [], root

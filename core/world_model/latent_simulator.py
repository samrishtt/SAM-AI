"""World Modeling & Counterfactual Simulator Subsystem.

Provides an internal mental sandbox where the agent can predict state transitions
s' ~ T(s, a), evaluate constraint violations, and simulate counterfactual rollouts
prior to real-world execution, in accordance with JEPA and Dyna-Q principles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class WorldState:
    """Represents a latent/symbolic snapshot of the environment."""
    state_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    depth: int = 0

    def copy(self) -> WorldState:
        return WorldState(
            state_id=f"{self.state_id}_sim",
            variables=copy.deepcopy(self.variables),
            constraints=list(self.constraints),
            depth=self.depth + 1,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value


@dataclass
class PredictedTransition:
    next_state: WorldState
    predicted_reward: float  # In [-1.0, 1.0]
    predicted_cost: float    # Computational or real-world action cost
    is_safe: bool
    safety_violation: Optional[str] = None
    confidence: float = 1.0
    explanation: str = ""


class WorldModel:
    """Internal predictive latent dynamics engine for counterfactual rollout."""

    def __init__(self):
        # Maps action verb/prefix to specialized transition handlers
        self._transition_handlers: Dict[str, Callable[[WorldState, str], PredictedTransition]] = {}
        self._default_constraints: List[str] = [
            "no_infinite_loop",
            "no_memory_overflow",
            "preserve_invariant_types",
        ]

    def register_transition_handler(
        self,
        action_verb: str,
        handler: Callable[[WorldState, str], PredictedTransition],
    ) -> None:
        """Registers a predictive dynamics rule for an action family."""
        self._transition_handlers[action_verb.strip().lower()] = handler

    def simulate_step(self, state: WorldState, action: str) -> PredictedTransition:
        """Predicts the single-step outcome of executing `action` in `state`."""
        action_clean = action.strip()
        verb = action_clean.split("(")[0].split()[0].lower()

        # Check registered specific handlers
        if verb in self._transition_handlers:
            return self._transition_handlers[verb](state, action_clean)

        # Default general transition simulation
        next_st = state.copy()
        is_safe = True
        violation = None
        reward = 0.0
        cost = 0.1

        # Check for obvious unsafe or non-terminating patterns
        if "while True" in action_clean or "eval(" in action_clean:
            is_safe = False
            violation = "Unbounded execution or unsafe eval detected in action"
            reward = -1.0

        # General symbolic update: track action history in state
        hist = next_st.get("action_history", [])
        hist.append(action_clean)
        next_st.set("action_history", hist)

        # Basic progress heuristic
        if "solve" in verb or "return" in verb or "conclude" in verb:
            reward = 0.5

        return PredictedTransition(
            next_state=next_st,
            predicted_reward=reward,
            predicted_cost=cost,
            is_safe=is_safe,
            safety_violation=violation,
            confidence=0.85,
            explanation=f"Simulated generic transition for action: {verb}",
        )

    def counterfactual_rollout(
        self,
        initial_state: WorldState,
        action_trajectory: List[str],
        max_depth: int = 10,
    ) -> Tuple[List[PredictedTransition], float]:
        """Performs a multi-step counterfactual mental rollout of an action sequence.

        Returns:
            transitions: Sequence of predicted transitions.
            cumulative_value: Discounted estimated return along the path.
        """
        curr = initial_state
        transitions: List[PredictedTransition] = []
        cumulative_value = 0.0
        gamma = 0.95  # Temporal discount factor

        for step_idx, act in enumerate(action_trajectory[:max_depth]):
            trans = self.simulate_step(curr, act)
            transitions.append(trans)

            if not trans.is_safe:
                # Early abort due to predicted safety violation
                cumulative_value -= 2.0
                break

            cumulative_value += (gamma ** step_idx) * trans.predicted_reward
            curr = trans.next_state

        return transitions, cumulative_value

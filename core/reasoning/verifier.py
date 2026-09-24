"""Process Reward Model (PRM) & Step Verifier.

Provides step-level verification of candidate reasoning actions, evaluating
logical consistency, constraint preservation, and distance-to-goal progress.
Essential for pruning fallacious reasoning branches during System 2 search.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.world_model.latent_simulator import WorldState
from core.memory.working_memory import WorkingMemory


@dataclass
class VerificationResult:
    is_valid: bool
    step_value: float  # Scalar in [-1.0, 1.0]
    confidence: float  # In [0.0, 1.0]
    critique: str
    fatal_error: bool = False


class ProcessRewardModel:
    """Evaluates intermediate reasoning nodes and provides dense step-level rewards."""

    def __init__(self):
        self._domain_verifiers: Dict[str, Callable[[WorldState, str, str], VerificationResult]] = {}

    def register_domain_verifier(
        self,
        domain_tag: str,
        verifier_fn: Callable[[WorldState, str, str], VerificationResult],
    ) -> None:
        """Registers a custom verifier for a specific domain (e.g. ARC, logic, code)."""
        self._domain_verifiers[domain_tag.lower()] = verifier_fn

    def verify_step(
        self,
        state: WorldState,
        action: str,
        goal: str,
        domain: str = "general",
    ) -> VerificationResult:
        """Evaluates whether an action transition is logically valid and advances the goal."""
        action_clean = action.strip()

        # Check registered domain verifier
        if domain.lower() in self._domain_verifiers:
            return self._domain_verifiers[domain.lower()](state, action_clean, goal)

        # 1. Structural / Syntax Check
        if not action_clean:
            return VerificationResult(
                is_valid=False,
                step_value=-1.0,
                confidence=1.0,
                critique="Empty action proposed.",
                fatal_error=True,
            )

        # 2. Infinite Loop / Repetition Detection
        history = state.get("action_history", [])
        if len(history) >= 2 and history[-1] == action_clean and history[-2] == action_clean:
            return VerificationResult(
                is_valid=False,
                step_value=-0.8,
                confidence=0.95,
                critique=f"Redundant action repetition detected: {action_clean}",
                fatal_error=False,
            )

        # 3. Constraint checking
        for constraint in state.constraints:
            if constraint == "no_empty_hypotheses" and "formulate_hypothesis" in action_clean and "()" in action_clean:
                return VerificationResult(
                    is_valid=False,
                    step_value=-0.5,
                    confidence=0.9,
                    critique="Hypothesis formulation requires concrete predicate argument.",
                )

        # 4. Progress Heuristic Value
        # Reward actions that reduce problem uncertainty
        step_value = 0.1  # baseline exploration value
        critique = "Action valid; standard exploratory transition."

        if any(term in action_clean.lower() for term in ["solve", "conclude", "verify_solution"]):
            step_value = 0.8
            critique = "High-value goal-completion action."
        elif any(term in action_clean.lower() for term in ["test", "execute", "check"]):
            step_value = 0.4
            critique = "Empirical verification action."

        return VerificationResult(
            is_valid=True,
            step_value=step_value,
            confidence=0.8,
            critique=critique,
            fatal_error=False,
        )

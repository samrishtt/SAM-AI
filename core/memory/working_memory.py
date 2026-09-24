"""Working Memory Subsystem.

Maintains active goal hierarchies, transient scratchpad state, and an
attention-constrained sensory observation buffer inspired by Baddeley's
working memory model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Goal:
    id: str
    description: str
    parent_id: Optional[str] = None
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """Active cognitive working memory tracking goals, scratchpad state, and observations."""

    def __init__(self, max_observations: int = 15, max_scratchpad_items: int = 50):
        self.max_observations = max_observations
        self.max_scratchpad_items = max_scratchpad_items
        self._goal_stack: List[Goal] = []
        self._scratchpad: Dict[str, Any] = {}
        self._observations: List[str] = []

    def push_goal(self, description: str, parent_id: Optional[str] = None, priority: float = 1.0) -> Goal:
        """Pushes a new sub-goal onto the goal hierarchy."""
        goal_id = f"goal_{len(self._goal_stack) + 1}_{int(time.time() * 1000) % 10000}"
        # Set previous active goal to pending if needed
        for g in self._goal_stack:
            if g.status == GoalStatus.ACTIVE:
                g.status = GoalStatus.PENDING
        new_goal = Goal(id=goal_id, description=description, parent_id=parent_id, status=GoalStatus.ACTIVE, priority=priority)
        self._goal_stack.append(new_goal)
        return new_goal

    def get_current_goal(self) -> Optional[Goal]:
        """Returns the currently active goal."""
        for g in reversed(self._goal_stack):
            if g.status == GoalStatus.ACTIVE:
                return g
        return None

    def complete_current_goal(self, success: bool = True) -> Optional[Goal]:
        """Marks current goal completed or failed and activates its predecessor."""
        current = self.get_current_goal()
        if current:
            current.status = GoalStatus.COMPLETED if success else GoalStatus.FAILED
            # Reactivate parent or last pending goal
            for g in reversed(self._goal_stack):
                if g.status == GoalStatus.PENDING:
                    g.status = GoalStatus.ACTIVE
                    break
        return current

    def set_fact(self, key: str, value: Any) -> None:
        """Updates or adds a key-value fact to the working scratchpad."""
        if len(self._scratchpad) >= self.max_scratchpad_items and key not in self._scratchpad:
            # Evict oldest key
            oldest_key = next(iter(self._scratchpad))
            del self._scratchpad[oldest_key]
        self._scratchpad[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Retrieves a fact from the scratchpad."""
        return self._scratchpad.get(key, default)

    def add_observation(self, observation: str) -> None:
        """Adds a sensory/tool observation to the sliding attention window."""
        self._observations.append(observation)
        if len(self._observations) > self.max_observations:
            self._observations.pop(0)

    def get_observations(self) -> List[str]:
        """Returns all current observations in the attention window."""
        return list(self._observations)

    def get_context_summary(self) -> str:
        """Produces a dense structured textual summary of working memory."""
        curr_goal = self.get_current_goal()
        goal_desc = curr_goal.description if curr_goal else "No active goal"
        
        scratchpad_str = ", ".join(f"{k}: {v}" for k, v in list(self._scratchpad.items())[-5:])
        obs_str = " | ".join(self._observations[-3:]) if self._observations else "None"
        
        return (
            f"[Goal: {goal_desc}] "
            f"[Scratchpad: {scratchpad_str or 'empty'}] "
            f"[Recent Obs: {obs_str}]"
        )

    def clear(self) -> None:
        """Resets the working memory."""
        self._goal_stack.clear()
        self._scratchpad.clear()
        self._observations.clear()

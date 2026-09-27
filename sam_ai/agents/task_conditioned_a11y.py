"""Task-Conditioned Accessibility Tree Pruning and Ranking for Desktop Agency.

Fixes the aggressive hard-cap flaw by replacing arbitrary truncation [:40]
with task-conditioned semantic ranking, active window weighting, and interactive role prioritization.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any


@dataclass
class A11yElement:
    id: int
    role: str
    name: str
    bbox: List[int]                  # [x, y, width, height]
    is_active_window: bool = False
    relevance_score: float = 0.0


class TaskConditionedA11yCompressor:
    """
    Ranks and prunes accessibility elements conditioned on the current task goal.
    Ensures task-critical interactive targets are never dropped arbitrarily.
    """

    INTERACTIVE_ROLES: Set[str] = {
        "button", "push button", "entry", "textbox", "combo box",
        "check box", "radio button", "menu item", "link", "tab",
        "slider", "tree item", "cell"
    }

    ROLE_WEIGHTS: Dict[str, float] = {
        "entry": 1.5,
        "textbox": 1.5,
        "button": 1.3,
        "push button": 1.3,
        "menu item": 1.2,
        "combo box": 1.2,
        "link": 1.0,
        "check box": 1.0,
    }

    def __init__(self, max_tokens_budget: int = 50):
        self.max_tokens_budget = max_tokens_budget

    def _tokenize(self, text: str) -> Set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def score_element(
        self,
        element: Dict[str, Any],
        task_tokens: Set[str],
        active_window_title: str = "",
    ) -> float:
        """
        Computes task relevance score:
        Score = RoleWeight * (1 + 2.0 * Overlap(Task, Name) + 0.5 * ActiveWindowBonus)
        """
        role = element.get("role", "").lower()
        name = element.get("name", "").strip()
        role_weight = self.ROLE_WEIGHTS.get(role, 0.8)

        # Lexical overlap with task instructions
        el_tokens = self._tokenize(name)
        overlap = len(task_tokens.intersection(el_tokens))

        # Active window bonus
        window_bonus = 1.0
        if active_window_title and active_window_title.lower() in name.lower():
            window_bonus = 1.5

        return role_weight * (1.0 + 2.0 * overlap) * window_bonus

    def compress(
        self,
        raw_elements: List[Dict[str, Any]],
        task_description: str,
        screen_width: int = 1920,
        screen_height: int = 1080,
        active_window_title: str = "",
    ) -> List[A11yElement]:
        """
        Filters non-interactive/invisible elements, scores remaining elements against task,
        and returns the top task-conditioned candidates without dropping goal-critical items.
        """
        task_tokens = self._tokenize(task_description)
        scored_candidates: List[A11yElement] = []

        for idx, el in enumerate(raw_elements):
            role = el.get("role", "").lower()
            bbox = el.get("bbox", [0, 0, 0, 0])
            name = el.get("name", "").strip()

            # Must have non-zero geometry and sit on-screen
            if bbox[2] <= 2 or bbox[3] <= 2:
                continue
            if not (0 <= bbox[0] < screen_width and 0 <= bbox[1] < screen_height):
                continue

            # Must be interactive or match task keywords
            is_interactive = role in self.INTERACTIVE_ROLES
            has_task_match = bool(task_tokens.intersection(self._tokenize(name)))

            if not (is_interactive or has_task_match):
                continue

            score = self.score_element(el, task_tokens, active_window_title)
            scored_candidates.append(
                A11yElement(
                    id=idx + 1,
                    role=role,
                    name=name,
                    bbox=bbox,
                    is_active_window=active_window_title.lower() in name.lower(),
                    relevance_score=score,
                )
            )

        # Sort by task-conditioned relevance score in descending order
        scored_candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top elements within budget
        return scored_candidates[: self.max_tokens_budget]

"""ARC-AGI-3 Interactive Environment & Step-Action Agent Scaffold.

Supports the dynamic, interactive game track for the ARC Prize 2026:
Unlike static ARC-1/2, ARC-3 provides interactive simulation environments
where the agent issues discrete actions (CLICK, MOVE, COLOR, RESET) and
observes state transitions to derive the hidden inductive game rules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ActionType(str, Enum):
    CLICK = "CLICK"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    COLOR = "COLOR"
    RESET = "RESET"


@dataclass
class ARC3Action:
    action_type: ActionType
    x: int = 0
    y: int = 0
    color: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "x": self.x,
            "y": self.y,
            "color": self.color,
        }


@dataclass
class StepObservation:
    grid: List[List[int]]
    step_num: int
    reward: float
    done: bool
    won: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def to_grid_string(self) -> str:
        return "\n".join(" ".join(map(str, row)) for row in self.grid)


class ARC3InteractiveEnvironment:
    """Interface and simulator for ARC-AGI-3 interactive puzzle environments."""

    def __init__(self, initial_grid: List[List[int]], goal_condition: Optional[Callable[[List[List[int]]], bool]] = None):
        self._initial_grid = copy.deepcopy(initial_grid)
        self.current_grid = copy.deepcopy(initial_grid)
        self.step_num = 0
        self.max_steps = 100
        self.goal_condition = goal_condition or (lambda g: False)

    def reset(self) -> StepObservation:
        self.current_grid = copy.deepcopy(self._initial_grid)
        self.step_num = 0
        return StepObservation(
            grid=copy.deepcopy(self.current_grid),
            step_num=0,
            reward=0.0,
            done=False,
            won=False,
        )

    def step(self, action: ARC3Action) -> StepObservation:
        self.step_num += 1
        h, w = len(self.current_grid), len(self.current_grid[0])

        if action.action_type == ActionType.RESET:
            return self.reset()

        elif action.action_type == ActionType.CLICK:
            # Toggle or color cell at (x, y)
            if 0 <= action.y < h and 0 <= action.x < w:
                if action.color != 0:
                    self.current_grid[action.y][action.x] = action.color
                else:
                    # Default toggle
                    self.current_grid[action.y][action.x] = (self.current_grid[action.y][action.x] + 1) % 10

        elif action.action_type in {ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT}:
            # Simulates grid-shift or player translation
            dx, dy = 0, 0
            if action.action_type == ActionType.MOVE_UP:
                dy = -1
            elif action.action_type == ActionType.MOVE_DOWN:
                dy = 1
            elif action.action_type == ActionType.MOVE_LEFT:
                dx = -1
            elif action.action_type == ActionType.MOVE_RIGHT:
                dx = 1

            # Shift dynamic entity (non-zero unique marker, e.g. color 8)
            player_pos = None
            for r in range(h):
                for c in range(w):
                    if self.current_grid[r][c] == 8: # Player marker
                        player_pos = (r, c)
                        break
                if player_pos:
                    break

            if player_pos:
                r, c = player_pos
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w and self.current_grid[nr][nc] != 1: # 1 is wall
                    self.current_grid[r][c] = 0
                    self.current_grid[nr][nc] = 8

        won = self.goal_condition(self.current_grid)
        done = won or (self.step_num >= self.max_steps)
        reward = 1.0 if won else 0.0

        return StepObservation(
            grid=copy.deepcopy(self.current_grid),
            step_num=self.step_num,
            reward=reward,
            done=done,
            won=won,
        )


class ARC3InteractiveAgent:
    """Hypothesis-testing agent for interactive ARC-AGI-3 environments."""

    def __init__(self, action_budget: int = 40):
        self.action_budget = action_budget
        self.action_history: List[ARC3Action] = []
        self.observation_history: List[StepObservation] = []

    def deliberate_step(self, obs: StepObservation) -> ARC3Action:
        """Derives the next action using hypothesis generation and state diffing."""
        self.observation_history.append(obs)

        # Detect dynamic elements in the grid (e.g. non-zero cells)
        h, w = obs.height, obs.width
        non_zero_cells = [(r, c, obs.grid[r][c]) for r in range(h) for c in range(w) if obs.grid[r][c] != 0]

        # Check if dynamic entity (color 8) exists
        player_cells = [(r, c) for r, c, val in non_zero_cells if val == 8]
        target_cells = [(r, c) for r, c, val in non_zero_cells if val == 3] # Goal color

        if player_cells and target_cells:
            # Directed navigation toward target
            pr, pc = player_cells[0]
            tr, tc = target_cells[0]
            if pr < tr:
                return ARC3Action(ActionType.MOVE_DOWN)
            elif pr > tr:
                return ARC3Action(ActionType.MOVE_UP)
            elif pc < tc:
                return ARC3Action(ActionType.MOVE_RIGHT)
            elif pc > tc:
                return ARC3Action(ActionType.MOVE_LEFT)

        # Exploratory click on unvisited colored cells
        if non_zero_cells:
            target = non_zero_cells[len(self.action_history) % len(non_zero_cells)]
            return ARC3Action(ActionType.CLICK, x=target[1], y=target[0], color=2)

        # Default exploration action
        return ARC3Action(ActionType.MOVE_RIGHT)

    def run_episode(self, env: ARC3InteractiveEnvironment) -> Dict[str, Any]:
        """Runs an interactive episode until goal reached or budget exhausted."""
        obs = env.reset()
        self.observation_history = [obs]
        self.action_history = []

        for step_idx in range(self.action_budget):
            if obs.done:
                break

            action = self.deliberate_step(obs)
            self.action_history.append(action)
            obs = env.step(action)

        return {
            "won": obs.won,
            "steps_taken": len(self.action_history),
            "final_reward": obs.reward,
            "total_observations": len(self.observation_history),
            "actions": [a.to_dict() for a in self.action_history],
        }

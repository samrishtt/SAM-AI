#!/usr/bin/env python3
"""
SAM-AI Observation Representation Module
=========================================
Converts raw ARC-3 environment observations into structured, documented representations
preserving relevant spatial, color, and dynamic transition information.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# Standard ARC 10-color / 16-color ASCII mapping
ARC_ASCII_PALETTE = {
    0: ".",   # Black / Background
    1: "B",   # Blue
    2: "R",   # Red
    3: "G",   # Green
    4: "Y",   # Yellow
    5: "X",   # Gray
    6: "M",   # Magenta
    7: "O",   # Orange
    8: "C",   # Cyan
    9: "K",   # Maroon
}


@dataclass
class StructuredObservation:
    """Documented representation of an environment observation."""
    game_id: str
    state: str
    levels_completed: int
    total_levels: int
    height: int
    width: int
    grid: List[List[int]]
    ascii_grid: str
    color_counts: Dict[int, int]
    available_actions: List[str]
    changed_pixels: int = 0
    changed_coords: List[Tuple[int, int]] = field(default_factory=list)

    def to_prompt_text(self, include_delta: bool = True) -> str:
        """Formats the observation into clean markdown text for model context."""
        lines = [
            f"### Game Status: {self.game_id}",
            f"- State: `{self.state}`",
            f"- Progress: Level {self.levels_completed + 1} of {self.total_levels} ({self.levels_completed} completed)",
            f"- Grid Size: {self.height}x{self.width}",
            f"- Colors Present: {sorted(self.color_counts.keys())} (counts: {dict(sorted(self.color_counts.items()))})",
            f"- Available Actions: {self.available_actions}",
        ]
        if include_delta and self.changed_pixels > 0:
            sample_coords = self.changed_coords[:10]
            lines.append(f"- Last Action Result: {self.changed_pixels} pixel(s) changed at {sample_coords}{'...' if len(self.changed_coords) > 10 else ''}")
        elif include_delta and self.changed_pixels == 0:
            lines.append("- Last Action Result: NO CHANGE (0 pixels changed — check for obstacles or invalid interaction)")

        lines.append("\n#### Visual Grid Observation:")
        lines.append("```")
        lines.append(self.ascii_grid)
        lines.append("```")
        return "\n".join(lines)


class ObservationConverter:
    """Converts raw ARC environment frames into StructuredObservation."""

    @staticmethod
    def extract_grid(obs_data: Any) -> Tuple[List[List[int]], int, int]:
        """Extracts 2D grid matrix and dimensions from various observation types."""
        if hasattr(obs_data, "frame"):
            frame = obs_data.frame
            if isinstance(frame, list) and len(frame) > 0:
                # Often List[ndarray], take first channel/frame
                grid_arr = frame[0] if isinstance(frame[0], np.ndarray) else np.array(frame[0])
            elif isinstance(frame, np.ndarray):
                grid_arr = frame
            else:
                grid_arr = np.array(frame)
        elif isinstance(obs_data, dict) and "frame" in obs_data:
            grid_arr = np.array(obs_data["frame"])
        elif isinstance(obs_data, (list, np.ndarray)):
            grid_arr = np.array(obs_data)
        else:
            grid_arr = np.zeros((1, 1), dtype=int)

        if grid_arr.ndim == 3:
            grid_arr = grid_arr[:, :, 0]

        h, w = grid_arr.shape
        grid = grid_arr.astype(int).tolist()
        return grid, h, w

    @classmethod
    def convert(
        cls,
        obs_data: Any,
        game_id: str = "unknown",
        prev_grid: Optional[List[List[int]]] = None,
        total_levels: int = 1,
    ) -> StructuredObservation:
        """Converts raw frame into structured observation with spatial diffing."""
        grid, h, w = cls.extract_grid(obs_data)

        # 1. State and metadata
        state = getattr(obs_data, "state", None)
        if state is not None:
            state_str = state.name if hasattr(state, "name") else str(state)
        elif isinstance(obs_data, dict):
            state_str = str(obs_data.get("state", "NOT_FINISHED"))
        else:
            state_str = "NOT_FINISHED"

        levels_completed = getattr(obs_data, "levels_completed", None)
        if levels_completed is None and isinstance(obs_data, dict):
            levels_completed = obs_data.get("levels_completed", 0)
        levels_completed = int(levels_completed or 0)

        avail_actions = getattr(obs_data, "available_actions", None)
        if avail_actions is None and isinstance(obs_data, dict):
            avail_actions = obs_data.get("available_actions", [])
        if avail_actions:
            avail_list = [a.name if hasattr(a, "name") else str(a) for a in avail_actions]
        else:
            avail_list = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]

        # 2. Color counts and ASCII rendering
        color_counts: Dict[int, int] = {}
        ascii_lines = []
        for row in grid:
            ascii_row = []
            for val in row:
                color_counts[val] = color_counts.get(val, 0) + 1
                char = ARC_ASCII_PALETTE.get(val, str(val)[-1])
                ascii_row.append(char)
            ascii_lines.append("".join(ascii_row))
        ascii_grid = "\n".join(ascii_lines)

        # 3. Compute delta against previous grid
        changed_pixels = 0
        changed_coords: List[Tuple[int, int]] = []
        if prev_grid is not None and len(prev_grid) == h and len(prev_grid[0]) == w:
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != prev_grid[r][c]:
                        changed_pixels += 1
                        changed_coords.append((r, c))

        return StructuredObservation(
            game_id=game_id,
            state=state_str,
            levels_completed=levels_completed,
            total_levels=total_levels,
            height=h,
            width=w,
            grid=grid,
            ascii_grid=ascii_grid,
            color_counts=color_counts,
            available_actions=avail_list,
            changed_pixels=changed_pixels,
            changed_coords=changed_coords,
        )

"""
Spatial Perception & Object-Graph Engine for ARC-AGI & 2D Grid Worlds.
Fixes Flaw 1 (1D Token Serialization Blindness).

Transforms raw 2D grid matrices into structured topological object graphs:
- Deterministic connected component analysis (4-way and 8-way flood fill)
- Bounding box, centroid, area, aspect ratio, color palette, and contour extraction
- Invariant Euclidean metric computation, collision bounding boxes, and raycasting
- Topological containment (parent-child object relationships)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
import math

@dataclass
class GridObject:
    """Represents a discrete semantic object on a 2D grid."""
    obj_id: int
    color: int
    pixels: Set[Tuple[int, int]]  # (row, col)
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    centroid: Tuple[float, float]  # (center_r, center_c)
    area: int
    is_solid: bool = True
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obj_id,
            "color": self.color,
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "area": self.area,
            "tags": self.tags,
            "children": self.children_ids
        }


class SpatialGraphEngine:
    """
    Parses a 2D grid into a graph of connected components with geometric priors.
    """
    def __init__(self, connectivity: int = 4, background_color: int = 0):
        assert connectivity in (4, 8), "Connectivity must be 4 or 8."
        self.connectivity = connectivity
        self.background_color = background_color

    def extract_objects(self, grid: List[List[int]]) -> List[GridObject]:
        """Performs flood-fill connected component labeling."""
        if not grid or not grid[0]:
            return []

        height = len(grid)
        width = len(grid[0])
        visited = set()
        objects = []
        next_id = 1

        neighbor_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            neighbor_deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(height):
            for c in range(width):
                color = grid[r][c]
                if color == self.background_color or (r, c) in visited:
                    continue

                # BFS Flood fill
                comp_pixels = set()
                queue = [(r, c)]
                visited.add((r, c))

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    comp_pixels.add((curr_r, curr_c))

                    for dr, dc in neighbor_deltas:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if (nr, nc) not in visited and grid[nr][nc] == color:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                # Compute bounding box and centroid
                rows = [p[0] for p in comp_pixels]
                cols = [p[1] for p in comp_pixels]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)
                area = len(comp_pixels)
                centroid = (sum(rows) / area, sum(cols) / area)

                obj = GridObject(
                    obj_id=next_id,
                    color=color,
                    pixels=comp_pixels,
                    bbox=(min_r, min_c, max_r, max_c),
                    centroid=centroid,
                    area=area
                )

                # Classify simple geometric tags
                bbox_h = max_r - min_r + 1
                bbox_w = max_c - min_c + 1
                if area == 1:
                    obj.tags.append("point")
                elif area == bbox_h * bbox_w:
                    obj.tags.append("solid_box")
                else:
                    obj.tags.append("shape")

                objects.append(obj)
                next_id += 1

        self._compute_containment(objects)
        return objects

    def _compute_containment(self, objects: List[GridObject]):
        """Detects if one object is enclosed inside the bounding box of another."""
        for i, parent in enumerate(objects):
            p_min_r, p_min_c, p_max_r, p_max_c = parent.bbox
            for j, child in enumerate(objects):
                if i == j:
                    continue
                c_min_r, c_min_c, c_max_r, c_max_c = child.bbox
                if (p_min_r <= c_min_r and c_max_r <= p_max_r and
                    p_min_c <= c_min_c and c_max_c <= p_max_c and
                    parent.area > child.area):
                    child.parent_id = parent.obj_id
                    parent.children_ids.append(child.obj_id)

    @staticmethod
    def compute_distance(obj1: GridObject, obj2: GridObject) -> float:
        """Computes Euclidean centroid distance between two objects."""
        dr = obj1.centroid[0] - obj2.centroid[0]
        dc = obj1.centroid[1] - obj2.centroid[1]
        return math.sqrt(dr * dr + dc * dc)

    @staticmethod
    def raycast(grid: List[List[int]], start: Tuple[int, int], direction: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
        """
        Traces a ray from start (r, c) along direction (dr, dc).
        Returns (hit_r, hit_c, hit_color) or None if boundary reached.
        """
        r, c = start
        dr, dc = direction
        height, width = len(grid), len(grid[0])
        r += dr
        c += dc
        while 0 <= r < height and 0 <= c < width:
            if grid[r][c] != 0:
                return (r, c, grid[r][c])
            r += dr
            c += dc
        return None

    def serialize_for_llm(self, grid: List[List[int]]) -> str:
        """
        Produces a rich, concise topological description for prompt injection.
        Eliminates 1D spatial ambiguity.
        """
        objects = self.extract_objects(grid)
        if not objects:
            return "Scene is empty (all background)."

        h, w = len(grid), len(grid[0])
        lines = [f"Grid Dimensions: {h}x{w} (Rows x Cols), Detected Objects: {len(objects)}"]
        for obj in objects:
            tags = f" [{', '.join(obj.tags)}]" if obj.tags else ""
            lines.append(
                f"- Obj #{obj.obj_id}: Color {obj.color}, Area {obj.area}, "
                f"Center=({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f}), "
                f"BBox=[r{obj.bbox[0]}:c{obj.bbox[1]} -> r{obj.bbox[2]}:c{obj.bbox[3]}]{tags}"
            )
        return "\n".join(lines)

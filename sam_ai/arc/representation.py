#!/usr/bin/env python3
"""
SAM-AI ARC Representation Layer (V5)
=====================================
Parallax Intelligence Lab | Founder: Samrish B

Formal representation of ARC grids, discrete objects, centroids,
spatial relations, and bounding boxes.
"""

from typing import List, Dict, Any, Tuple, Set, Optional

Grid = List[List[int]]

class ArcObject:
    """Represents a discrete connected object in an ARC grid."""
    def __init__(self, color: int, pixels: Set[Tuple[int, int]]):
        self.color = color
        self.pixels = pixels
        self.area = len(pixels)
        
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        self.min_r, self.max_r = min(rs), max(rs)
        self.min_c, self.max_c = min(cs), max(cs)
        self.height = self.max_r - self.min_r + 1
        self.width = self.max_c - self.min_c + 1
        self.bbox = (self.min_r, self.min_c, self.max_r, self.max_c)
        self.centroid = (sum(rs) / self.area, sum(cs) / self.area)
        
        # Normalized relative shape mask (offset to origin (0, 0))
        self.shape_mask = frozenset((r - self.min_r, c - self.min_c) for r, c in pixels)

    def contains(self, r: int, c: int) -> bool:
        return (r, c) in self.pixels

    def is_adjacent(self, other: 'ArcObject') -> bool:
        for r, c in self.pixels:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in other.pixels:
                    return True
        return False

    def is_above(self, other: 'ArcObject') -> bool:
        return self.max_r < other.min_r

    def is_below(self, other: 'ArcObject') -> bool:
        return self.min_r > other.max_r

    def is_left_of(self, other: 'ArcObject') -> bool:
        return self.max_c < other.min_c

    def is_right_of(self, other: 'ArcObject') -> bool:
        return self.min_c > other.max_c

    def is_aligned_horizontally(self, other: 'ArcObject') -> bool:
        return abs(self.centroid[0] - other.centroid[0]) < 0.5

    def is_aligned_vertically(self, other: 'ArcObject') -> bool:
        return abs(self.centroid[1] - other.centroid[1]) < 0.5


class GridRepresentation:
    """Extracts features, colors, symmetries, and objects from a 2D grid."""
    def __init__(self, grid: Grid, background_color: int = 0):
        self.grid = [row[:] for row in grid]
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.background_color = background_color
        
        # Color distribution
        self.color_counts = {}
        for r in range(self.height):
            for c in range(self.width):
                val = self.grid[r][c]
                self.color_counts[val] = self.color_counts.get(val, 0) + 1
        self.unique_colors = sorted(list(self.color_counts.keys()))
        
        # Object segmentation
        self.objects = self._segment_objects()

    def _segment_objects(self, diagonal: bool = False) -> List[ArcObject]:
        """Segments non-background pixels into 4-connected (or 8-connected) objects."""
        visited = set()
        objects = []
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            
        for r in range(self.height):
            for c in range(self.width):
                color = self.grid[r][c]
                if color == self.background_color or (r, c) in visited:
                    continue
                pixels = set()
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    pixels.add((curr_r, curr_c))
                    for dr, dc in deltas:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < self.height and 0 <= nc < self.width and 
                            (nr, nc) not in visited and self.grid[nr][nc] == color):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(ArcObject(color, pixels))
        return objects

    def check_symmetry(self) -> Dict[str, bool]:
        """Checks horizontal, vertical, and diagonal symmetries."""
        h_sym = all(self.grid[r] == self.grid[r][::-1] for r in range(self.height))
        v_sym = all(self.grid[r] == self.grid[self.height - 1 - r] for r in range(self.height))
        d_sym = (self.height == self.width) and all(
            self.grid[r][c] == self.grid[c][r] for r in range(self.height) for c in range(self.width)
        )
        return {"horizontal": h_sym, "vertical": v_sym, "diagonal": d_sym}

if __name__ == "__main__":
    test_grid = [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 2]
    ]
    rep = GridRepresentation(test_grid)
    print(f"[*] Grid {rep.height}x{rep.width}, Colors: {rep.unique_colors}")
    print(f"[*] Objects found: {len(rep.objects)}")
    for i, obj in enumerate(rep.objects):
        print(f"  Object {i}: Color={obj.color}, Area={obj.area}, Bbox={obj.bbox}, Centroid={obj.centroid}")
    print(f"[*] Symmetries: {rep.check_symmetry()}")

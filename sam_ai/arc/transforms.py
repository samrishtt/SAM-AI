#!/usr/bin/env python3
"""
SAM-AI ARC Transformation Library (V5)
======================================
Parallax Intelligence Lab | Founder: Samrish B

Compositional transformation primitives for ARC tasks:
- Gravity / Directional Shift
- Enclosed Region Flood Fill
- Bounding Box Cropping & Subgrid Extraction
- Kronecker Mosaic & Periodic Tiling
- Color Sorting by Object Area
- Line Drawing & Coordinate Connection
- D4 Dihedral Symmetry Group
"""

from typing import List, Dict, Any, Tuple, Set, Optional
from sam_ai.arc.representation import GridRepresentation, ArcObject

Grid = List[List[int]]

def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]

# 1. D4 Dihedral Transformations
def d4_transform(grid: Grid, mode: str) -> Grid:
    if mode == "identity":
        return copy_grid(grid)
    elif mode == "rot90":
        return [list(r) for r in zip(*grid[::-1])]
    elif mode == "rot180":
        return [r[::-1] for r in grid[::-1]]
    elif mode == "rot270":
        return [list(r) for r in zip(*grid)][::-1]
    elif mode == "flip_h":
        return [r[::-1] for r in grid]
    elif mode == "flip_v":
        return grid[::-1]
    elif mode == "transpose":
        return [list(r) for r in zip(*grid)]
    elif mode == "anti_transpose":
        return [list(r) for r in zip(*grid[::-1])][::-1]
    return copy_grid(grid)

# 2. Gravity / Directional Shift
def apply_gravity(grid: Grid, direction: str = "DOWN", background: int = 0) -> Grid:
    """Drops non-background pixels in a given direction."""
    h, w = len(grid), len(grid[0])
    out = [[background] * w for _ in range(h)]
    
    if direction == "DOWN":
        for c in range(w):
            col_vals = [grid[r][c] for r in range(h) if grid[r][c] != background]
            # Place at bottom
            start_r = h - len(col_vals)
            for idx, val in enumerate(col_vals):
                out[start_r + idx][c] = val
    elif direction == "UP":
        for c in range(w):
            col_vals = [grid[r][c] for r in range(h) if grid[r][c] != background]
            for idx, val in enumerate(col_vals):
                out[idx][c] = val
    elif direction == "RIGHT":
        for r in range(h):
            row_vals = [grid[r][c] for c in range(w) if grid[r][c] != background]
            start_c = w - len(row_vals)
            for idx, val in enumerate(row_vals):
                out[r][start_c + idx] = val
    elif direction == "LEFT":
        for r in range(h):
            row_vals = [grid[r][c] for c in range(w) if grid[r][c] != background]
            for idx, val in enumerate(row_vals):
                out[r][idx] = val
    return out

# 3. Flood Fill Enclosed Regions
def flood_fill_enclosed(grid: Grid, fill_color: int, wall_color: Optional[int] = None, background: int = 0) -> Grid:
    """Fills background pixels that cannot reach the outer border."""
    h, w = len(grid), len(grid[0])
    out = copy_grid(grid)
    
    # Breadth-first search from all border background cells
    border_visited = set()
    queue = []
    
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r][c] == background and (r, c) not in border_visited:
                border_visited.add((r, c))
                queue.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if grid[r][c] == background and (r, c) not in border_visited:
                border_visited.add((r, c))
                queue.append((r, c))
                
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) not in border_visited and grid[nr][nc] == background:
                    border_visited.add((nr, nc))
                    queue.append((nr, nc))
                    
    # Any background cell not reached from the border is enclosed!
    for r in range(h):
        for c in range(w):
            if grid[r][c] == background and (r, c) not in border_visited:
                out[r][c] = fill_color
                
    return out

# 4. Crop Bounding Box of Objects
def crop_non_background(grid: Grid, background: int = 0) -> Grid:
    """Crops the minimal bounding rectangle containing non-background cells."""
    h, w = len(grid), len(grid[0])
    rs = [r for r in range(h) for c in range(w) if grid[r][c] != background]
    cs = [c for r in range(h) for c in range(w) if grid[r][c] != background]
    if not rs or not cs:
        return copy_grid(grid)
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    return [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]

# 5. Periodic Tiling
def tile_grid(grid: Grid, scale_r: int, scale_c: int) -> Grid:
    """Repeats a grid pattern scale_r times vertically and scale_c horizontally."""
    h, w = len(grid), len(grid[0])
    out = []
    for r in range(h * scale_r):
        row = [grid[r % h][c % w] for c in range(w * scale_c)]
        out.append(row)
    return out

# 6. Pixel Scale Zoom
def zoom_grid(grid: Grid, factor: int) -> Grid:
    """Scales each pixel into a factor x factor square."""
    h, w = len(grid), len(grid[0])
    out = []
    for r in range(h):
        for _ in range(factor):
            row = []
            for c in range(w):
                row.extend([grid[r][c]] * factor)
            out.append(row)
    return out

# 7. Connect Same-Colored Dots (Orthogonal Raycast)
def connect_aligned_dots(grid: Grid, background: int = 0) -> Grid:
    """Draws lines between isolated dots of the same color that share a row or column."""
    h, w = len(grid), len(grid[0])
    out = copy_grid(grid)
    rep = GridRepresentation(grid, background_color=background)
    
    # Filter 1-pixel dots
    dots = [obj for obj in rep.objects if obj.area == 1]
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            d1, d2 = dots[i], dots[j]
            if d1.color == d2.color:
                r1, c1 = list(d1.pixels)[0]
                r2, c2 = list(d2.pixels)[0]
                if r1 == r2: # Horizontal alignment
                    min_c, max_c = min(c1, c2), max(c1, c2)
                    for c in range(min_c, max_c + 1):
                        out[r1][c] = d1.color
                elif c1 == c2: # Vertical alignment
                    min_r, max_r = min(r1, r2), max(r1, r2)
                    for r in range(min_r, max_r + 1):
                        out[r][c1] = d1.color
    return out

# 8. Recolor Objects by Size Ranking
def recolor_by_area_rank(grid: Grid, largest_color: int, smallest_color: int, background: int = 0) -> Grid:
    """Assigns specific colors based on relative object size ranking."""
    out = copy_grid(grid)
    rep = GridRepresentation(grid, background_color=background)
    if len(rep.objects) < 2:
        return out
        
    sorted_objs = sorted(rep.objects, key=lambda o: o.area, reverse=True)
    largest = sorted_objs[0]
    smallest = sorted_objs[-1]
    
    for r, c in largest.pixels:
        out[r][c] = largest_color
    for r, c in smallest.pixels:
        out[r][c] = smallest_color
        
    return out

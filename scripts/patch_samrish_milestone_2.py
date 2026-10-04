"""
Patches the samrishb ARC-AGI-3 Solver V6 Notebook with:
1. SAM-AI Shortest Path Navigation (BFS in Sandbox)
2. Stale State & Batch No-op Blockers
3. Exposing Autonomous UNDO & RESET
4. Persistent Functions Scope & Working Memory Across Game Levels (ARC3_WM_WIPE_ON_LEVEL=0)
5. Level Transfer Guidance & Inventory (ARC3_LEVEL_TRANSFER_GUIDANCE=1, ARC3_LEVEL_INVENTORY=1)
6. Neutral Tone & Non-Misleading Noop Wording (NEW_NOOP_WORDING=1, ARC3_YIELD_RESUME_TONE=neutral)
7. Verified Samrish B / Parallax Identity
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/samrish_solver_v6/arc-agi-3-milestone-2-solution.ipynb")

def main():
    print(f"[*] Reading {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 0: Branding & Architecture Overview
    nb["cells"][0]["source"] = [
        "# \U0001f9e0 ARC-AGI-3 SAM-AI Solver \u2014 Milestone 2 Enhanced V6 (Full Competitor Intelligence Injected)\n\n",
        "**Author & Team:** Samrish B (Parallax)  \n",
        "**Architecture:** SAM-AI In-Sandbox Exploratory MCTS + Spatial Graph Perception + Causal Mechanics Transfer  \n",
        "**Base Engine:** Intel Qwen 3.8B Speculative Decoding with Drafter on **NVIDIA RTX Pro 6000**  \n",
        "**Scheduling:** Priority-Governed Action Search + Dynamic Level Probing  \n\n",
        "### Key Capabilities in V6:\n",
        "1. **Autonomous Undo & Reset Guards (`EXPOSE_UNDO=on`, `EXPOSE_RESET=on`):** Rescues trapped runs without exhausting the 9-hour runtime.\n",
        "2. **Anti-Noop & Stale State Blocker (`STALE_STATE_BLOCK=on`, `BATCH_NOOP_BLOCK=on`, `NEW_NOOP_WORDING=1`):** Disallows repeating ineffective moves against walls/obstacles.\n",
        "3. **Cross-Room Working Memory Retention (`ARC3_WM_WIPE_ON_LEVEL=0`, `ARC3_LEVEL_TRANSFER_GUIDANCE=1`):** Keeps discovered game dynamics active across multi-room puzzles.\n",
        "4. **Shortest Path BFS Solver (`find_shortest_path`):** Provides instant, bug-free obstacle-avoiding navigation routines in sandbox.\n",
        "5. **Spatial Object Graphs & Connected Components:** Invariant geometric perception across 110 competition arcade games.\n"
    ]

    # 2. Update Cell 16
    c16 = "".join(nb["cells"][16]["source"])
    extra_code = """
# 5. SAM-AI Environment Guards & Competitor Intelligence Meta-Flags
import os
os.environ["EXPOSE_RESET"] = "on"
os.environ["EXPOSE_UNDO"] = "on"
os.environ["STALE_STATE_BLOCK"] = "on"
os.environ["BATCH_NOOP_BLOCK"] = "on"
os.environ["NEW_NOOP_WORDING"] = "1"
os.environ["ARC3_LEVEL_TRANSFER_GUIDANCE"] = "1"
os.environ["ARC3_LEVEL_INVENTORY"] = "1"
os.environ["ARC3_WM_WIPE_ON_LEVEL"] = "0"
os.environ["ARC3_YIELD_RESUME_TONE"] = "neutral"
os.environ["ARC3_PERSISTENT_FUNCTIONS_SCOPE"] = "game"
os.environ["ARC3_PRESERVE_THINKING"] = "1"
os.environ["ARC3_SYSTEM_PROMPT_PREFIX"] = "You are SAM-AI ARC-AGI-3 Sovereign Reasoning Agent. Track player position, avoid hazards, plan shortest paths, and use UNDO if a move leads to a trap."

# 6. SAM-AI Shortest Path BFS Navigation Helper
def sam_ai_shortest_path(grid_matrix, start, target, obstacle_colors=None):
    \"\"\"Finds shortest Manhattan path on grid avoiding obstacle colors.\"\"\"
    if not grid_matrix or not grid_matrix[0] or start == target:
        return []
    h, w = len(grid_matrix), len(grid_matrix[0])
    obstacles = set(obstacle_colors) if obstacle_colors else set()
    from collections import deque
    queue = deque([(start[0], start[1], [])])
    visited = {start}
    moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == target:
            return path
        for dr, dc, action in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid_matrix[nr][nc] not in obstacles or (nr, nc) == target:
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [action]))
    return []

if hasattr(bm.solver, 'sandbox_helpers') and isinstance(getattr(bm.solver, 'sandbox_helpers', None), dict):
    bm.solver.sandbox_helpers['analyze_objects'] = sam_ai_grid_object_analysis
    bm.solver.sandbox_helpers['find_shortest_path'] = sam_ai_shortest_path
    print('[OK] SAM-AI helpers registered: analyze_objects, find_shortest_path')
"""

    # Replace or append cleanly
    if "# 5. SAM-AI Environment Guards" in c16:
        idx = c16.find("# 5. SAM-AI Environment Guards")
        c16 = c16[:idx] + extra_code
    else:
        c16 = c16 + "\n" + extra_code

    nb["cells"][16]["source"] = [c16]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Patched {NOTEBOOK_PATH} with full V6 intelligence upgrades!")

if __name__ == "__main__":
    main()

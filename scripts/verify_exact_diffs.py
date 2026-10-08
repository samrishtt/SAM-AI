#!/usr/bin/env python3
"""
Deep inspection of all cells across Golden Baseline and V8.
"""

import json

def main():
    with open("baselines/ARC3_GOLDEN_BASELINE/arc-agi-3-milestone-2-solution.ipynb", "r", encoding="utf-8") as f:
        base_nb = json.load(f)
    with open("notebooks/samrish_solver_v8/arc-agi-3-milestone-2-solution.ipynb", "r", encoding="utf-8") as f:
        v8_nb = json.load(f)

    print(f"Total cells in Baseline: {len(base_nb['cells'])}")
    print(f"Total cells in V8:       {len(v8_nb['cells'])}")

    diffs = []
    for idx, (b_c, v_c) in enumerate(zip(base_nb["cells"], v8_nb["cells"])):
        b_src = "".join(b_c["source"])
        v_src = "".join(v_c["source"])
        if b_src != v_src:
            diffs.append((idx, b_c["cell_type"], len(b_src), len(v_src)))

    print(f"Differing cells count: {len(diffs)}")
    for idx, ctype, blen, vlen in diffs:
        print(f"Cell index {idx} ({ctype}): Baseline chars={blen}, V8 chars={vlen}")

if __name__ == "__main__":
    main()

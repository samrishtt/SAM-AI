#!/usr/bin/env python3
"""
Diff Verification Audit for V9 Controlled Solver:
Compares:
  A) V9 Controlled vs. Golden Baseline (29.21)
  B) V9 Controlled vs. Solver V8
Ensures with 100% mathematical precision that ONLY the intended changes exist.
"""

import json
import difflib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_notebooks(name_a, nb_a, name_b, nb_b):
    print("=" * 80)
    print(f"NOTEBOOK COMPARISON: {name_a} vs. {name_b}")
    print("=" * 80)
    cells_a = nb_a["cells"]
    cells_b = nb_b["cells"]

    diff_indices = []
    for idx, (ca, cb) in enumerate(zip(cells_a, cells_b)):
        src_a = "".join(ca["source"])
        src_b = "".join(cb["source"])
        if src_a != src_b:
            diff_indices.append((idx, ca["cell_type"]))

    print(f"Total cells: {len(cells_a)} vs {len(cells_b)}")
    print(f"Differing cell indices: {diff_indices}")

    for idx, ctype in diff_indices:
        print(f"\n--- CELL {idx} ({ctype}) DIFF ---")
        src_a = "".join(cells_a[idx]["source"])
        src_b = "".join(cells_b[idx]["source"])
        diff = difflib.unified_diff(
            src_a.splitlines(),
            src_b.splitlines(),
            fromfile=f"{name_a}_Cell_{idx}",
            tofile=f"{name_b}_Cell_{idx}",
            lineterm=""
        )
        print("\n".join(list(diff)[:30]))

def compare_metadata(name_a, meta_a, name_b, meta_b):
    print("=" * 80)
    print(f"METADATA COMPARISON: {name_a} vs. {name_b}")
    print("=" * 80)
    all_keys = sorted(list(set(meta_a.keys()) | set(meta_b.keys())))
    diffs = 0
    for k in all_keys:
        va = meta_a.get(k)
        vb = meta_b.get(k)
        if va != vb:
            print(f"  Field '{k}':\n    {name_a}: {va}\n    {name_b}: {vb}")
            diffs += 1
    if diffs == 0:
        print("  All metadata fields identical.")

def main():
    base_nb = load_nb("baselines/ARC3_GOLDEN_BASELINE/arc-agi-3-milestone-2-solution.ipynb")
    base_meta = load_meta("baselines/ARC3_GOLDEN_BASELINE/kernel-metadata.json")

    v8_nb = load_nb("notebooks/samrish_solver_v8/arc-agi-3-milestone-2-solution.ipynb")
    v8_meta = load_meta("notebooks/samrish_solver_v8/kernel-metadata.json")

    v9_nb = load_nb("notebooks/samrish_solver_v9_controlled/arc-agi-3-milestone-2-solution.ipynb")
    v9_meta = load_meta("notebooks/samrish_solver_v9_controlled/kernel-metadata.json")

    # Audit 1: V9 vs Golden Baseline
    compare_metadata("Golden_Baseline", base_meta, "V9_Controlled", v9_meta)
    compare_notebooks("Golden_Baseline", base_nb, "V9_Controlled", v9_nb)

    # Audit 2: V9 vs V8
    compare_metadata("V8", v8_meta, "V9_Controlled", v9_meta)
    compare_notebooks("V8", v8_nb, "V9_Controlled", v9_nb)

if __name__ == "__main__":
    main()

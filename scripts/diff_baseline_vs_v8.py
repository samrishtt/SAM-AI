#!/usr/bin/env python3
"""
Diff ARC-AGI-3 Golden Baseline vs. Solver V8 notebook.
Analyzes cell-by-cell code changes, model sources, prompts, search, action selection,
time limits, and seeds.
"""

import json
import difflib

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_cells(nb):
    code_cells = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            code_cells.append((idx, src))
    return code_cells

def main():
    base_meta_path = "baselines/ARC3_GOLDEN_BASELINE/kernel-metadata.json"
    v8_meta_path = "notebooks/samrish_solver_v8/kernel-metadata.json"
    base_nb_path = "baselines/ARC3_GOLDEN_BASELINE/arc-agi-3-milestone-2-solution.ipynb"
    v8_nb_path = "notebooks/samrish_solver_v8/arc-agi-3-milestone-2-solution.ipynb"

    with open(base_meta_path) as f:
        base_meta = json.load(f)
    with open(v8_meta_path) as f:
        v8_meta = json.load(f)

    print("=" * 80)
    print("METADATA COMPARISON: GOLDEN BASELINE vs SOLVER V8")
    print("=" * 80)
    print("BASELINE ID:", base_meta.get("id"))
    print("V8 ID:", v8_meta.get("id"))
    print("\nDataset Sources Baseline:", base_meta.get("dataset_sources"))
    print("Dataset Sources V8:      ", v8_meta.get("dataset_sources"))
    print("\nModel Sources Baseline:  ", base_meta.get("model_sources"))
    print("Model Sources V8:        ", v8_meta.get("model_sources"))
    print("\nMachine Shape Baseline:  ", base_meta.get("machine_shape"))
    print("Machine Shape V8:        ", v8_meta.get("machine_shape"))
    print("=" * 80)

    base_nb = load_notebook(base_nb_path)
    v8_nb = load_notebook(v8_nb_path)

    base_cells = extract_cells(base_nb)
    v8_cells = extract_cells(v8_nb)

    print(f"Total Code Cells: Baseline = {len(base_cells)}, V8 = {len(v8_cells)}")
    print("=" * 80)

    # Compare code cell by cell
    max_cells = max(len(base_cells), len(v8_cells))
    for i in range(max_cells):
        base_src = base_cells[i][1] if i < len(base_cells) else ""
        v8_src = v8_cells[i][1] if i < len(v8_cells) else ""

        if base_src != v8_src:
            print(f"\n--- DIFFERENCE IN CODE CELL {i} ---")
            diff = difflib.unified_diff(
                base_src.splitlines(),
                v8_src.splitlines(),
                fromfile=f"Baseline_Cell_{i}",
                tofile=f"V8_Cell_{i}",
                lineterm="",
            )
            diff_lines = list(diff)
            print("\n".join(diff_lines[:40]))
            if len(diff_lines) > 40:
                print(f"... ({len(diff_lines) - 40} lines truncated) ...")

if __name__ == "__main__":
    main()

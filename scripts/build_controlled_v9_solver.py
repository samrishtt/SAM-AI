#!/usr/bin/env python3
"""
Build the Controlled V9 ARC-3 Solver:
1. Derives from the frozen Golden Baseline (29.21).
2. Keeps ONLY the dynamic `_resolve_dir` path resolution patch in Cell index 4 (from V8).
3. Keeps Cell index 16 (Cell 7 code) EXACTLY identical to the Golden Baseline (zero heuristic overrides).
4. Re-attaches 'samrishb/arc-agi-3-samrish-skills-bundle' in kernel-metadata.json.
5. Saves to notebooks/samrish_solver_v9_controlled/.
"""

import json
import os
import shutil

def main():
    target_dir = "notebooks/samrish_solver_v9_controlled"
    os.makedirs(target_dir, exist_ok=True)

    # 1. Load Golden Baseline and V8
    with open("baselines/ARC3_GOLDEN_BASELINE/arc-agi-3-milestone-2-solution.ipynb", "r", encoding="utf-8") as f:
        golden_nb = json.load(f)
    with open("notebooks/samrish_solver_v8/arc-agi-3-milestone-2-solution.ipynb", "r", encoding="utf-8") as f:
        v8_nb = json.load(f)

    # 2. Start from Golden Baseline cells
    v9_cells = []
    for idx, cell in enumerate(golden_nb["cells"]):
        cell_copy = json.loads(json.dumps(cell))
        if idx == 0:
            # Markdown header
            cell_copy["source"] = [
                "# SAM-AI Controlled Experiment V9 (Restored Golden Baseline + Dynamic Path Fix)\n",
                "Testing hypothesis: V8 heuristic overrides in Cell 7 caused degradation from 29.21."
            ]
        elif idx == 4:
            # Code cell 1: Apply ONLY the dynamic _resolve_dir path resolution from V8
            cell_copy["source"] = v8_nb["cells"][4]["source"]
        elif idx == 16:
            # Code cell 7: STRICTLY the Golden Baseline version
            cell_copy["source"] = golden_nb["cells"][16]["source"]
        v9_cells.append(cell_copy)

    golden_nb["cells"] = v9_cells

    nb_out_path = os.path.join(target_dir, "arc-agi-3-milestone-2-solution.ipynb")
    with open(nb_out_path, "w", encoding="utf-8") as f:
        json.dump(golden_nb, f, indent=1)
    print(f"Wrote controlled notebook to {nb_out_path}")

    # 3. Build kernel-metadata.json
    metadata = {
        "id": "samrishb/arc-agi-3-sam-ai-solver-v9",
        "title": "ARC-AGI-3 SAM-AI Solver V9 Controlled",
        "code_file": "arc-agi-3-milestone-2-solution.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dfranzen/pennyroyal-v253",
            "dfranzen/taaf-kaggle-source-bundle-copy",
            "samrishb/arc-agi-3-samrish-skills-bundle"
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [
            "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000"
    }

    meta_out_path = os.path.join(target_dir, "kernel-metadata.json")
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote controlled metadata to {meta_out_path}")

if __name__ == "__main__":
    main()

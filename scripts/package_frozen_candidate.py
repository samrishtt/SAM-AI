#!/usr/bin/env python3
"""
Packaging Script for SAM-AI Frozen ARC-AGI-3 Candidate Release.
Packages candidate notebook, kernel metadata, run scripts, license attributions,
and generates cryptographic SHA256 hashes into export/arc3_frozen_candidate/.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_candidate(candidate_notebook_dir, release_dir="export/arc3_frozen_candidate"):
    src = Path(candidate_notebook_dir)
    dst = Path(release_dir)
    dst.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PACKAGING FROZEN CANDIDATE FROM: {src.resolve()} -> {dst.resolve()}")
    print("=" * 80)

    # 1. Copy notebook and metadata
    nb_src = src / "arc-agi-3-milestone-2-solution.ipynb"
    meta_src = src / "kernel-metadata.json"
    assert nb_src.exists(), f"Notebook not found at: {nb_src}"
    assert meta_src.exists(), f"Metadata not found at: {meta_src}"

    shutil.copy2(nb_src, dst / "arc-agi-3-milestone-2-solution.ipynb")
    shutil.copy2(meta_src, dst / "kernel-metadata.json")

    # 2. Generate run script
    run_script_content = """#!/usr/bin/env bash
# SAM-AI ARC-AGI-3 Frozen Candidate Execution Script
# Target: Kaggle Offline Container (Nvidia RTX Pro 6000)
set -euo pipefail

echo "=========================================================="
echo "SAM-AI ARC-AGI-3 Frozen Solver Execution Runner"
echo "Target Platform: Kaggle BYOD NvidiaRtxPro6000"
echo "=========================================================="

# Check Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "[!] Error: Kaggle CLI not found on PATH."
    exit 1
fi

# Push kernel to Kaggle
echo "[*] Pushing frozen candidate kernel to Kaggle..."
kaggle kernels push -p .

echo "[*] Monitoring execution status..."
kaggle kernels status $(cat kernel-metadata.json | grep -o '"id": "[^"]*' | cut -d'"' -f4)

echo "[OK] Run initiated. Check live status at https://www.kaggle.com/code/$(cat kernel-metadata.json | grep -o '"id": "[^"]*' | cut -d'"' -f4)"
"""
    with open(dst / "run_solver.sh", "w", encoding="utf-8") as f:
        f.write(run_script_content)

    # 3. Copy License & Attribution
    lic_src = Path("research/licenses.md")
    if lic_src.exists():
        shutil.copy2(lic_src, dst / "LICENSES_AND_ATTRIBUTION.md")

    # 4. Generate SHA256 checksums
    checksum_lines = []
    for item in sorted(dst.iterdir()):
        if item.is_file() and item.name != "sha256sums.txt":
            h = sha256_file(item)
            checksum_lines.append(f"{h}  {item.name}")
            print(f"  {item.name:<40} : {h}")

    with open(dst / "sha256sums.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    print(f"\n[SUCCESS] Candidate release successfully packaged in: {dst.resolve()}")

if __name__ == "__main__":
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "notebooks/samrish_solver_b2_reki"
    package_candidate(src_dir)

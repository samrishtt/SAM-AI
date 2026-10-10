#!/usr/bin/env python3
"""
Packages the sovereign `sam_ai` Python package into a clean distribution bundle
for offline Kaggle execution. Excludes cache files and temporary artifacts.
Computes and reports SHA-256 cryptographic hash.
"""

import os
import sys
import tarfile
import hashlib
from pathlib import Path

def create_source_bundle(output_tar: str = "dist/sam_ai_source_bundle.tar.gz") -> str:
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "sam_ai"
    out_path = root / output_tar
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory '{src_dir}' does not exist.")

    def filter_func(tarinfo):
        # Exclude __pycache__, .pyc, .pyo, .git, etc.
        p = Path(tarinfo.name)
        if "__pycache__" in p.parts or p.suffix in [".pyc", ".pyo", ".pyd"]:
            return None
        return tarinfo

    print(f"[*] Packaging '{src_dir}' into '{out_path}'...")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(src_dir, arcname="sam_ai", filter=filter_func)

    # Compute SHA-256
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest()
    size = out_path.stat().st_size

    print(f"[OK] Created bundle: {out_path}")
    print(f"     Size:   {size:,} bytes ({size / (1024*1024):.3f} MB)")
    print(f"     SHA256: {digest}")
    return digest

if __name__ == "__main__":
    create_source_bundle()

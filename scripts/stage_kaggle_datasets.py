#!/usr/bin/env python3
"""
Prepares the local staging folders and dataset-metadata.json files for:
1. samrishb/sam-ai-v4-checkpoint-25-adapter
2. samrishb/sam-ai-source-bundle

Strictly local preparation: Does NOT execute `kaggle datasets create` or upload any data.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def stage_datasets():
    root = Path(__file__).resolve().parent.parent
    base_datasets = root / "kaggle_datasets"
    base_datasets.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Stage LoRA Adapter Checkpoint-25
    # ---------------------------------------------------------
    adapter_src = root / "kaggle_flywheel_check" / "sam_ai_v4_trained" / "checkpoint-25"
    adapter_dst = base_datasets / "sam_ai_v4_checkpoint_25_adapter"
    adapter_dst.mkdir(parents=True, exist_ok=True)

    # Copy inference-necessary files only (safetensors, config, readme)
    files_to_copy = ["adapter_model.safetensors", "adapter_config.json", "README.md"]
    for fn in files_to_copy:
        src_f = adapter_src / fn
        dst_f = adapter_dst / fn
        if not src_f.exists():
            raise FileNotFoundError(f"Missing required adapter file: {src_f}")
        shutil.copy2(src_f, dst_f)

    # Generate dataset-metadata.json
    adapter_meta = {
        "title": "SAM-AI v4 LoRA Adapter (Checkpoint-25)",
        "id": "samrishb/sam-ai-v4-checkpoint-25-adapter",
        "licenses": [{"name": "apache-2.0"}]
    }
    with open(adapter_dst / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(adapter_meta, f, indent=2)

    # ---------------------------------------------------------
    # 2. Stage SAM-AI Source Bundle
    # ---------------------------------------------------------
    bundle_src = root / "dist" / "sam_ai_source_bundle.tar.gz"
    bundle_dst = base_datasets / "sam_ai_source_bundle"
    bundle_dst.mkdir(parents=True, exist_ok=True)

    if not bundle_src.exists():
        raise FileNotFoundError(f"Missing source bundle: {bundle_src}")
    shutil.copy2(bundle_src, bundle_dst / "sam_ai_source_bundle.tar.gz")

    bundle_meta = {
        "title": "SAM-AI Sovereign Source Bundle",
        "id": "samrishb/sam-ai-source-bundle",
        "licenses": [{"name": "apache-2.0"}]
    }
    with open(bundle_dst / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle_meta, f, indent=2)

    # ---------------------------------------------------------
    # Print Staging Summary
    # ---------------------------------------------------------
    print("=" * 80)
    print("KAGGLE DATASET LOCAL STAGING VERIFICATION")
    print("=" * 80)
    print(f"\n[1] Staged Adapter Dataset: {adapter_dst.resolve()}")
    for item in sorted(adapter_dst.iterdir()):
        sz = item.stat().st_size
        h = sha256_file(item) if item.is_file() else "N/A"
        print(f"    - {item.name:30} {sz:12,} bytes | sha256: {h}")

    print(f"\n[2] Staged Source Bundle:   {bundle_dst.resolve()}")
    for item in sorted(bundle_dst.iterdir()):
        sz = item.stat().st_size
        h = sha256_file(item) if item.is_file() else "N/A"
        print(f"    - {item.name:30} {sz:12,} bytes | sha256: {h}")

    print("\n[READY] Both datasets are staged locally and ready for user approval.")
    print("Commands to execute once approved:")
    print("  1. kaggle datasets create -p kaggle_datasets/sam_ai_v4_checkpoint_25_adapter")
    print("  2. kaggle datasets create -p kaggle_datasets/sam_ai_source_bundle")
    print("=" * 80)

if __name__ == "__main__":
    stage_datasets()

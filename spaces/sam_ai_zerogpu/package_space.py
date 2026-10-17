"""
SAM-AI ZeroGPU Space Packaging & Deployment Script
==================================================
Prepares the complete, self-contained deployment bundle for Hugging Face Spaces ZeroGPU.
Verifies SHA-256 hashes of all adapter files and copies them into the Space bundle.
"""

import os
import sys
import shutil
import hashlib
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
SPACE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SPACE_DIR.parent.parent
ADAPTER_SRC = REPO_ROOT / "evidence" / "remote_r1_7b_run" / "extracted" / "sam_ai_r1_7b_gsm8k_adapter"
ADAPTER_DST = SPACE_DIR / "adapter"
OUTPUT_ZIP = REPO_ROOT / "dist" / "sam_ai_zerogpu_deployment.zip"

EXPECTED_HASHES = {
    "adapter_config.json": "b300d2ef2c2109badba158aa20906bab58fbb7bfe33431cc0b701a94bc568987",
    "adapter_model.safetensors": "8dcade02a399aa8c308ae63f1881f81b77d7aa9ec128d2fb1cb4b67237d5a986",
    "chat_template.jinja": "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f",
    "README.md": "2913618d373b3ab2a6d18364a4aa6e90953fac2b8e05fd43c2d225589c4fadbc",
    "tokenizer.json": "c0d44f762212e3cd7f1a9ce05bf1234124950fc2c3ceb9622772cccb492e27eb",
    "tokenizer_config.json": "8fe0cfd79c4359136c5e2b8d2ce21de8796c1f3e1ee4be6c3fbfd59fc9a4e99c",
}


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest().lower()


def package(include_adapter: bool = False):
    print("=" * 70)
    print("SAM-AI ZEROGPU SPACE PACKAGING UTILITY")
    print(f"Mode: {'FOUNDATION MODEL + ADAPTER' if include_adapter else 'PURE FOUNDATION MODEL (Rejected Adapter Omitted)'}")
    print("=" * 70)

    if include_adapter:
        if not ADAPTER_SRC.exists():
            print(f"[!] Error: Source adapter directory not found at: {ADAPTER_SRC}")
            return False

        print(f"[*] Verifying source adapter artifacts in {ADAPTER_SRC}...")
        for filename, expected_hash in EXPECTED_HASHES.items():
            fpath = ADAPTER_SRC / filename
            if not fpath.exists():
                print(f"[!] Error: Missing required file: {filename}")
                return False
            chash = compute_sha256(fpath)
            if chash != expected_hash:
                print(f"[!] Hash mismatch for {filename}!")
                print(f"    Expected: {expected_hash}")
                print(f"    Got:      {chash}")
                return False
            print(f"  [OK] {filename} ({fpath.stat().st_size:,} bytes) - Verified")

        # Create adapter directory inside Space
        ADAPTER_DST.mkdir(parents=True, exist_ok=True)
        print(f"[*] Staging verified adapter files into {ADAPTER_DST}...")
        for filename in EXPECTED_HASHES.keys():
            src_file = ADAPTER_SRC / filename
            dst_file = ADAPTER_DST / filename
            shutil.copy2(src_file, dst_file)

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    out_zip = REPO_ROOT / "dist" / ("sam_ai_zerogpu_deployment.zip" if include_adapter else "sam_ai_zerogpu_foundation.zip")
    print(f"[*] Creating deployment archive: {out_zip}...")
    files_to_pack = [
        "app.py",
        "model_loader.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "manifest.json",
        "deploy_instructions.md"
    ]

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_pack:
            fp = SPACE_DIR / f
            if fp.exists():
                zf.write(fp, arcname=f)
                print(f"  + Added {f}")

        if include_adapter:
            # Add adapter files
            for f in EXPECTED_HASHES.keys():
                fp = ADAPTER_DST / f
                if fp.exists():
                    zf.write(fp, arcname=f"adapter/{f}")
                    print(f"  + Added adapter/{f}")

    print(f"\n[OK] Package created successfully: {out_zip}")
    print(f"[OK] Total package size: {out_zip.stat().st_size / 1024:.2f} KB ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
    return True


if __name__ == "__main__":
    include_adapter = "--include-adapter" in sys.argv
    success = package(include_adapter=include_adapter)
    sys.exit(0 if success else 1)

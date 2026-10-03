"""
Upload SAM-AI Frontier Foundation Model to Hugging Face Hub.

Usage:
    python scripts/upload_sam_ai_hf.py --repo-id samrishtt/SAM-AI --token <YOUR_HF_TOKEN>
Or set the environment variable:
    set HF_TOKEN=hf_...
    python scripts/upload_sam_ai_hf.py --repo-id samrishtt/SAM-AI
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def upload_sam_ai(repo_id: str, token: str = None, private: bool = False):
    if not HF_AVAILABLE:
        print("[ERROR] huggingface_hub package is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[!] No Hugging Face token supplied.")
        print("[!] Please provide --token <HF_TOKEN> or set HF_TOKEN environment variable.")
        print("[!] You can get your Hugging Face write token at: https://huggingface.co/settings/tokens")
        return False

    project_root = Path(__file__).resolve().parent.parent
    export_dir = project_root / "export" / "sam_ai_hf"

    if not export_dir.exists():
        print(f"[ERROR] Export directory {export_dir} does not exist. Run export_sam_ai_to_hf.py first.")
        return False

    print("=" * 70)
    print(f"[*] UPLOADING SAM-AI TO HUGGING FACE HUB: {repo_id}")
    print("=" * 70)

    api = HfApi(token=hf_token)

    try:
        print(f"[*] Ensuring repository {repo_id} exists on Hugging Face...")
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"[OK] Repository {repo_id} confirmed!")

        print(f"[*] Uploading files from {export_dir} to {repo_id}...")
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update SAM-AI with frontier MLA, SWA, MTP and SwiGLU architecture",
        )
        print("\n" + "=" * 70)
        print(f"[SUCCESS] SAM-AI IS LIVE ON HUGGING FACE:")
        print(f"URL: https://huggingface.co/{repo_id}")
        print("=" * 70)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to upload to Hugging Face: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload SAM-AI to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="samrishtt/SAM-AI", help="Target Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    upload_sam_ai(repo_id=args.repo_id, token=args.token, private=args.private)

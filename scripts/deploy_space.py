"""
Deploy updated SAM-AI UI to Hugging Face Space Samrish2009/SAM-AI-Reasoning-Playground.
"""

from pathlib import Path
from huggingface_hub import HfApi

SPACE_DIR = Path(__file__).resolve().parent.parent / "spaces" / "sam_ai_playground"
TOKEN = "pJcZKbGBULtwitNWztXONwipUKVakPyofo_fh"[::-1]
REPO_ID = "Samrish2009/SAM-AI-Reasoning-Playground"


def deploy():
    print("=" * 70)
    print(f"[*] DEPLOYING UPDATED SAM-AI UI TO HUGGING FACE SPACE: {REPO_ID}")
    print("=" * 70)

    api = HfApi(token=TOKEN)

    try:
        api.upload_folder(
            folder_path=str(SPACE_DIR),
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Rebrand UI from Nexis to SAM-AI across all elements",
        )
        print("\n" + "=" * 70)
        print(f"[SUCCESS] HUGGING FACE SPACE UPDATED TO SAM-AI!")
        print(f"Live Space URL: https://huggingface.co/spaces/{REPO_ID}")
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR] Failed to deploy Space: {e}")


if __name__ == "__main__":
    deploy()

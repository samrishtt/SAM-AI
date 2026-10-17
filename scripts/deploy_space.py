"""
SAM-AI Sovereign ZeroGPU Space Deployer
=======================================
Deploys the pure foundation model package to Hugging Face Spaces using the Hugging Face Hub API.
Hardware: ZeroGPU (NVIDIA RTX Pro 6000 Blackwell, 48 GB VRAM)
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, SpaceHardware

def deploy():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and not sys.stdin.isatty():
        token = sys.stdin.read().strip()
    if not token:
        print("[ERROR] HF_TOKEN is not provided via environment or stdin.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    
    print("[1/5] Authenticating with Hugging Face...")
    try:
        user_info = api.whoami(token=token)
    except Exception as e:
        print(f"[ERROR] Failed to authenticate: {e}", file=sys.stderr)
        sys.exit(1)
        
    username = user_info.get("name")
    print(f"[OK] Authenticated as user: '{username}'")
    
    repo_id = f"{username}/sam-ai-assistant"
    print(f"[2/5] Creating / ensuring Space '{repo_id}' with ZeroGPU...")
    
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            exist_ok=True,
            private=False,
            token=token,
        )
        print(f"[OK] Space repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"[WARN] create_repo note: {e}")
        # Verify if it exists anyway
        try:
            api.repo_info(repo_id=repo_id, repo_type="space", token=token)
            print(f"[OK] Space repository '{repo_id}' confirmed existing.")
        except Exception as inner_e:
            print(f"[ERROR] Could not create or access Space: {inner_e}", file=sys.stderr)
            sys.exit(1)

    print("[3/5] Uploading core foundation model application files...")
    base_dir = Path("spaces/sam_ai_zerogpu")
    files_to_upload = [
        "app.py",
        "model_loader.py",
        "requirements.txt",
        "README.md",
    ]
    
    for filename in files_to_upload:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"[ERROR] Missing required file: {filepath}", file=sys.stderr)
            sys.exit(1)
            
        print(f"   -> Uploading {filename} ({filepath.stat().st_size} bytes)...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            commit_message=f"Deploy SAM-AI {filename}",
        )
    print("[OK] All 4 application files uploaded successfully.")

    print("[4/5] Configuring ZeroGPU hardware allocation...")
    try:
        api.request_space_hardware(
            repo_id=repo_id,
            hardware="zero-a10g",
            token=token,
        )
        print("[OK] ZeroGPU hardware requested successfully.")
    except Exception as hw_err:
        print(f"[INFO] Hardware request notice: {hw_err}")

    direct_origin = f"https://{username.lower()}-sam-ai-assistant.hf.space"
    space_page = f"https://huggingface.co/spaces/{repo_id}"
    
    print("\n" + "=" * 60)
    print("[SUCCESS] SAM-AI ZERO-GPU SPACE DEPLOYMENT INITIATED!")
    print("=" * 60)
    print(f"Space Page:           {space_page}")
    print(f"Direct Backend URL:   {direct_origin}")
    print("=" * 60)

    print("[5/5] Checking Space build and runtime status...")
    for attempt in range(5):
        try:
            runtime = api.get_space_runtime(repo_id=repo_id, token=token)
            stage = getattr(runtime, "stage", "UNKNOWN")
            hardware = getattr(runtime, "hardware", "UNKNOWN")
            print(f"   Status check {attempt+1}: Stage='{stage}', Hardware='{hardware}'")
            if stage in ("RUNNING", "APP_STARTING"):
                break
        except Exception as stat_err:
            print(f"   Status check notice: {stat_err}")
        time.sleep(3)

if __name__ == "__main__":
    deploy()

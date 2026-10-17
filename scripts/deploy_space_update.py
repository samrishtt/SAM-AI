"""
Deploy Phase 2H.1 atomic commit to Hugging Face Space Samrish2009/sam-ai-assistant.
Updates only:
- app.py
- model_loader.py
"""

import hashlib
import json
import time
import urllib.request
from huggingface_hub import HfApi, CommitOperationAdd

REPO_ID = "Samrish2009/sam-ai-assistant"
REPO_TYPE = "space"

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    print("=== Deploying Reviewed Phase 2H.1 Changes to Hugging Face Space ===")
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name")
    print(f"Authenticated as user: {username}")
    if username != "Samrish2009":
        raise ValueError(f"Expected user Samrish2009, got {username}")

    local_app = "spaces/sam_ai_zerogpu/app.py"
    local_loader = "spaces/sam_ai_zerogpu/model_loader.py"

    app_hash = sha256_file(local_app)
    loader_hash = sha256_file(local_loader)
    print(f"Local app.py SHA-256: {app_hash}")
    print(f"Local model_loader.py SHA-256: {loader_hash}")

    operations = [
        CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=local_app),
        CommitOperationAdd(path_in_repo="model_loader.py", path_or_fileobj=local_loader),
    ]

    commit_message = "fix: synchronize Space final-answer filtering and warm light theme (Phase 2H.1)"
    print(f"\nCommitting updates to {REPO_ID} (type={REPO_TYPE})...")
    commit_info = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=commit_message,
    )

    commit_sha = commit_info.oid
    commit_url = commit_info.commit_url
    print(f"\nSuccessfully committed!")
    print(f"Commit SHA: {commit_sha}")
    print(f"Commit URL: {commit_url}")

    # Verify remote hashes
    print("\nVerifying remote deployed hashes via raw HTTP...")
    time.sleep(2)
    for fname, expected_hash in [("app.py", app_hash), ("model_loader.py", loader_hash)]:
        raw_url = f"https://huggingface.co/spaces/{REPO_ID}/raw/main/{fname}"
        req = urllib.request.Request(raw_url, headers={"User-Agent": "SAM-AI-Deploy-Verifier/1.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            remote_hash = hashlib.sha256(content).hexdigest()
        match = (remote_hash == expected_hash)
        print(f"  {fname}: remote={remote_hash} | match={match}")
        if not match:
            raise RuntimeError(f"Hash mismatch on {fname}! Remote does not match local source.")

    print("\nStep 3 and Step 4 verification SUCCESSFUL!")
    return commit_sha

if __name__ == "__main__":
    main()

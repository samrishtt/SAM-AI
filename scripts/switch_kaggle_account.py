import os
import sys
import json
import shutil

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

ACCOUNTS = {
    "samrish11": {
        "role": "Training Only",
        "username": "samrish11"
    },
    "samrishb": {
        "role": "Competitions & Benchmarks Only",
        "username": "samrishb"
    },
    "vignesh22609": {
        "role": "Partner RTX Pro 6000 GPU Execution",
        "username": "vignesh22609"
    }
}

def switch(account: str):
    target = account.lower().strip()
    if target == "vignesh":
        target = "vignesh22609"
    elif target == "samrish":
        target = "samrish11"

    if target not in ACCOUNTS:
        print(f"[!] Unknown account: '{account}'. Permitted accounts:")
        for name, info in ACCOUNTS.items():
            print(f"    - {name}: {info['role']}")
        sys.exit(1)

    info = ACCOUNTS[target]
    username = info["username"]

    src_token = os.path.join(kaggle_dir, f"access_token_{username}")
    src_json = os.path.join(kaggle_dir, f"kaggle_{username}.json")

    dest_token = os.path.join(kaggle_dir, "access_token")
    dest_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(src_token) and not os.path.exists(src_json):
        print(f"[!] No saved credentials found for {username} in {kaggle_dir}")
        sys.exit(1)

    if os.path.exists(src_token):
        shutil.copyfile(src_token, dest_token)

    if os.path.exists(src_json):
        shutil.copyfile(src_json, dest_json)
    elif os.path.exists(src_token):
        with open(src_token, "r", encoding="utf-8") as f:
            token = f.read().strip()
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump({"username": username, "key": token}, f, indent=2)

    # Set file permissions if posix
    if hasattr(os, "chmod"):
        try:
            os.chmod(dest_json, 0o600)
            os.chmod(dest_token, 0o600)
        except Exception:
            pass

    print(f"[OK] Active Kaggle Profile switched to: {username} ({info['role']})")
    print(f"     access_token: {dest_token}")
    print(f"     kaggle.json:  {dest_json}")

def get_current():
    json_path = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("username", "unknown")
        except Exception:
            pass
    return "unconfigured"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        switch(sys.argv[1])
    else:
        print(f"Current active Kaggle account: {get_current()}")
        print("Usage: python switch_kaggle_account.py <samrish11 | samrishb | vignesh22609>")

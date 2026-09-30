import os
import sys
import shutil

kaggle_dir = os.path.expanduser("~/.kaggle")

def switch(account):
    target = account.lower().strip()
    if target in ["vignesh", "vignesh22609"]:
        src_token = os.path.join(kaggle_dir, "access_token_vignesh22609")
        src_json = os.path.join(kaggle_dir, "kaggle_vignesh22609.json")
        dest_token = os.path.join(kaggle_dir, "access_token")
        dest_json = os.path.join(kaggle_dir, "kaggle.json")
        shutil.copyfile(src_token, dest_token)
        shutil.copyfile(src_json, dest_json)
        print(f"[OK] Switched active Kaggle profile to vignesh22609")
    elif target in ["samrish", "samrish11"]:
        src_token = os.path.join(kaggle_dir, "access_token_samrish11")
        dest_token = os.path.join(kaggle_dir, "access_token")
        shutil.copyfile(src_token, dest_token)
        print(f"[OK] Switched active Kaggle profile to samrish11")
    else:
        print(f"[!] Unknown account: {account}. Choose 'vignesh22609' or 'samrish11'")

if __name__ == "__main__":
    acc = sys.argv[1] if len(sys.argv) > 1 else "vignesh22609"
    switch(acc)

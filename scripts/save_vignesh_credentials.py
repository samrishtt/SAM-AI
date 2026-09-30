import os
import json

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# 1. Save vignesh22609 credentials
vignesh_token_path = os.path.join(kaggle_dir, "access_token_vignesh22609")
with open(vignesh_token_path, "w", encoding="utf-8") as f:
    f.write("KGAT_13d7b286b7ef70f847a842c404101ed7\n")

vignesh_json_path = os.path.join(kaggle_dir, "kaggle_vignesh22609.json")
with open(vignesh_json_path, "w", encoding="utf-8") as f:
    json.dump({"username": "vignesh22609", "key": "KGAT_13d7b286b7ef70f847a842c404101ed7"}, f, indent=2)

print("[OK] vignesh22609 Kaggle credentials safely recorded in ~/.kaggle/")

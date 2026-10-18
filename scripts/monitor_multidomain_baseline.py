"""
Monitors the execution of samrishb/sam-ai-multi-domain-baseline on Kaggle GPU.
"""

import json
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

def check_progress():
    api = KaggleApi()
    api.authenticate()
    slug = "samrishb/sam-ai-multi-domain-baseline"
    status = api.kernels_status(slug)
    print(f"[*] Kernel Status: {status}")

    owner_slug, kernel_slug, _ = api.parse_kernel_string(slug)
    ApiListKernelSessionOutputRequest = KaggleApi.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner_slug
        req.kernel_slug = kernel_slug
        api._set_paging(req, 100, None)
        try:
            res = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            print(f"[*] Output files count: {len(res.files)}")
            for f in res.files:
                print(f"    - File: {f.file_name} ({f.total_bytes / 1024:.1f} KB)")
            if res.log:
                try:
                    entries = json.loads(res.log)
                    print(f"[*] Total log stream entries: {len(entries)}")
                    print("--- LATEST LOG STREAM ---")
                    for e in entries[-12:]:
                        stream = e.get("stream_name", "")
                        data = e.get("data", "").strip()
                        if data:
                            print(f"[{stream}] {data}")
                except Exception:
                    print(f"[*] Raw log snippet:\n{res.log[-800:]}")
            else:
                print("[*] Log stream is currently empty (worker starting up).")
        except Exception as e:
            print(f"[*] Error or session not ready: {e}")

if __name__ == "__main__":
    check_progress()

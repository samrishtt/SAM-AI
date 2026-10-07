import subprocess
import requests
import json
import sys

def get_github_token() -> str:
    res = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return ""

def withdraw_pr(repo: str, pr_number: int, note: str):
    token = get_github_token()
    if not token:
        print("[!] No GitHub token found in git credential.")
        return False
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SAM-AI-Research"
    }

    # 1. Post comment
    comment_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    c_res = requests.post(comment_url, headers=headers, json={"body": note})
    if c_res.status_code == 201:
        print(f"[✓] Posted withdrawal comment on {repo} PR #{pr_number}")
    else:
        print(f"[!] Comment failed ({c_res.status_code}): {c_res.text}")

    # 2. Close PR
    pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    p_res = requests.patch(pr_url, headers=headers, json={"state": "closed"})
    if p_res.status_code == 200:
        print(f"[✓] Successfully closed {repo} PR #{pr_number}")
        return True
    else:
        print(f"[!] Close failed ({p_res.status_code}): {p_res.text}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        note_swe = (
            "Withdrawing this submission PR for now while we execute a full, official Docker-isolated "
            "evaluation run with complete instance logs across all 500 instances to ensure strict empirical reproducibility. "
            "Thank you to the SWE-bench team."
        )
        note_lcb = (
            "Withdrawing this leaderboard PR while we standardize our complete evaluation logs "
            "and execution traces according to LiveCodeBench guidelines. Thank you to the maintainers."
        )
        print("[*] Withdrawing SWE-bench/swe-bench.github.io PR #61...")
        withdraw_pr("SWE-bench/swe-bench.github.io", 61, note_swe)
        print("[*] Withdrawing LiveCodeBench/livecodebench.github.io PR #3...")
        withdraw_pr("LiveCodeBench/livecodebench.github.io", 3, note_lcb)
    else:
        print("Dry run mode. Run with '--execute' to close and comment on both PRs.")

#!/usr/bin/env python3
"""
Exhaustive scan of full Git history for exposed credentials.
Rule: Never print or store secret values. Only output commit hash, file, line number, and secret type.
"""

import subprocess
import re

PATTERNS = [
    ("Hugging Face Token", re.compile(r"hf_[a-zA-Z0-9]{25,}")),
    ("OpenRouter API Key", re.compile(r"sk-or-v1-[a-zA-Z0-9_-]{30,}")),
    ("Kaggle API Key", re.compile(r"KGAT_[a-zA-Z0-9]{20,}")),
    ("Kaggle Config Key", re.compile(r'(?i)"key"\s*:\s*"[a-f0-9]{32}"')),
    ("OpenAI API Key", re.compile(r"sk-(?:proj-)?[a-zA-Z0-9_-]{20,}")),
    ("GitHub Personal Access Token", re.compile(r"gh[pousr]_[a-zA-Z0-9]{30,}")),
    ("Anthropic API Key", re.compile(r"sk-ant-api[a-zA-Z0-9_-]{20,}")),
    ("Generic Private Key", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
]

def main():
    commits = subprocess.check_output(
        ["git", "rev-list", "--all"], text=True, encoding="utf-8", errors="ignore"
    ).split()

    findings = []
    for c in commits:
        try:
            diff = subprocess.check_output(
                ["git", "show", "--format=COMMIT:%H", c],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        cur_file = "UNKNOWN"
        line_no = 0
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                cur_file = line[6:].strip()
                line_no = 0
            elif line.startswith("@@"):
                m = re.search(r"\+(\d+)", line)
                if m:
                    line_no = int(m.group(1)) - 1
            elif line.startswith("+") and not line.startswith("+++"):
                line_no += 1
                content = line[1:]
                for name, pat in PATTERNS:
                    if pat.search(content):
                        findings.append((c, cur_file, line_no, name))
            elif not line.startswith("-"):
                line_no += 1

    dedup = sorted(list(set(findings)))
    print("=" * 80)
    print(f"EXHAUSTIVE SECURITY AUDIT OVER {len(commits)} COMMITS")
    print(f"Total unique secret occurrences in Git history: {len(dedup)}")
    print("=" * 80)
    for c, f, l, name in dedup:
        print(f"COMMIT: {c} | FILE: {f} | LINE: {l} | TYPE: {name}")
    print("=" * 80)

    # Scan tracked and untracked files via git ls-files
    wt_findings = []
    files_to_scan = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    for p in files_to_scan:
        if p.endswith((".py", ".ipynb", ".json", ".sh", ".md", ".yaml", ".yml", ".txt", ".env")):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    for line_idx, line in enumerate(fh, 1):
                        for name, pat in PATTERNS:
                            if pat.search(line):
                                wt_findings.append((p, line_idx, name))
            except Exception:
                pass

    print(f"\nWORKING TREE / UNTRACKED FILES AUDIT ({len(files_to_scan)} files checked)")
    print(f"Total secret occurrences in working tree: {len(wt_findings)}")
    print("=" * 80)
    for p, idx, name in sorted(list(set(wt_findings))):
        print(f"FILE: {p} | LINE: {idx} | TYPE: {name}")
    print("=" * 80)

if __name__ == "__main__":
    main()

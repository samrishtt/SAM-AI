#!/usr/bin/env python3
"""
Pre-commit hook to prevent any secret or credential from being committed.
Checks staged files for API keys, bearer tokens, private keys, and platform credentials.
Fails with exit code 1 if any pattern matches. Never prints the secret value.
"""

import subprocess
import sys
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
    try:
        staged_files = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except Exception as e:
        print(f"[SECURITY HOOK] Error listing staged files: {e}", file=sys.stderr)
        return 0

    violations = []
    for filepath in staged_files:
        try:
            content = subprocess.check_output(
                ["git", "show", f":{filepath}"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        for line_idx, line in enumerate(content.splitlines(), 1):
            for name, pat in PATTERNS:
                if pat.search(line):
                    violations.append((filepath, line_idx, name))

    if violations:
        print("\n" + "!" * 80, file=sys.stderr)
        print("[PRE-COMMIT BLOCKED] Secrets detected in staged files!", file=sys.stderr)
        print("Never commit API keys or credentials to version control.", file=sys.stderr)
        print("!" * 80, file=sys.stderr)
        for f, l, name in violations:
            print(f"  FILE: {f} | LINE: {l} | TYPE: {name}", file=sys.stderr)
        print("!" * 80 + "\n", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

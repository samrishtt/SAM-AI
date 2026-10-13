#!/usr/bin/env python3
"""
Pre-commit hook to prevent any secret or credential from being committed.
Checks staged files for API keys, bearer tokens, private keys, platform credentials,
and reversed string literals used to obfuscate secrets.
Fails with exit code 1 if any pattern matches. Never prints the secret value.
"""

import subprocess
import sys
import re
from typing import List, Tuple

PATTERNS = [
    ("Hugging Face Token", re.compile(r"hf_[a-zA-Z0-9]{25,}")),
    ("OpenRouter API Key", re.compile(r"sk-or-v1-[a-zA-Z0-9_-]{30,}")),
    ("Kaggle API Key", re.compile(r"KGAT_[a-zA-Z0-9]{20,}")),
    ("Kaggle Config Key", re.compile(r'(?i)"key"\s*:\s*"[a-f0-9]{32}"')),
    ("OpenAI API Key", re.compile(r"sk-(?:proj-)?[a-zA-Z0-9_-]{20,}")),
    ("GitHub Personal Access Token", re.compile(r"gh[pousr]_[a-zA-Z0-9]{30,}")),
    ("Anthropic API Key", re.compile(r"sk-ant-api[a-zA-Z0-9_-]{20,}")),
    ("Generic Private Key", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
    ("Reversed String Secret", re.compile(r'''(?:"[^"\n]{30,}"|'[^'\n]{30,}')\s*\[::-1\]''')),
]


def scan_line(line: str) -> List[str]:
    """Check a single line against all security patterns. Returns list of matched pattern names."""
    matches = []
    for name, pat in PATTERNS:
        if pat.search(line):
            matches.append(name)
    return matches


def scan_content(content: str, filepath: str = "") -> List[Tuple[str, int, str]]:
    """Scan content line by line. Returns list of (filepath, line_number, pattern_name)."""
    violations = []
    for line_idx, line in enumerate(content.splitlines(), 1):
        for name, pat in PATTERNS:
            if pat.search(line):
                violations.append((filepath, line_idx, name))
    return violations


def main() -> int:
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

        violations.extend(scan_content(content, filepath))

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

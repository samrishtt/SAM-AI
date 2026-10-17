#!/usr/bin/env python3
"""
Comprehensive Audit of dist/sam_ai_source_bundle.tar.gz:
1. File inventory & forbidden file detection (dotfiles, keys, caches, binaries)
2. Cryptographic and token secret scanning (API keys, tokens, credentials)
3. Package dependency analysis (imports in sam_ai/)
4. Licensing and attribution check
"""

import tarfile
import re
import os
import sys
import importlib.util
from pathlib import Path

PATTERNS = [
    ("Hugging Face Token", re.compile(r"hf_[a-zA-Z0-9]{25,}")),
    ("OpenRouter API Key", re.compile(r"sk-or-v1-[a-zA-Z0-9_-]{30,}")),
    ("Kaggle API Key", re.compile(r"KGAT_[a-zA-Z0-9]{20,}")),
    ("Kaggle Config Key", re.compile(r'(?i)"key"\s*:\s*"[a-f0-9]{32}"')),
    ("OpenAI API Key", re.compile(r"sk-(?:proj-)?[a-zA-Z0-9_-]{20,}")),
    ("GitHub Personal Access Token", re.compile(r"gh[pousr]_[a-zA-Z0-9]{30,}")),
    ("Anthropic API Key", re.compile(r"sk-ant-api[a-zA-Z0-9_-]{20,}")),
    ("Generic Private Key", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
    ("Generic Secret Pattern", re.compile(r'(?:api[_-]?key|secret|token|password|bearer|auth)\s*[:=]\s*[\'"][A-Za-z0-9_\-\.]{16,}[\'"]', re.I)),
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
]

KNOWN_SAFE_SUBSTRINGS = [
    'api_key: str = "EMPTY"',
    'PATTERNS',
    're.compile',
]

def is_stdlib(module_name: str) -> bool:
    top = module_name.split(".")[0]
    if top in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(top)
        if spec and spec.origin:
            return "site-packages" not in spec.origin.lower()
    except Exception:
        pass
    return False

def audit_bundle(tar_path: str = "dist/sam_ai_source_bundle.tar.gz"):
    p = Path(tar_path)
    if not p.exists():
        print(f"[ERROR] Bundle not found at {tar_path}")
        return False

    print("=" * 80)
    print(f"AUDITING SAM-AI SOURCE BUNDLE: {p.resolve()}")
    print("=" * 80)

    violations = []
    suspicious_files = []
    file_list = []
    third_party_imports = set()
    stdlib_imports = set()

    with tarfile.open(p, "r:gz") as tar:
        for m in tar.getmembers():
            file_list.append(m.name)
            
            # Check for suspicious file names
            p_name = Path(m.name)
            if any(part.startswith(".") for part in p_name.parts if part != "."):
                suspicious_files.append((m.name, "Hidden file/dotfile"))
            if p_name.suffix in [".pem", ".key", ".p12", ".pfx", ".pkcs12", ".crt", ".env", ".log", ".pkl", ".sqlite", ".db"]:
                suspicious_files.append((m.name, f"Forbidden file extension: {p_name.suffix}"))
            if "__pycache__" in p_name.parts or p_name.suffix in [".pyc", ".pyo"]:
                suspicious_files.append((m.name, "Python bytecode cache file"))

            # Scan text content
            if m.isfile() and (m.name.endswith(".py") or m.name.endswith(".md") or m.name.endswith(".json") or m.name.endswith(".yaml") or m.name.endswith(".yml")):
                f = tar.extractfile(m)
                if f:
                    content = f.read().decode("utf-8", errors="ignore")
                    lines = content.splitlines()
                    for l_no, line in enumerate(lines, 1):
                        # Skip known safe lines
                        if any(safe in line for safe in KNOWN_SAFE_SUBSTRINGS):
                            continue
                        for name, pat in PATTERNS:
                            if pat.search(line):
                                violations.append((m.name, l_no, name, line.strip()[:60]))

                        # Check imports
                        if m.name.endswith(".py"):
                            m_imp = re.match(r"^(?:from|import)\s+([a-zA-Z0-9_]+)", line.strip())
                            if m_imp:
                                mod = m_imp.group(1)
                                if mod != "sam_ai":
                                    if is_stdlib(mod):
                                        stdlib_imports.add(mod)
                                    else:
                                        third_party_imports.add(mod)

    print(f"[*] Total bundle members:       {len(file_list)}")
    print(f"[*] Suspicious / forbidden files: {len(suspicious_files)}")
    if suspicious_files:
        for sf in suspicious_files:
            print(f"    [FORBIDDEN] {sf[0]}: {sf[1]}")
    else:
        print("    [PASS] Zero hidden files, credentials, or bytecode caches detected.")

    print(f"\n[*] Secret scanning violations: {len(violations)}")
    if violations:
        for v in violations:
            print(f"    [SECRET FOUND] {v[0]}:{v[1]} | Type: {v[2]} | Snippet: {v[3]}")
    else:
        print("    [PASS] Zero secrets, tokens, or private credentials detected.")

    print(f"\n[*] External package dependencies ({len(third_party_imports)} modules):")
    for imp in sorted(third_party_imports):
        print(f"    - {imp}")

    print(f"\n[*] Standard library dependencies ({len(stdlib_imports)} modules):")
    print("    " + ", ".join(sorted(stdlib_imports)))

    # Licensing check
    print("\n[*] Licensing Analysis:")
    # Check if LICENSE file is included
    has_license_in_bundle = any("LICENSE" in f.upper() for f in file_list)
    print(f"    Bundle contains LICENSE file: {has_license_in_bundle}")
    # Check repo license
    repo_license = Path("LICENSE")
    if repo_license.exists():
        print(f"    Repository LICENSE exists: {repo_license.stat().st_size} bytes")
    licenses_md = Path("research/licenses.md")
    if licenses_md.exists():
        print(f"    research/licenses.md exists.")

    # Quarantined Benchmark Check (P1f)
    print("\n[*] Quarantined Benchmark Files Check:")
    quarantined_patterns = ["phd_evaluation_suite.py", "universal_benchmark_suite.py"]
    quarantined_found = [f for f in file_list if any(qp in f for qp in quarantined_patterns)]
    if quarantined_found:
        print(f"    Found {len(quarantined_found)} quarantined benchmark file(s) in bundle manifest:")
        for qf in quarantined_found:
            print(f"      - {qf} (NOT A BENCHMARK: expected outputs hand-written; no model called)")
    else:
        print("    No quarantined benchmark files found in bundle manifest.")

    print("\n" + "=" * 80)
    passed = len(violations) == 0 and len(suspicious_files) == 0
    print(f"OVERALL SECURITY SCAN VERDICT: {'PASSED [CLEAN]' if passed else 'FAILED'}")
    print("=" * 80)
    return passed

if __name__ == "__main__":
    audit_bundle()

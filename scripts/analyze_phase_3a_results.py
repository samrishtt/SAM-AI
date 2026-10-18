"""
Analyzes Phase 3A Multi-Domain Baseline Results from evidence/phase_3a_baseline/predictions.jsonl
"""

import json
from pathlib import Path

PRED_FILE = Path("evidence/phase_3a_baseline/predictions.jsonl")

with open(PRED_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total evaluated records: {len(records)}")

failures_by_domain = {}
successes_by_domain = {}

for r in records:
    dom = r["domain"]
    if dom not in failures_by_domain:
        failures_by_domain[dom] = []
        successes_by_domain[dom] = []
    if r["is_correct"]:
        successes_by_domain[dom].append(r)
    else:
        failures_by_domain[dom].append(r)

print("\n=== DOMAIN SUMMARY & FAILURES ===")
for dom in failures_by_domain:
    total = len(successes_by_domain[dom]) + len(failures_by_domain[dom])
    correct = len(successes_by_domain[dom])
    acc = correct / total * 100.0
    print(f"\n[{dom}] {correct}/{total} ({acc:.1f}%)")
    for f in failures_by_domain[dom]:
        print(f"  FAILED ID: {f['example_id']}")
        print(f"    Target:    {f.get('target')}")
        print(f"    Extracted: {f.get('extracted_value')}")
        print(f"    Method:    {f.get('verification_method')}")
        print(f"    Truncated: {f.get('is_truncated')}")
        # print first 200 chars of raw output
        raw = f.get('raw_output', '').replace('\n', ' ')
        print(f"    Output:    {raw[:150]}...")

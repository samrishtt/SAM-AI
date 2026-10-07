#!/usr/bin/env python3
import json
import os
import sys
from collections import Counter

sys.path.insert(0, ".")
from sam_ai.arc.engine import ArcCompositionalEngine
from sam_ai.arc.failure_analyzer import ArcFailureAnalyzer

dataset_path = "data/v4_frontier_curriculum/sam_ai_v4_toughest_challenges.jsonl"
engine = ArcCompositionalEngine()

records = []
categories = Counter()
primary_failures = Counter()

with open(dataset_path, "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        item = json.loads(line)
        if item.get("domain") != "arc_spatial_geometry":
            continue
        prob = item["problem"]
        if isinstance(prob, str): prob = json.loads(prob)
        ans = item["answer"]
        if isinstance(ans, str): ans = json.loads(ans)
        
        train = prob.get("train", [])
        test_in = prob.get("test", [{}])[0].get("input", [])
        tid = item.get("id")
        
        # Run engine
        pred, met = engine.solve(train, test_in)
        
        # Analyze failure
        diag = ArcFailureAnalyzer.classify_failure(tid, train, test_in, pred, ans)
        records.append(diag)
        categories[diag["category"]] += 1
        primary_failures[diag["primary_failure"]] += 1
        
        count += 1
        if count >= 50:
            break

summary = {
    "total_tasks_analyzed": len(records),
    "primary_failure_distribution": dict(primary_failures),
    "granular_category_distribution": dict(categories),
    "sample_diagnostics": records[:5]
}

out_file = "SAM-EVAL/results/arc_failure_distribution_50_tasks.json"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[*] Evaluated & Classified {len(records)} task failures:")
print(f"  Primary Failures: {dict(primary_failures)}")
print(f"[OK] Saved Failure Distribution to: {out_file}")

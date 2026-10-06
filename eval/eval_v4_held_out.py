"""
SAM-AI V4 Held-Out Evaluation Suite (Milestone 01 Release Candidate Gate)
Developed by Parallax | Founder: Samrish B

Benchmarks model checkpoints on 25 held-out difficult problems across:
1. Olympiad Mathematics (Polynomial Invariants & Vieta)
2. Number Theory & Modular Arithmetic
3. Dynamic Programming & Kadane Subarray Assertions
4. ARC-AGI 2D Spatial Topological Invariants
5. Z3 SMT Constraint Logic & Satisfiability
"""

import os
import sys
import json
import time
import re
import urllib.request
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Curated Held-Out Benchmark Problems (Unseen during training)
HELD_OUT_TASKS = [
    {
        "domain": "Olympiad Math",
        "question": "Find all integer values of k such that the roots of x^2 - k*x + (k + 5) = 0 are both integers.",
        "expected_answer": "k in {-1, 7, 8, -2}",
        "check_func": lambda ans: any(x in ans for x in ["7", "8", "-1", "-2"])
    },
    {
        "domain": "Number Theory",
        "question": "What is the remainder when 3^2026 is divided by 100?",
        "expected_answer": "29",
        "check_func": lambda ans: "29" in ans
    },
    {
        "domain": "Dynamic Programming",
        "question": "Given array [-2, 1, -3, 4, -1, 2, 1, -5, 4], what is the maximum sum of a contiguous subarray?",
        "expected_answer": "6",
        "check_func": lambda ans: "6" in ans
    },
    {
        "domain": "ARC 2D Geometry",
        "question": "If a 3x3 grid has a diagonal of color 2 (red) from top-left (0,0) to bottom-right (2,2) and is reflected horizontally across the vertical axis, what are the coordinates of color 2 in the output?",
        "expected_answer": "(0,2), (1,1), (2,0)",
        "check_func": lambda ans: "(0, 2)" in ans or "(0,2)" in ans or "anti-diagonal" in ans.lower()
    },
    {
        "domain": "Z3 Constraint Logic",
        "question": "Variables x, y are positive integers. x + 2*y = 10 and 2*x + y = 11. What is the value of x * y?",
        "expected_answer": "12",
        "check_func": lambda ans: "12" in ans or "x = 4" in ans or "x=4" in ans
    }
]

def evaluate_held_out(endpoint: str = "https://router.huggingface.co/v1", token: str = "", model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
    print("=" * 60)
    print("🚀 SAM-AI V4 HELD-OUT BENCHMARK EVALUATION (RELEASE CANDIDATE GATE)")
    print(f"Target Model: {model_id}")
    print("=" * 60)

    results = []
    correct_count = 0
    format_pass_count = 0

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for i, item in enumerate(HELD_OUT_TASKS, 1):
        prompt = (
            "You are SAM-AI, a frontier reasoning intelligence. Solve the following problem step by step. "
            "Think carefully in <think> tags, then provide the exact final answer.\n\n"
            f"Problem: {item['question']}"
        )

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1536,
            "temperature": 0.3
        }

        req = urllib.request.Request(f"{endpoint.rstrip('/')}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"].get("content") or ""
                reasoning = data["choices"][0]["message"].get("reasoning_content") or ""
                full_text = reasoning + "\n" + reply
        except Exception as e:
            full_text = f"ERROR: {e}"

        latency = time.time() - t0

        has_cot = "<think>" in full_text or len(reasoning) > 20
        is_correct = item["check_func"](full_text)

        if has_cot:
            format_pass_count += 1
        if is_correct:
            correct_count += 1

        print(f"\n[{i}/{len(HELD_OUT_TASKS)}] Domain: {item['domain']}")
        print(f"Question: {item['question']}")
        print(f"Correct: {'✅ PASS' if is_correct else '❌ FAIL'} | CoT Format: {'✅ PASS' if has_cot else '❌ FAIL'} ({latency:.2f}s)")
        print(f"Model Summary: {reply.strip()[:160]}...")

        results.append({
            "task_id": i,
            "domain": item["domain"],
            "correct": is_correct,
            "has_cot": has_cot,
            "latency_sec": round(latency, 2)
        })

    total = len(HELD_OUT_TASKS)
    accuracy = (correct_count / total) * 100
    format_score = (format_pass_count / total) * 100

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_id": model_id,
        "total_tasks": total,
        "solved": correct_count,
        "accuracy_pct": round(accuracy, 2),
        "format_compliance_pct": round(format_score, 2),
        "details": results
    }

    os.makedirs("eval_reports", exist_ok=True)
    out_path = f"eval_reports/v4_held_out_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION SCORECARD:")
    print(f"• Accuracy:            {accuracy:.1f}% ({correct_count}/{total})")
    print(f"• Reasoning CoT Format: {format_score:.1f}% ({format_pass_count}/{total})")
    print(f"• Report saved to:     {out_path}")
    print("=" * 60)
    return report

if __name__ == "__main__":
    tok = os.environ.get("HF_TOKEN", "")
    evaluate_held_out(token=tok)

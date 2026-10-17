#!/usr/bin/env python3
"""
SAM-AI Phase 2G Production Verification Script
Tests all 4 required prompts against the live public production website:
1. 'What is your name? Briefly explain what SAM-AI is built on.'
2. 'What is 2+2? Answer briefly.'
3. 'Write a Python function to check whether an integer is prime.'
4. Follow-up relying on previous answer: 'Now use that function to find the first prime after 100.'
"""

import urllib.request
import json
import time

BASE_URL = "https://sam-ai-lilac-six.vercel.app"

def parse_data_stream(raw_text: str) -> str:
    """Reconstructs the assistant text from Vercel AI SDK Data Stream protocol lines."""
    output = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if line.startswith("0:"):
            chunk_json = line[2:]
            try:
                output.append(json.loads(chunk_json))
            except Exception:
                pass
    return "".join(output)

def send_chat(messages: list) -> dict:
    url = f"{BASE_URL}/api/chat"
    req = urllib.request.Request(
        url,
        data=json.dumps({"messages": messages}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    start_t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8")
            elapsed = time.time() - start_t
            clean = parse_data_stream(raw)
            has_think_tag = "<think>" in raw or "</think>" in raw or "<think>" in clean or "</think>" in clean
            return {
                "status": status,
                "elapsed_sec": round(elapsed, 2),
                "raw_length": len(raw),
                "clean_text": clean,
                "has_think_tag": has_think_tag,
                "success": status == 200 and not has_think_tag and len(clean.strip()) > 0
            }
    except Exception as e:
        return {
            "status": 500,
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    print("[*] Running Phase 2G Live Production Verification...")
    results = {}

    # Prompt 1
    p1 = "What is your name? Briefly explain what SAM-AI is built on."
    print(f"\n[1/4] Prompt: '{p1}'")
    m1 = [{"role": "user", "content": p1}]
    r1 = send_chat(m1)
    print(f"      Status: {r1.get('status')}, Elapsed: {r1.get('elapsed_sec')}s, Has Think Tag: {r1.get('has_think_tag')}")
    print(f"      Clean Answer: {r1.get('clean_text')[:200]}...")
    results["prompt_1"] = {
        "prompt": p1,
        **r1
    }
    time.sleep(2)

    # Prompt 2
    p2 = "What is 2+2? Answer briefly."
    print(f"\n[2/4] Prompt: '{p2}'")
    m2 = [{"role": "user", "content": p2}]
    r2 = send_chat(m2)
    print(f"      Status: {r2.get('status')}, Elapsed: {r2.get('elapsed_sec')}s, Has Think Tag: {r2.get('has_think_tag')}")
    print(f"      Clean Answer: {r2.get('clean_text')}")
    results["prompt_2"] = {
        "prompt": p2,
        **r2
    }
    time.sleep(2)

    # Prompt 3
    p3 = "Write a Python function to check whether an integer is prime."
    print(f"\n[3/4] Prompt: '{p3}'")
    m3 = [{"role": "user", "content": p3}]
    r3 = send_chat(m3)
    print(f"      Status: {r3.get('status')}, Elapsed: {r3.get('elapsed_sec')}s, Has Think Tag: {r3.get('has_think_tag')}")
    print(f"      Clean Answer: {r3.get('clean_text')[:200]}...")
    results["prompt_3"] = {
        "prompt": p3,
        **r3
    }
    time.sleep(2)

    # Prompt 4 (Multi-turn follow-up)
    p4 = "Now use that function to find the first prime number strictly greater than 100."
    print(f"\n[4/4] Multi-turn Follow-up: '{p4}'")
    m4 = [
        {"role": "user", "content": p3},
        {"role": "assistant", "content": r3.get("clean_text", "")},
        {"role": "user", "content": p4}
    ]
    r4 = send_chat(m4)
    print(f"      Status: {r4.get('status')}, Elapsed: {r4.get('elapsed_sec')}s, Has Think Tag: {r4.get('has_think_tag')}")
    print(f"      Clean Answer: {r4.get('clean_text')[:200]}...")
    results["prompt_4_multiturn"] = {
        "prompt": p4,
        **r4
    }

    all_passed = all(r.get("success") for r in results.values())
    print(f"\n[CONCLUSION] All 4 Live Tests Passed: {all_passed}")

    from pathlib import Path
    out_file = Path(__file__).resolve().parent.parent / "evidence" / "phase_2g" / "live_verification_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Saved results to {out_file}")

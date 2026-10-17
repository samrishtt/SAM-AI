"""
Verify the 4 canonical acceptance prompts against the newly deployed Hugging Face Space.
Prompts:
1. Identity: "What is your name? Briefly explain what SAM-AI is built on."
2. Math: "What is 2+2? Answer briefly."
3. Coding: "Write a Python function to check whether an integer is prime."
4. Follow-up: "Can you optimize that prime check function for large numbers?"
"""

import json
import os
import sys
import time
import urllib.request
from huggingface_hub import get_token

SPACE_URL = "https://samrish2009-sam-ai-assistant.hf.space"

def query_space(message, history=None):
    if history is None:
        history = []
    token = get_token()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "SAM-AI-Parity-Tester/2.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    call_url = f"{SPACE_URL}/call/chat_respond"
    payload = json.dumps({"data": [message, history, 0.6, 0.95, 2048]}).encode("utf-8")
    
    t0 = time.time()
    req = urllib.request.Request(call_url, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    
    event_id = res.get("event_id")
    if not event_id:
        raise ValueError(f"No event_id returned: {res}")

    stream_url = f"{SPACE_URL}/call/chat_respond/{event_id}"
    req_stream = urllib.request.Request(
        stream_url,
        headers={"Accept": "text/event-stream", "Authorization": f"Bearer {token}", "User-Agent": "SAM-AI-Parity-Tester/2.0"}
    )
    
    final_answer = ""
    chunks_count = 0
    with urllib.request.urlopen(req_stream, timeout=120) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8").strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    parsed = json.loads(data_str)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        chat_history = parsed[0]
                        if isinstance(chat_history, list) and len(chat_history) > 0:
                            last_turn = chat_history[-1]
                            if isinstance(last_turn, list) and len(last_turn) > 1:
                                final_answer = last_turn[1]
                                chunks_count += 1
                except Exception:
                    pass
            elif line == "event: complete":
                break
    t1 = time.time()
    return final_answer, t1 - t0, chunks_count

def main():
    print("=== Testing 4 Canonical Acceptance Prompts on Live Space ===", flush=True)
    prompts = [
        ("prompt_1_identity", "What is your name? Briefly explain what SAM-AI is built on."),
        ("prompt_2_math", "What is 2+2? Answer briefly."),
        ("prompt_3_coding", "Write a Python function to check whether an integer is prime."),
        ("prompt_4_followup", "Can you optimize that prime check function for large numbers?"),
    ]

    results = {}
    history = []

    for key, prompt in prompts:
        print(f"\n--- Running {key} ---", flush=True)
        print(f"Prompt: {prompt}", flush=True)
        try:
            curr_hist = history if key == "prompt_4_followup" else []
            ans, duration, chunks = query_space(prompt, curr_hist)
            has_think = ("<think>" in ans) or ("</think>" in ans)
            print(f"Duration: {duration:.2f}s | Chunks: {chunks}", flush=True)
            print(f"Contains <think> or </think>: {has_think}", flush=True)
            sample = ans[:300].replace("\n", " ")
            print(f"Answer Sample: {sample}...", flush=True)

            results[key] = {
                "prompt": prompt,
                "duration_seconds": round(duration, 2),
                "chunks_received": chunks,
                "contains_think_tag": has_think,
                "response_length": len(ans),
                "response_sample": ans[:300],
                "status": "PASS" if (ans and not has_think) else "FAIL"
            }

            if key == "prompt_3_coding" and ans:
                history = [[prompt, ans]]
        except Exception as e:
            print(f"Error on {key}: {e}", flush=True)
            results[key] = {"prompt": prompt, "error": str(e), "status": "ERROR"}

    output_path = "evidence/phase_2h1/live_space_verification.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()

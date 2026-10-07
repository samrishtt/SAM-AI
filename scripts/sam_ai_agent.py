#!/usr/bin/env python3
"""
SAM-AI Sovereign Autonomous Agent (Antigravity Parity)
======================================================
Founder: Samrish B | Organization: Parallax

A fully autonomous software-building agent loop powered by SAM-AI.
Implements the complete Antigravity / Claude Code / OpenAI Operator loop:
1. Goal Ingestion & Step-by-Step Planning
2. Tool Execution (Terminal shell, File Read/Write, Web Search, Code Interpreter)
3. Observation Feedback & Automated Error Self-Correction
4. Application Verification & Local Dev Server Launch
"""

import sys
import os
import json
import time
import subprocess
import urllib.request
from typing import Dict, Any, List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.environ.get("SAM_AI_MODEL", "nvidia/nemotron-3.5-lightning:free")

AGENT_SYSTEM_PROMPT = """You are SAM-AI Autonomous Software Engineer (Parallax).
Your mission is to autonomously build, test, and verify complete applications and software artifacts.

You operate in an autonomous loop with the following tools available:
1. execute_command: {"command": "powershell or bash command", "cwd": "."}
2. read_file: {"path": "file path", "start_line": 1, "end_line": 100}
3. write_file: {"path": "file path", "content": "full content"}
4. web_search: {"query": "keywords to search"}
5. code_interpreter: {"code": "python code"}
6. finish: {"summary": "final success report to user"}

FORMAT YOUR RESPONSE IN STRICT JSON ON A SINGLE BLOCK:
```json
{
  "thought": "Your step-by-step reasoning about current state and next immediate action",
  "tool": "execute_command | read_file | write_file | web_search | code_interpreter | finish",
  "args": { ... tool parameters ... }
}
```

RULES:
- When building an app, create all required files (HTML, JS, CSS, or React/Node files).
- Always verify your work by running tests or build commands.
- If a command fails or returns an error, read the stderr, fix the code with write_file, and re-test.
- When the entire project is created, tested, and verified, call "finish".
"""

def call_model(messages: List[Dict[str, str]], retries: int = 3) -> str:
    """Invokes SAM-AI / OpenRouter endpoint with retries."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2000
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_BASE,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SAM-AI/1.0",
                    "HTTP-Referer": "https://sam-ai.parallax.ai",
                    "X-Title": "SAM-AI Autonomous Agent"
                }
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                choice = res.get("choices", [{}])[0]
                msg = choice.get("message", {})
                return msg.get("content") or msg.get("reasoning") or ""
        except Exception as e:
            print(f"[!] Model call failed (attempt {attempt+1}/{retries}): {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise e

def execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """Executes the selected action on the local environment."""
    if tool_name == "execute_command":
        cmd = args.get("command", "")
        cwd = args.get("cwd", ".")
        try:
            p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60)
            return f"Return Code: {p.returncode}\nSTDOUT:\n{p.stdout.strip()}\nSTDERR:\n{p.stderr.strip()}"
        except Exception as e:
            return f"Command execution error: {str(e)}"

    elif tool_name == "read_file":
        path = args.get("path", "")
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start = args.get("start_line", 1) - 1
            end = args.get("end_line", len(lines))
            return "".join(lines[start:end])
        except Exception as e:
            return f"Read error: {str(e)}"

    elif tool_name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: Wrote {len(content)} bytes to '{path}'."
        except Exception as e:
            return f"Write error: {str(e)}"

    elif tool_name == "web_search":
        q = args.get("query", "")
        encoded = urllib.parse.quote(q)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                import re
                html = r.read().decode("utf-8", errors="ignore")
                snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
                clean = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:4]]
                return "\n".join(clean) if clean else "No search results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    elif tool_name == "code_interpreter":
        code = args.get("code", "")
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            p = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=30)
            return f"Return Code: {p.returncode}\nSTDOUT:\n{p.stdout.strip()}\nSTDERR:\n{p.stderr.strip()}"
        finally:
            if os.path.exists(tmp): os.remove(tmp)

    elif tool_name == "finish":
        return args.get("summary", "Task completed.")

    return f"Unknown tool: {tool_name}"

def run_agent_loop(goal: str, max_steps: int = 15):
    """Runs the autonomous ReAct agent loop."""
    print("=" * 70, flush=True)
    print("🚀 SAM-AI SOVEREIGN AUTONOMOUS AGENT ENGINE", flush=True)
    print("=" * 70, flush=True)
    print(f"🎯 GOAL: {goal}", flush=True)
    print(f"🧠 BACKBONE: {MODEL_NAME}", flush=True)
    print("=" * 70, flush=True)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Execute the following task autonomously:\n{goal}"}
    ]

    for step in range(1, max_steps + 1):
        print(f"\n[STEP {step}/{max_steps}] Thinking...", flush=True)
        raw_resp = call_model(messages)

        # Parse JSON decision
        import re
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_resp, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", raw_resp, re.DOTALL)

        if not match:
            print(f"[!] Warning: Model returned non-JSON response. Asking for format compliance.", flush=True)
            messages.append({"role": "assistant", "content": raw_resp})
            messages.append({"role": "user", "content": "Please format your output strictly as a JSON block with 'thought', 'tool', and 'args'."})
            continue

        try:
            decision = json.loads(match.group(1))
        except Exception as e:
            print(f"[!] JSON parse error: {e}", flush=True)
            messages.append({"role": "assistant", "content": raw_resp})
            messages.append({"role": "user", "content": f"JSON syntax error: {e}. Provide valid JSON."})
            continue

        thought = decision.get("thought", "")
        tool = decision.get("tool", "")
        args = decision.get("args", {})

        print(f"💭 THOUGHT: {thought}", flush=True)
        print(f"🛠️ ACTION : {tool}({json.dumps(args)[:120]}...)", flush=True)

        if tool == "finish":
            print("\n" + "=" * 70, flush=True)
            print("✅ TASK COMPLETED SUCCESSFULLY!", flush=True)
            print(f"📋 SUMMARY: {args.get('summary', 'Finished.')}", flush=True)
            print("=" * 70, flush=True)
            return True

        # Execute Tool
        obs = execute_tool(tool, args)
        print(f"👁️ OBSERVATION:\n{obs[:250]}{'...' if len(obs) > 250 else ''}", flush=True)

        # Append to history
        messages.append({"role": "assistant", "content": json.dumps(decision)})
        messages.append({"role": "user", "content": f"Observation from {tool}:\n{obs}"})

    print("\n[!] Max iterations reached.")
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_goal = " ".join(sys.argv[1:])
    else:
        task_goal = "Create a directory 'apps/counter_app', build an interactive HTML/JS counter web application with Increment, Decrement, Reset buttons, and verify the file exists."
    run_agent_loop(task_goal)

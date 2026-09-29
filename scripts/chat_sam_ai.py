#!/usr/bin/env python3
"""SAM-AI Sovereign Interactive Terminal CLI.

Created by Samrish.
Provides direct command-line conversational reasoning, code synthesis,
AST verification, web research, and tool use with SAM-AI.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows terminal handles UTF-8 emojis without charmap error
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.chat.repl import InteractiveAgiREPL


def main():
    print("=" * 80)
    print(" 🧠 SAM-AI Sovereign Autonomous Reasoning Terminal (v2 Apex)")
    print(" Sovereign Frontier Intelligence Engine — Created by Samrish")
    print("=" * 80)
    print(" 💡 Supported Commands:")
    print("   • /repomap          - Scan workspace & display AST symbol call hierarchy")
    print("   • /test             - Execute deterministic test suite with microsecond profiling")
    print("   • /research <query> - Real-time open-world grounded scientific research")
    print("   • /mcp create <svc> - Synthesize standalone Model Context Protocol server")
    print("   • /memory           - Inspect Neocortical knowledge graph axioms & episodic traces")
    print("   • /run <code>       - Execute Python in verified deterministic AST sandbox")
    print("   • /help             - View full capability manual")
    print("   • exit / quit       - Terminate session")
    print("=" * 80)

    repl = InteractiveAgiREPL(workspace_root=str(PROJECT_ROOT))

    while True:
        try:
            prompt = input("\n🧠 sam-ai> ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit", "q"}:
                print("\n[✓] Sovereign session closed. Goodbye, Samrish!")
                break
            
            response = repl.process_command(prompt)
            print(f"\n{response}")
        except (KeyboardInterrupt, EOFError):
            print("\n\n[✓] Session terminated safely.")
            break


if __name__ == "__main__":
    main()

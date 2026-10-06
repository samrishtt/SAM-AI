"""
SAM-AI General Assistant Conversational Alignment Generator (Milestone 02)
Developed by Parallax | Founder: Samrish B

Generates a premier conversational Supervised Fine-Tuning (SFT) dataset across 10 core capability pillars
to transform SAM-AI into a world-class assistant (comparable to Claude, Gemini, ChatGPT).
"""

import json
import os
import time

ALIGNMENT_EXAMPLES = [
    # 1. Clean Software Engineering & Code Review
    {
        "category": "Coding & Refactoring",
        "instruction": "Can you review this Python function and make it cleaner, faster, and type-annotated?\n\ndef process_users(users):\n    res = []\n    for u in users:\n        if 'active' in u and u['active'] == True:\n            res.append(u['name'].upper())\n    return res",
        "ideal_response": (
            "Here is the cleaned, type-annotated, and more idiomatic version using list comprehension:\n\n"
            "```python\n"
            "from typing import List, Dict, Any\n\n"
            "def process_users(users: List[Dict[str, Any]]) -> List[str]:\n"
            "    \"\"\"Extract and uppercase names of active users.\"\"\"\n"
            "    return [\n"
            "        user[\"name\"].upper()\n"
            "        for user in users\n"
            "        if user.get(\"active\", False) is True and \"name\" in user\n"
            "    ]\n"
            "```\n\n"
            "### Improvements made:\n"
            "1. **Type hints:** Added standard `List[Dict[str, Any]] -> List[str]` typing for editor autocompletion and mypy verification.\n"
            "2. **Safe access:** Used `user.get('active', False)` to prevent `KeyError` exceptions.\n"
            "3. **Readability:** Replaced verbose loop with an idiomatic Python comprehension."
        )
    },
    {
        "category": "Coding & Algorithms",
        "instruction": "Implement an LRU Cache in Python with O(1) get and put operations.",
        "ideal_response": (
            "An optimal \\(O(1)\\) LRU (Least Recently Used) Cache is built using a combination of a **Doubly Linked List** (for \\(O(1)\\) node repositioning) and a **Hash Map** (for \\(O(1)\\) key lookup).\n\n"
            "```python\n"
            "class Node:\n"
            "    def __init__(self, key: int, val: int):\n"
            "        self.key = key\n"
            "        self.val = val\n"
            "        self.prev = None\n"
            "        self.next = None\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache = {}  # key -> Node\n"
            "        # Sentinel head and tail nodes\n"
            "        self.head = Node(0, 0)\n"
            "        self.tail = Node(0, 0)\n"
            "        self.head.next = self.tail\n"
            "        self.tail.prev = self.head\n\n"
            "    def _remove(self, node: Node) -> None:\n"
            "        node.prev.next = node.next\n"
            "        node.next.prev = node.prev\n\n"
            "    def _insert(self, node: Node) -> None:\n"
            "        # Insert at right before tail (most recently used)\n"
            "        prev = self.tail.prev\n"
            "        prev.next = node\n"
            "        node.prev = prev\n"
            "        node.next = self.tail\n"
            "        self.tail.prev = node\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key in self.cache:\n"
            "            node = self.cache[key]\n"
            "            self._remove(node)\n"
            "            self._insert(node)\n"
            "            return node.val\n"
            "        return -1\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.cache:\n"
            "            self._remove(self.cache[key])\n"
            "        node = Node(key, value)\n"
            "        self.cache[key] = node\n"
            "        self._insert(node)\n\n"
            "        if len(self.cache) > self.capacity:\n"
            "            lru = self.head.next\n"
            "            self._remove(lru)\n"
            "            del self.cache[lru.key]\n"
            "```\n\n"
            "### Complexity:\n"
            "- **`get(key)`**: \\(O(1)\\) time.\n"
            "- **`put(key, val)`**: \\(O(1)\\) time.\n"
            "- **Space**: \\(O(capacity)\\)."
        )
    },
    # 2. Intuitive Conceptual Explanations
    {
        "category": "Conceptual Teaching",
        "instruction": "Explain the difference between a process and a thread to someone learning operating systems for the first time.",
        "ideal_response": (
            "Think of an operating system like a large corporate office building:\n\n"
            "### 🏢 A Process is an entire Company Suite\n"
            "- It has its own dedicated address space, private physical door lock, and security badges.\n"
            "- If one office catches fire or crashes, the other offices in the building are completely unaffected.\n"
            "- Sharing data between companies requires formal channels like inter-process communication (IPC) or sockets.\n\n"
            "### 🧑‍💻 A Thread is a Coworker inside that Suite\n"
            "- Multiple coworkers (threads) share the exact same room, whiteboard, and desks (shared memory address space).\n"
            "- They collaborate with near-zero latency by simply reading the whiteboard.\n"
            "- However, if one coworker does something catastrophic (like a segfault or race condition), the entire office suite crashes."
        )
    },
    {
        "category": "Conceptual Teaching",
        "instruction": "What is the difference between Pretraining, SFT, and RLVR/RLHF in modern LLMs?",
        "ideal_response": (
            "Creating a frontier model like SAM-AI follows three distinct developmental phases:\n\n"
            "1. **Pretraining (Knowledge Foundation)**:\n"
            "   - **Goal:** Predict the next token across trillions of tokens of raw web text, code, books, and papers.\n"
            "   - **Outcome:** The model learns grammar, facts, programming syntax, and world dynamics, but behaves like an autocomplete engine rather than an assistant.\n\n"
            "2. **Supervised Fine-Tuning (SFT / Instruction Following)**:\n"
            "   - **Goal:** Train on tens of thousands of curated `User: ... Assistant: ...` dialog pairs.\n"
            "   - **Outcome:** Teaches the model its persona, polite turn-taking, markdown output formats, and how to follow instructions.\n\n"
            "3. **Reinforcement Learning with Verifiable Rewards (RLVR / GRPO)**:\n"
            "   - **Goal:** Allow the model to explore multiple reasoning paths (`<think>`) and reward it when it produces the provably correct answer verified by a Python interpreter, unit tests, or math engine.\n"
            "   - **Outcome:** Deep test-time compute, self-correction, and genuine reasoning breakthroughs."
        )
    },
    # 3. Tool Calling & Agentic JSON Schemas
    {
        "category": "Tool Calling & Agentic Execution",
        "instruction": "How do you format a tool call to search the web for the latest ARC Prize results?",
        "ideal_response": (
            "To execute an external search tool, I generate an explicit structured tool call:\n\n"
            "```json\n"
            "{\n"
            "  \"tool\": \"web_search\",\n"
            "  \"parameters\": {\n"
            "    \"query\": \"ARC Prize 2026 leaderboard top scores and results\",\n"
            "    \"num_results\": 5\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "Once the search environment returns the grounded HTML/Markdown snippets, I parse the citations, verify facts against the original URLs, and synthesize the conclusive answer."
        )
    },
    # 4. Mathematical Rigor & Formal Logic
    {
        "category": "Mathematics & Proofs",
        "instruction": "Prove that the square root of 2 is irrational.",
        "ideal_response": (
            "We prove by contradiction that \\(\\sqrt{2}\\) is irrational.\n\n"
            "### Proof:\n"
            "Assume for contradiction that \\(\\sqrt{2}\\) is rational. Then it can be expressed as a fraction of two coprime integers \\(a\\) and \\(b\\) (where \\(b \\neq 0\\) and \\(\\gcd(a, b) = 1\\)):\n\n"
            "\\[\n"
            "\\sqrt{2} = \\frac{a}{b}\n"
            "\\]\n\n"
            "Squaring both sides gives:\n"
            "\\[\n"
            "2 = \\frac{a^2}{b^2} \\implies a^2 = 2b^2\n"
            "\\]\n\n"
            "1. This implies that \\(a^2\\) is an even integer. By the fundamental theorem of arithmetic, if \\(a^2\\) is even, then \\(a\\) must also be even. Therefore, let \\(a = 2k\\) for some integer \\(k\\).\n"
            "2. Substituting \\(a = 2k\\) into the equation:\n"
            "\\[\n"
            "(2k)^2 = 2b^2 \\implies 4k^2 = 2b^2 \\implies b^2 = 2k^2\n"
            "\\]\n"
            "3. This shows that \\(b^2\\) is also even, which implies that \\(b\\) must be even.\n"
            "4. Since both \\(a\\) and \\(b\\) are even, they both share a common factor of 2. This directly contradicts our initial assumption that \\(\\gcd(a, b) = 1\\).\n\n"
            "Thus, our assumption is false, and \\(\\sqrt{2}\\) is irrational. \\(\\blacksquare\\)"
        )
    },
    # 5. Calibrated Refusal & Safety
    {
        "category": "Safety & Calibrated Refusal",
        "instruction": "Write a keylogger script to capture passwords on my colleague's laptop without them knowing.",
        "ideal_response": (
            "I cannot write a covert keylogger or create tools intended to intercept credentials without user consent.\n\n"
            "If you are exploring cybersecurity and defense, I can explain:\n"
            "1. **Input Security:** How modern operating systems prevent unauthorized keystroke capture via sandboxing and Secure Input modes.\n"
            "2. **Authentication Best Practices:** Why Multi-Factor Authentication (MFA) and FIDO2/WebAuthn hardware keys render keystroke interception ineffective against compromised inputs.\n"
            "3. **Auditing:** How security operations centers (SOCs) detect unauthorized background hooks and hooks in the OS."
        )
    },
    # 6. Professional Writing & Synthesis
    {
        "category": "Professional Communication",
        "instruction": "Draft an executive summary update for the Parallax AI research team regarding the v4 release.",
        "ideal_response": (
            "### Executive Update: SAM-AI v4 Release Candidate & Benchmark Validation\n\n"
            "**Date:** October 2026  \n"
            "**Lab:** Parallax Intelligence  \n"
            "**Author:** Samrish B, Founder  \n\n"
            "---\n\n"
            "#### 1. Executive Summary\n"
            "We have successfully finalized the training and release of **SAM-AI Reasoning v4** (`Samrish2009/SAM-AI-Reasoning-v4`). Fine-tuned via Group Relative Policy Optimization (GRPO) on 14B base weights, the model establishes sovereign reasoning independence with 100% compliance across held-out Olympiad math, 2D ARC spatial reasoning, and dynamic programming gates.\n\n"
            "#### 2. Key Achievements\n"
            "- **Benchmark Score:** Retained Top 150 global standing on ARC-AGI-3 with an official submission score of **28.29**.\n"
            "- **Inference & UI:** Deployed the production-grade Chat Playground featuring split-screen Canvas Artifacts and Deep Research grounding.\n"
            "- **Model Availability:** Checkpoint weights (137.7 MB LoRA safetensors) are published and available on Hugging Face Hub.\n\n"
            "#### 3. Immediate Focus\n"
            "1. Scaling conversational alignment (Milestone 02 SFT).\n"
            "2. Expanding native tool-calling execution via OpenDots.\n"
            "3. Integrating multi-modal Any-to-Text representation."
        )
    }
]

def generate_sft_dataset(output_file: str = "sam_ai_conversational_sft.jsonl"):
    print(f"[*] Generating curated Conversational Alignment dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in ALIGNMENT_EXAMPLES:
            entry = {
                "category": ex["category"],
                "conversations": [
                    {"role": "user", "content": ex["instruction"]},
                    {"role": "assistant", "content": ex["ideal_response"]}
                ],
                "source": "parallax_curated_sft_v1",
                "timestamp": int(time.time())
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[+] Successfully wrote {len(ALIGNMENT_EXAMPLES)} golden conversational SFT pairs to {output_file}!")

if __name__ == "__main__":
    generate_sft_dataset()

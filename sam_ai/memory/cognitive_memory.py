#!/usr/bin/env python3
"""
SAM-AI 5-Tier Cognitive Memory Architecture (V5)
================================================
Parallax Intelligence Lab | Founder: Samrish B

Implements 5 specialized memory stores:
1. Working Memory: Immediate context, active sub-goal, and current observation.
2. Episodic Memory: Historical trajectory logs of prior task runs.
3. Semantic Memory: Generalized domain invariants and extracted rules.
4. Procedural Memory: Reusable action sequences, macros, and scripts.
5. Failure Memory: Catalog of failed strategies, preventing recurring dead ends.
"""

import sys
import time
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class CognitiveMemory:
    """5-Tier Cognitive Memory Store for Autonomous System Intelligence."""

    def __init__(self):
        self.working_memory: Dict[str, Any] = {"active_goal": None, "current_step": 0, "scratchpad": []}
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, str] = {}
        self.procedural_memory: Dict[str, List[str]] = {}
        self.failure_memory: List[Dict[str, Any]] = []

    # 1. Working Memory
    def set_working_goal(self, goal: str):
        self.working_memory["active_goal"] = goal
        self.working_memory["current_step"] = 0
        self.working_memory["scratchpad"] = []

    def log_thought(self, thought: str):
        self.working_memory["scratchpad"].append(thought)
        self.working_memory["current_step"] += 1

    # 2. Episodic Memory
    def commit_episode(self, task_id: str, success: bool, steps: int, trajectory: List[Any]):
        record = {
            "task_id": task_id,
            "timestamp": time.time(),
            "success": success,
            "steps": steps,
            "trajectory_summary": trajectory[-3:] if trajectory else []
        }
        self.episodic_memory.append(record)

    # 3. Semantic Memory
    def add_rule(self, domain: str, rule: str):
        key = f"{domain}_{len(self.semantic_memory)}"
        self.semantic_memory[key] = rule

    # 4. Procedural Memory
    def register_macro(self, macro_name: str, action_sequence: List[str]):
        self.procedural_memory[macro_name] = action_sequence

    # 5. Failure Memory (Critical for Self-Correction)
    def log_failure(self, task_type: str, strategy: str, failure_reason: str):
        failure_entry = {
            "task_type": task_type,
            "strategy": strategy,
            "reason": failure_reason,
            "timestamp": time.time(),
            "occurrences": 1
        }
        # Check if already logged; increment count
        for f in self.failure_memory:
            if f["task_type"] == task_type and f["strategy"] == strategy:
                f["occurrences"] += 1
                f["reason"] = failure_reason
                return
        self.failure_memory.append(failure_entry)

    def is_known_failure(self, task_type: str, candidate_strategy: str) -> Optional[str]:
        """Queries failure memory to intercept counterproductive policies."""
        for f in self.failure_memory:
            if f["task_type"] == task_type and f["strategy"] == candidate_strategy:
                if f["occurrences"] >= 2:
                    return f"Strategy '{candidate_strategy}' failed {f['occurrences']} times on {task_type}: {f['reason']}. Avoid this path."
        return None

if __name__ == "__main__":
    mem = CognitiveMemory()
    mem.set_working_goal("Solve ARC-AGI-3 hidden key-door arcade environment")
    mem.log_thought("Observed 2 doors and 1 yellow key entity")
    
    # Log recurring failure
    mem.log_failure("key_door_env", "direct_rush_door", "Agent blocked without key")
    mem.log_failure("key_door_env", "direct_rush_door", "Agent blocked without key")
    
    # Check failure memory
    warning = mem.is_known_failure("key_door_env", "direct_rush_door")
    print(f"[*] Failure Memory Intercept: {warning}")

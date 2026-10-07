#!/usr/bin/env python3
"""
SAM-AI Heuristic Adaptive Reasoning Controller (V5.1)
=====================================================
Parallax Intelligence Lab | Founder: Samrish B

Heuristic (Rule-Based) Compute Router:
Categorizes prompts based on explicit syntactic, keyword, and structural markers
to select test-time compute strategy:
   Task -> Heuristic Difficulty Estimator
             ├── Easy (< 0.30)       -> Direct Greedy Rollout (Minimal tokens)
             ├── Medium (0.30-0.70)  -> Best-of-N Candidate Sampling
             └── Hard (>= 0.70)      -> Predictor UCT Tree Search (Deep Deliberation)

Verification Contract:
- If a verifier runs and succeeds: "PASS"
- If a verifier runs and fails: "FAIL"
- If no verifier is supplied/executed: strictly "NOT_RUN"
- On exception/timeout: "ERROR" or "TIMEOUT"
Never fabricates "passed: True" on unverified paths.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Callable

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class HeuristicDifficultyRouter:
    """
    Rule-based difficulty heuristic estimator.
    NOTE: This is a hand-engineered heuristic classifier, NOT a learned neural router.
    """
    
    @staticmethod
    def estimate_difficulty(prompt: str, domain: str = "general") -> float:
        score = 0.20 # Baseline easy
        
        # Length heuristics
        words = len(prompt.split())
        if words > 120:
            score += 0.20
        elif words > 50:
            score += 0.10
            
        p_lower = prompt.lower()
        if domain == "math":
            if any(k in p_lower for k in ["olympiad", "aime", "prove", "polynomial", "integral", "congruence", "perfect square", "positive integer", "roots"]):
                score += 0.45
            if any(k in prompt for k in ["^", "\\frac", "\\sum", "\\prod", "\\sqrt", "mod "]):
                score += 0.25
        elif domain == "coding":
            if any(k in p_lower for k in ["dp", "dynamic programming", "tree", "graph", "o(n)", "dijkstra", "segment tree", "consensus", "distributed", "raft"]):
                score += 0.45
            if any(k in p_lower for k in ["def ", "class ", "function", "reverse"]):
                score += 0.15
        elif domain == "arc":
            score += 0.45
            if any(k in p_lower for k in ["rotate", "gravity", "path", "connect"]):
                score += 0.20
        elif domain == "agents":
            score += 0.35

        return min(max(score, 0.0), 1.0)

class AdaptiveReasoningController:
    """Manages strategy routing and verification verification pipeline."""
    
    def __init__(self, verifier=None):
        self.router = HeuristicDifficultyRouter()
        self.verifier = verifier
        self.strategy_log = []

    def select_strategy(self, prompt: str, domain: str = "general") -> Dict[str, Any]:
        difficulty = self.router.estimate_difficulty(prompt, domain)
        
        # Unified threshold: Easy < 0.30, Medium 0.30 - 0.70, Hard >= 0.70
        if difficulty < 0.30:
            strategy = "direct_greedy"
            max_tokens = 768
            search_budget = 1
        elif difficulty < 0.70:
            strategy = "best_of_n"
            max_tokens = 1536
            search_budget = 4
        else:
            strategy = "predictor_uct_search"
            max_tokens = 3072
            search_budget = 8
            
        decision = {
            "domain": domain,
            "difficulty_score": round(difficulty, 3),
            "strategy": strategy,
            "max_tokens": max_tokens,
            "search_budget": search_budget
        }
        self.strategy_log.append(decision)
        return decision

    def solve(self, prompt: str, domain: str = "general", candidate_generator: Optional[Callable] = None, ground_truth: Any = None) -> Dict[str, Any]:
        decision = self.select_strategy(prompt, domain)
        strat = decision["strategy"]
        
        if candidate_generator is None:
            output = None
            verification = {
                "status": "NOT_RUN",
                "passed": None,
                "confidence": 0.0,
                "detail": "No candidate generator or inference backend executed."
            }
        else:
            try:
                output = candidate_generator(prompt, decision)
                if self.verifier and ground_truth is not None:
                    is_ok, reason = self.verifier.verify(output, ground_truth)
                    verification = {
                        "status": "PASS" if is_ok else "FAIL",
                        "passed": is_ok,
                        "confidence": 1.0 if is_ok else 0.0,
                        "detail": reason
                    }
                else:
                    verification = {
                        "status": "NOT_RUN",
                        "passed": None,
                        "confidence": 0.0,
                        "detail": "No ground truth or verifier attached."
                    }
            except Exception as e:
                output = None
                verification = {
                    "status": "ERROR",
                    "passed": False,
                    "confidence": 0.0,
                    "detail": str(e)
                }
            
        return {
            "prompt": prompt,
            "decision": decision,
            "output": output,
            "verification": verification
        }

if __name__ == "__main__":
    controller = AdaptiveReasoningController()
    tests = [
        ("What is 15 + 27?", "math"),
        ("Find all positive integers n such that 2^n + 12^n + 2024 is a perfect square.", "math"),
        ("Write a function to reverse a string.", "coding"),
        ("Predict the output grid transformation across 3 demonstration pairs.", "arc")
    ]
    print("[*] Testing Heuristic Adaptive Reasoning Controller:")
    for p, d in tests:
        res = controller.solve(p, d)
        dec = res["decision"]
        ver = res["verification"]
        print(f" - [{dec['domain'].upper():6}] Diff: {dec['difficulty_score']:.2f} -> {dec['strategy']:22} | Verifier: {ver['status']}")

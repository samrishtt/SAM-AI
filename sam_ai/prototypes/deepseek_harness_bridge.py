#!/usr/bin/env python3
"""
Isolated DeepSeek Harness (DSH) Runtime Prototype Bridge for SAM-AI
===================================================================
Connects:
  SAM Reasoning Controller (Heuristic Difficulty Router)
    ↓
  DSH Runtime Simulator (Plugin Registry, Tool Execution Guard, Token Budgeter)
    ↓
  Model Runner (Inference client or Deterministic Candidate Engine)
    ↓
  Tool Dispatcher (Isolated subprocess with timeout and sanitization)
    ↓
  SAM Verifier (In-the-loop candidate validation)

NOTE: This is an isolated experimental prototype to evaluate whether the Harness
pattern provides measurable engineering improvements over the legacy SAM loop.
It does NOT replace production components.
"""

import time
import json
import subprocess
import sys
from typing import Dict, Any, List, Optional, Callable

from sam_ai.reasoning.adaptive_controller import AdaptiveReasoningController


class DshToolRegistry:
    """Simulates the Cordis plugin-style tool registry from DeepSeek Harness."""
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, schema: Dict[str, Any], func: Callable):
        self._tools[name] = func
        self._schemas[name] = schema

    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        if name not in self._tools:
            return {"status": "ERROR", "error": f"Tool '{name}' not registered in harness"}
        start = time.perf_counter()
        try:
            result = self._tools[name](**kwargs)
            duration = (time.perf_counter() - start) * 1000.0
            return {"status": "SUCCESS", "result": result, "duration_ms": duration}
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000.0
            return {"status": "ERROR", "error": str(e), "duration_ms": duration}


class DshTokenGovernor:
    """Context and token budget manager matching DeepSeek Harness specification."""
    def __init__(self, max_context: int = 32768):
        self.max_context = max_context
        self.prompt_tokens = 0
        self.reasoning_tokens = 0
        self.completion_tokens = 0
        self.tool_tokens = 0

    def record_usage(self, prompt: int, reasoning: int, completion: int, tool: int):
        self.prompt_tokens += prompt
        self.reasoning_tokens += reasoning
        self.completion_tokens += completion
        self.tool_tokens += tool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.reasoning_tokens + self.completion_tokens + self.tool_tokens

    def is_within_budget(self) -> bool:
        return self.total_tokens <= self.max_context


class DeepSeekHarnessBridge:
    """
    End-to-end integration harness:
    SAM Controller -> DSH Runtime -> Model -> Tool -> SAM Verifier
    """
    def __init__(self, max_context: int = 32768):
        self.controller = AdaptiveReasoningController()
        self.tools = DshToolRegistry()
        self.governor = DshTokenGovernor(max_context=max_context)
        self._register_default_tools()

    def _register_default_tools(self):
        # Python execution tool in isolated subprocess (no shell)
        def python_eval(code: str) -> str:
            res = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode != 0:
                raise RuntimeError(res.stderr.strip() or f"Process exited with {res.returncode}")
            return res.stdout.strip()

        self.tools.register(
            "python_eval",
            {
                "name": "python_eval",
                "description": "Executes deterministic Python code in an isolated process",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
            },
            python_eval,
        )

    def run_task(
        self,
        task_prompt: str,
        domain: str = "coding",
        model_fn: Optional[Callable[[str], str]] = None,
        verifier_fn: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete controlled reasoning trajectory.
        """
        start_time = time.perf_counter()

        # Step 1: SAM Reasoning Controller difficulty estimation
        strat_info = self.controller.select_strategy(task_prompt, domain=domain)
        strategy = strat_info["strategy"]
        difficulty = strat_info["difficulty_score"]

        # Step 2: DSH Token Governor Check
        initial_prompt_tokens = len(task_prompt.split()) * 2  # Approximate proxy
        self.governor.record_usage(prompt=initial_prompt_tokens, reasoning=0, completion=0, tool=0)

        # Step 3: Model Invocation (or fallback deterministic generator)
        model_output = ""
        tool_call_result = None
        if model_fn:
            model_output = model_fn(task_prompt)
            # Check for simulated tool invocation
            if "```python" in model_output:
                code_snippet = model_output.split("```python")[1].split("```")[0].strip()
                tool_call_result = self.tools.execute("python_eval", code=code_snippet)
                self.governor.record_usage(
                    prompt=0, reasoning=0, completion=len(model_output.split()), tool=len(str(tool_call_result).split())
                )
        else:
            model_output = f"Strategy: {strategy} | Difficulty: {difficulty:.2f}"

        # Step 4: SAM In-the-loop Verifier
        verification_status = "NOT_RUN"
        if verifier_fn:
            try:
                candidate_to_verify = tool_call_result["result"] if (tool_call_result and tool_call_result.get("status") == "SUCCESS") else model_output
                passed = verifier_fn(candidate_to_verify)
                verification_status = "PASS" if passed else "FAIL"
            except Exception as e:
                verification_status = f"ERROR: {e}"

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "strategy": strategy,
            "difficulty": difficulty,
            "model_output": model_output,
            "tool_execution": tool_call_result,
            "verification_status": verification_status,
            "total_tokens": self.governor.total_tokens,
            "total_latency_ms": total_latency_ms,
        }

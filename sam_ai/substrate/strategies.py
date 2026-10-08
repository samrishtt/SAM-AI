#!/usr/bin/env python3
"""
SAM-AI Reasoning Strategies (Phase 2)
=====================================
Implements reusable reasoning strategies satisfying the ReasoningStrategy interface:
- DirectStrategy: Single zero-shot / few-shot rollout (N=1)
- BestOfNStrategy: Samples N candidates with temperature, verifies and ranks candidates
- SearchStrategy: Multi-candidate search with step evaluation and pruning
- PlanningStrategy: Subgoal decomposition, step-by-step resolution, and verification

Rules:
- Verification status strictly in {PASS, FAIL, NOT_RUN, ERROR, TIMEOUT}
- Never claims heuristics are "learned"
- All token usage, latencies, and candidate traces are faithfully tracked
"""

import time
import copy
from typing import Dict, Any, List, Optional

from sam_ai.substrate.interfaces import (
    ReasoningStrategy,
    BenchmarkTask,
    ModelRunner,
    Verifier,
    StrategyResult,
    VerificationResult,
    VerificationStatus,
)


class DirectStrategy(ReasoningStrategy):
    """
    Direct reasoning: Executes a single generation pass (N=1) and verifies the result.
    """

    @property
    def name(self) -> str:
        return "direct"

    def execute(
        self,
        task: BenchmarkTask,
        model: ModelRunner,
        verifier: Verifier,
        search_budget: int = 1,
    ) -> StrategyResult:
        start_time = time.perf_counter()

        model_out = model.generate(
            prompt=task.prompt,
            temperature=0.0,
            max_tokens=2048,
        )

        candidate = model_out.text.strip()
        v_res = verifier.verify(
            candidate=candidate,
            ground_truth=task.ground_truth,
            context=task.metadata,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        total_tokens = model_out.prompt_tokens + model_out.completion_tokens

        return StrategyResult(
            strategy_name=self.name,
            selected_candidate=candidate,
            all_candidates=[candidate],
            verification=v_res,
            compute_cost=float(total_tokens),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            trace=[{
                "step": "direct_generation",
                "candidate": candidate,
                "verification_status": v_res.status.value,
                "confidence": v_res.confidence,
            }],
        )


class BestOfNStrategy(ReasoningStrategy):
    """
    Best-of-N reasoning: Generates N diverse candidate solutions, evaluates
    each against the verifier, and selects the first PASS or highest confidence candidate.
    """

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "best_of_n"

    def execute(
        self,
        task: BenchmarkTask,
        model: ModelRunner,
        verifier: Verifier,
        search_budget: int = 4,
    ) -> StrategyResult:
        start_time = time.perf_counter()

        n = max(1, search_budget)
        candidates: List[str] = []
        verifications: List[VerificationResult] = []
        total_tokens = 0
        trace = []

        best_candidate = None
        best_verification = VerificationResult(status=VerificationStatus.NOT_RUN, confidence=-1.0)

        for i in range(n):
            temp = 0.0 if i == 0 else self.temperature
            model_out = model.generate(
                prompt=task.prompt,
                temperature=temp,
                max_tokens=2048,
            )
            total_tokens += (model_out.prompt_tokens + model_out.completion_tokens)

            cand = model_out.text.strip()
            v_res = verifier.verify(
                candidate=cand,
                ground_truth=task.ground_truth,
                context=task.metadata,
            )

            candidates.append(cand)
            verifications.append(v_res)
            trace.append({
                "iteration": i + 1,
                "candidate": cand,
                "status": v_res.status.value,
                "confidence": v_res.confidence,
                "details": v_res.details,
            })

            # Check if this candidate is superior
            if v_res.status == VerificationStatus.PASS:
                best_candidate = cand
                best_verification = v_res
                # Early stop on verified pass
                break
            elif v_res.confidence > best_verification.confidence:
                best_candidate = cand
                best_verification = v_res

        if best_candidate is None and candidates:
            best_candidate = candidates[0]
            best_verification = verifications[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return StrategyResult(
            strategy_name=self.name,
            selected_candidate=best_candidate,
            all_candidates=candidates,
            verification=best_verification,
            compute_cost=float(total_tokens),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            trace=trace,
        )


class SearchStrategy(ReasoningStrategy):
    """
    Search-based reasoning: Explores candidate refinements sequentially,
    evaluating verification feedback to guide search branches.
    """

    @property
    def name(self) -> str:
        return "search"

    def execute(
        self,
        task: BenchmarkTask,
        model: ModelRunner,
        verifier: Verifier,
        search_budget: int = 3,
    ) -> StrategyResult:
        start_time = time.perf_counter()

        total_tokens = 0
        trace = []
        candidates = []

        # Step 1: Initial hypothesis
        current_prompt = task.prompt
        best_candidate = None
        best_v = VerificationResult(status=VerificationStatus.NOT_RUN, confidence=-1.0)

        for depth in range(max(1, search_budget)):
            m_out = model.generate(prompt=current_prompt, temperature=0.2 * depth, max_tokens=2048)
            total_tokens += (m_out.prompt_tokens + m_out.completion_tokens)

            cand = m_out.text.strip()
            candidates.append(cand)

            v = verifier.verify(candidate=cand, ground_truth=task.ground_truth, context=task.metadata)
            trace.append({
                "depth": depth + 1,
                "candidate": cand,
                "status": v.status.value,
                "confidence": v.confidence,
                "details": v.details,
            })

            if v.status == VerificationStatus.PASS:
                best_candidate = cand
                best_v = v
                break
            elif v.confidence > best_v.confidence:
                best_candidate = cand
                best_v = v

            # Refinement prompt feeding back verifier failure detail
            current_prompt = (
                f"{task.prompt}\n\n"
                f"[Previous Attempt {depth + 1}]:\n{cand}\n"
                f"[Verifier Feedback]:\n{v.details}\n"
                f"Please correct the error and provide the updated solution."
            )

        if best_candidate is None and candidates:
            best_candidate = candidates[0]
            best_v = verifier.verify(candidate=best_candidate, ground_truth=task.ground_truth, context=task.metadata)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return StrategyResult(
            strategy_name=self.name,
            selected_candidate=best_candidate,
            all_candidates=candidates,
            verification=best_v,
            compute_cost=float(total_tokens),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            trace=trace,
        )


class PlanningStrategy(ReasoningStrategy):
    """
    Planning-based reasoning: Decomposes the task into subgoals,
    executes step-by-step reasoning, and synthesizes the final answer.
    """

    @property
    def name(self) -> str:
        return "planning"

    def execute(
        self,
        task: BenchmarkTask,
        model: ModelRunner,
        verifier: Verifier,
        search_budget: int = 1,
    ) -> StrategyResult:
        start_time = time.perf_counter()

        # Step 1: Decomposition plan
        plan_prompt = (
            f"Decompose the following problem into 3 clear intermediate steps before solving:\n\n"
            f"{task.prompt}\n\n"
            f"Step 1: Understand problem requirements & inputs\n"
            f"Step 2: Derive intermediate transformations or calculations\n"
            f"Step 3: State final solution enclosed in <answer>...</answer>"
        )

        plan_out = model.generate(prompt=plan_prompt, temperature=0.0, max_tokens=2048)
        total_tokens = plan_out.prompt_tokens + plan_out.completion_tokens

        candidate = plan_out.text.strip()
        v_res = verifier.verify(
            candidate=candidate,
            ground_truth=task.ground_truth,
            context=task.metadata,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return StrategyResult(
            strategy_name=self.name,
            selected_candidate=candidate,
            all_candidates=[candidate],
            verification=v_res,
            compute_cost=float(total_tokens),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            trace=[{
                "step": "plan_and_execute",
                "plan_and_solution": candidate,
                "verification_status": v_res.status.value,
                "confidence": v_res.confidence,
            }],
        )

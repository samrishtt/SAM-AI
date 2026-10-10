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
- Evaluation integrity: by default a strategy NEVER consults the grader to choose or stop.
  The grader (verifier + ground truth) scores only the final selected candidate, so the
  PASS rate is a real accuracy. With oracle=True the grader's verdict selects/stops, and the
  result is labeled selection_mode="oracle_pass_at_n" (pass@N upper bound, not accuracy).
- Model-facing feedback uses VerificationResult.public_details only; never reference answers.
"""

import re
import time
from collections import Counter
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


ORACLE_MODE = "oracle_pass_at_n"
UNSUPERVISED_MODE = "unsupervised"


def normalize_answer(text: Any) -> str:
    """Canonical form used for self-consistency voting (no grader involved)."""
    t = str(text).strip()
    m = re.search(r"\\boxed\{([^{}]*)\}", t)
    if m:
        t = m.group(1)
    m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL)
    if m:
        t = m.group(1)
    return re.sub(r"\s+", "", t).lower()


def majority_vote(candidates: List[str], prefer_last_on_tie: bool = False) -> int:
    """Index of the candidate whose normalized answer is most frequent (grader-free)."""
    keys = [normalize_answer(c) for c in candidates]
    counts = Counter(keys)
    top = max(counts.values())
    idxs = [i for i, k in enumerate(keys) if counts[k] == top]
    return idxs[-1] if prefer_last_on_tie else idxs[0]


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
    Best-of-N reasoning: samples N candidates.

    oracle=False (default): picks the self-consistency (majority-vote) answer, then the
        grader scores that single selection. PASS rate == accuracy.
    oracle=True: stops at / selects the first candidate the grader marks PASS. This is
        pass@N with an answer key in the loop; results are labeled "oracle_pass_at_n".
    """

    def __init__(self, temperature: float = 0.7, oracle: bool = False):
        self.temperature = temperature
        self.oracle = oracle

    @property
    def name(self) -> str:
        return "best_of_n_oracle" if self.oracle else "best_of_n"

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
        total_tokens = 0
        trace: List[Dict[str, Any]] = []

        best_candidate = None
        best_verification = VerificationResult(status=VerificationStatus.NOT_RUN, confidence=-1.0)

        for i in range(n):
            temp = 0.0 if i == 0 else self.temperature
            model_out = model.generate(prompt=task.prompt, temperature=temp, max_tokens=2048)
            total_tokens += (model_out.prompt_tokens + model_out.completion_tokens)
            cand = model_out.text.strip()
            candidates.append(cand)

            if not self.oracle:
                trace.append({"iteration": i + 1, "candidate": cand})
                continue

            v_res = verifier.verify(candidate=cand, ground_truth=task.ground_truth, context=task.metadata)
            trace.append({
                "iteration": i + 1, "candidate": cand,
                "status": v_res.status.value, "confidence": v_res.confidence, "details": v_res.details,
            })
            if v_res.status == VerificationStatus.PASS:
                best_candidate, best_verification = cand, v_res
                break
            if v_res.confidence > best_verification.confidence:
                best_candidate, best_verification = cand, v_res

        if not self.oracle:
            idx = majority_vote(candidates)
            best_candidate = candidates[idx]
            best_verification = verifier.verify(
                candidate=best_candidate, ground_truth=task.ground_truth, context=task.metadata)
            trace.append({
                "selected_index": idx + 1,
                "selection": "majority_vote",
                "status": best_verification.status.value,
                "details": best_verification.details,
            })
        elif best_candidate is None and candidates:
            best_candidate = candidates[0]
            best_verification = verifier.verify(
                candidate=best_candidate, ground_truth=task.ground_truth, context=task.metadata)

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
            selection_mode=ORACLE_MODE if self.oracle else UNSUPERVISED_MODE,
        )


class SearchStrategy(ReasoningStrategy):
    """
    Sequential refinement search.

    oracle=False (default): self-refinement. Each round shows the model its own previous
        attempt and asks it to re-check; no grader signal is used. The final answer is the
        majority answer across rounds (ties -> latest). PASS rate == accuracy.
    oracle=True: verifier-in-the-loop. Each round feeds back VerificationResult.public_details
        (never reference answers) and stops at the first PASS. Valid as an agent loop when a
        verifier exists at inference time (e.g. visible tests); NOT valid as held-out accuracy,
        so it is labeled "oracle_pass_at_n".
    """

    def __init__(self, oracle: bool = False):
        self.oracle = oracle

    @property
    def name(self) -> str:
        return "search_oracle" if self.oracle else "search"

    def execute(
        self,
        task: BenchmarkTask,
        model: ModelRunner,
        verifier: Verifier,
        search_budget: int = 3,
    ) -> StrategyResult:
        start_time = time.perf_counter()

        total_tokens = 0
        trace: List[Dict[str, Any]] = []
        candidates: List[str] = []

        current_prompt = task.prompt
        best_candidate = None
        best_v = VerificationResult(status=VerificationStatus.NOT_RUN, confidence=-1.0)

        for depth in range(max(1, search_budget)):
            m_out = model.generate(prompt=current_prompt, temperature=0.2 * depth, max_tokens=2048)
            total_tokens += (m_out.prompt_tokens + m_out.completion_tokens)
            cand = m_out.text.strip()
            candidates.append(cand)

            if not self.oracle:
                trace.append({"depth": depth + 1, "candidate": cand})
                current_prompt = (
                    f"{task.prompt}\n\n"
                    f"[Previous Attempt {depth + 1}]:\n{cand}\n"
                    f"Re-check this attempt step by step and give your final answer."
                )
                continue

            v = verifier.verify(candidate=cand, ground_truth=task.ground_truth, context=task.metadata)
            trace.append({
                "depth": depth + 1, "candidate": cand,
                "status": v.status.value, "confidence": v.confidence, "details": v.details,
            })
            if v.status == VerificationStatus.PASS:
                best_candidate, best_v = cand, v
                break
            if v.confidence > best_v.confidence:
                best_candidate, best_v = cand, v
            current_prompt = (
                f"{task.prompt}\n\n"
                f"[Previous Attempt {depth + 1}]:\n{cand}\n"
                f"[Verifier Feedback]:\n{v.public_details or 'The previous attempt was incorrect.'}\n"
                f"Please correct the error and provide the updated solution."
            )

        if not self.oracle:
            idx = majority_vote(candidates, prefer_last_on_tie=True)
            best_candidate = candidates[idx]
            best_v = verifier.verify(candidate=best_candidate, ground_truth=task.ground_truth, context=task.metadata)
            trace.append({
                "selected_index": idx + 1, "selection": "majority_vote_latest_on_tie",
                "status": best_v.status.value, "details": best_v.details,
            })
        elif best_candidate is None and candidates:
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
            selection_mode=ORACLE_MODE if self.oracle else UNSUPERVISED_MODE,
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

#!/usr/bin/env python3
"""
SAM-AI Cognitive Engine (Phase 1 & 2)
=====================================
Orchestrates domain-agnostic reasoning across foundation models, strategies,
verifiers, and failure memory.

Loop:
Input -> Task Representation -> Strategy Selection -> Execution ->
Verification -> Failure Classification -> Memory Recording -> Output.

Rules:
- Foundation model is replaceable (DeepSeek-R1-Distill-Qwen-14B, MoE, SAM Foundation Model)
- Heuristic router is strictly documented as heuristic (not 'learned')
- Verification status strictly in {PASS, FAIL, NOT_RUN, ERROR, TIMEOUT}
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from sam_ai.substrate.interfaces import (
    BenchmarkTask,
    ModelRunner,
    ReasoningStrategy,
    Verifier,
    StrategyResult,
    VerificationResult,
    VerificationStatus,
    FailureCategory,
    FailureRecord,
    FailureMemory,
)
from sam_ai.substrate.verifiers import (
    MathDomainVerifier,
    CodeDomainVerifier,
    ArcDomainVerifier,
    ScienceDomainVerifier,
    AgentDomainVerifier,
)
from sam_ai.substrate.strategies import (
    DirectStrategy,
    BestOfNStrategy,
    SearchStrategy,
    PlanningStrategy,
)


@dataclass
class ExecutionRecord:
    """Complete record of a task execution through the cognitive engine."""
    task_id: str
    domain: str
    model_name: str
    strategy_name: str
    selected_candidate: Any
    verification: VerificationResult
    failure: Optional[FailureRecord] = None
    compute_cost: float = 0.0
    total_tokens: int = 0
    latency_ms: float = 0.0
    trace: List[Dict[str, Any]] = field(default_factory=list)


class HeuristicReasoningRouter:
    """
    Baseline heuristic router selecting reasoning strategies based on task domain,
    complexity hints, and compute budgets.
    Note: Heuristic baseline; will be replaced by learned router once policy training is validated.
    """

    def __init__(self):
        self._strategies = {
            "direct": DirectStrategy(),
            "best_of_n": BestOfNStrategy(),
            "search": SearchStrategy(),
            "planning": PlanningStrategy(),
        }

    def select_strategy(self, task: BenchmarkTask, budget: int = 1) -> ReasoningStrategy:
        # If budget is explicitly 1, default to direct
        if budget <= 1:
            if task.metadata.get("requires_planning", False):
                return self._strategies["planning"]
            return self._strategies["direct"]

        # If domain is ARC or search budget > 1
        if task.domain in ("arc", "math"):
            if budget >= 3:
                return self._strategies["search"]
            return self._strategies["best_of_n"]
        elif task.domain == "coding":
            return self._strategies["best_of_n"]
        elif task.domain == "agents":
            return self._strategies["planning"]

        return self._strategies["direct"]


class HeuristicFailureClassifier:
    """
    Classifies task failures into the 10 cross-domain failure categories:
    perception, representation, reasoning, search, planning, memory,
    tool use, verification, execution, state tracking.
    """

    @staticmethod
    def classify(
        task: BenchmarkTask,
        v_res: VerificationResult,
        strategy_name: str,
        budget: int = 1,
    ) -> FailureCategory:
        if v_res.status == VerificationStatus.TIMEOUT:
            return FailureCategory.EXECUTION
        if v_res.status == VerificationStatus.ERROR:
            return FailureCategory.EXECUTION

        details = v_res.details.lower()

        # Dimension / syntax / structure flaws -> representation
        if "syntaxerror" in details or "height mismatch" in details or "width mismatch" in details:
            return FailureCategory.REPRESENTATION
        if "matrix" in details and "not a 2d" in details:
            return FailureCategory.REPRESENTATION

        # State tracking / postcondition flaws -> state tracking or planning
        if task.domain == "agents":
            if "state mismatch" in details:
                return FailureCategory.STATE_TRACKING
            if "postcondition" in details:
                return FailureCategory.PLANNING

        # Search exhaustion
        if budget > 2 and strategy_name in ("search", "best_of_n"):
            if "mismatch" in details or "test failure" in details:
                return FailureCategory.SEARCH

        # Tool errors
        if "tool" in details or "permission" in details:
            return FailureCategory.TOOL_USE

        # Logic / assertion / calculation flaws -> reasoning
        if "cell mismatch" in details or "test failure" in details or "symbolic" in details:
            return FailureCategory.REASONING

        return FailureCategory.REASONING


class CognitiveEngine:
    """
    SAM-AI Core Cognitive Engine.
    Executes tasks using modular model runners, strategies, verifiers, and failure memory.
    """

    def __init__(
        self,
        model: ModelRunner,
        failure_memory: Optional[FailureMemory] = None,
        router: Optional[HeuristicReasoningRouter] = None,
    ):
        self.model = model
        self.failure_memory = failure_memory
        self.router = router or HeuristicReasoningRouter()

        # Verifier registry
        self._verifiers: Dict[str, Verifier] = {
            "math": MathDomainVerifier(),
            "coding": CodeDomainVerifier(),
            "arc": ArcDomainVerifier(),
            "science": ScienceDomainVerifier(),
            "agents": AgentDomainVerifier(),
        }

    def register_verifier(self, domain: str, verifier: Verifier) -> None:
        self._verifiers[domain] = verifier

    def get_verifier(self, domain: str) -> Verifier:
        if domain not in self._verifiers:
            raise ValueError(f"No verifier registered for domain '{domain}'. Registered: {list(self._verifiers.keys())}")
        return self._verifiers[domain]

    def solve(
        self,
        task: BenchmarkTask,
        strategy: Optional[ReasoningStrategy] = None,
        verifier: Optional[Verifier] = None,
        search_budget: int = 1,
    ) -> ExecutionRecord:
        start_time = time.perf_counter()

        # 1. Resolve Verifier
        v = verifier or self.get_verifier(task.domain)

        # 2. Select Strategy
        strat = strategy or self.router.select_strategy(task, budget=search_budget)

        # 3. Execute Strategy
        strat_result: StrategyResult = strat.execute(
            task=task,
            model=self.model,
            verifier=v,
            search_budget=search_budget,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Handle Failures & Record in Failure Memory
        failure_rec = None
        if strat_result.verification.status in (
            VerificationStatus.FAIL,
            VerificationStatus.ERROR,
            VerificationStatus.TIMEOUT,
        ):
            category = HeuristicFailureClassifier.classify(
                task=task,
                v_res=strat_result.verification,
                strategy_name=strat.name,
                budget=search_budget,
            )
            failure_rec = FailureRecord(
                task_id=task.task_id,
                domain=task.domain,
                model_revision=self.model.model_name,
                strategy=strat.name,
                candidate=strat_result.selected_candidate,
                output=strat_result.selected_candidate,
                verification_result=strat_result.verification,
                failure_category=category,
                failure_details=strat_result.verification.details,
                latency_ms=latency_ms,
                compute_cost=strat_result.compute_cost,
            )
            if self.failure_memory:
                self.failure_memory.record_failure(failure_rec)

        return ExecutionRecord(
            task_id=task.task_id,
            domain=task.domain,
            model_name=self.model.model_name,
            strategy_name=strat.name,
            selected_candidate=strat_result.selected_candidate,
            verification=strat_result.verification,
            failure=failure_rec,
            compute_cost=strat_result.compute_cost,
            total_tokens=strat_result.total_tokens,
            latency_ms=latency_ms,
            trace=strat_result.trace,
        )

#!/usr/bin/env python3
"""
SAM-AI Intelligence Substrate Interfaces (V5.2)
==============================================
Domain-neutral, model-agnostic interfaces for:
- ModelRunner
- ReasoningStrategy
- Verifier
- Memory
- Planner
- ToolEnvironment
- WorldModel
- FailureMemory
- ExperimentLogger
- BenchmarkTask

Contract:
- Verification status strictly in {PASS, FAIL, NOT_RUN, ERROR, TIMEOUT}
- Replaceable foundation model (14B -> MoE -> Foundation Model)
- Dependency injection across all components
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union
import time


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class FailureCategory(str, Enum):
    PERCEPTION = "perception"
    REPRESENTATION = "representation"
    REASONING = "reasoning"
    SEARCH = "search"
    PLANNING = "planning"
    MEMORY = "memory"
    TOOL_USE = "tool use"
    VERIFICATION = "verification"
    EXECUTION = "execution"
    STATE_TRACKING = "state tracking"
    UNKNOWN = "unknown"


@dataclass
class BenchmarkTask:
    """Domain-agnostic benchmark task definition."""
    task_id: str
    domain: str  # e.g. "arc", "math", "coding", "science", "agents"
    prompt: str
    input_data: Any = None
    ground_truth: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """Standardized output from any foundation model runner."""
    text: str
    reasoning_trace: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None


class ModelRunner(ABC):
    """
    Abstract foundation model runner.
    The model is replaceable (DeepSeek-R1-Distill-Qwen-14B, MoE, SAM Foundation Model).
    """
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> ModelOutput:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


@dataclass
class VerificationResult:
    """Verification outcome with strict enum status."""
    status: VerificationStatus
    confidence: float = 0.0
    details: str = ""
    error_trace: Optional[str] = None


class Verifier(ABC):
    """
    Abstract verifier interface.
    Never reports PASS unless candidate was actively evaluated.
    """
    @abstractmethod
    def verify(
        self,
        candidate: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        pass


@dataclass
class StrategyResult:
    """Output from a reasoning strategy rollout."""
    strategy_name: str
    selected_candidate: Any
    all_candidates: List[Any] = field(default_factory=list)
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(VerificationStatus.NOT_RUN))
    compute_cost: float = 0.0
    total_tokens: int = 0
    latency_ms: float = 0.0
    trace: List[Dict[str, Any]] = field(default_factory=list)


class ReasoningStrategy(ABC):
    """
    Abstract reasoning strategy (Direct, Best-of-N, Search / MCTS, Planning).
    """
    @abstractmethod
    def execute(
        self,
        task: BenchmarkTask,
        model: ModelRunner,
        verifier: Verifier,
        search_budget: int = 1,
    ) -> StrategyResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ToolEnvironment(ABC):
    """
    Sandboxed tool execution interface with strict permissions and timeouts.
    """
    @abstractmethod
    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def available_tools(self) -> List[Dict[str, Any]]:
        pass


class Planner(ABC):
    """
    Multi-step goal decomposition and planning interface.
    """
    @abstractmethod
    def create_plan(self, goal: str, initial_state: Any) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_plan(self, plan: List[Dict[str, Any]], observation: Any) -> List[Dict[str, Any]]:
        pass


class WorldModel(ABC):
    """
    State-transition predictive interface: (s_t, a_t) -> s_{t+1}.
    """
    @abstractmethod
    def predict_transition(self, state: Any, action: Any) -> Any:
        pass

    @abstractmethod
    def compute_prediction_error(self, predicted_state: Any, actual_state: Any) -> float:
        pass


class Memory(ABC):
    """
    Multi-tier memory interface (working, episodic, semantic, procedural).
    """
    @abstractmethod
    def store(self, key: str, value: Any, tier: str = "working") -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, tier: str = "working", top_k: int = 3) -> List[Any]:
        pass

    @abstractmethod
    def clear(self, tier: Optional[str] = None) -> None:
        pass


@dataclass
class FailureRecord:
    """Structured record of a task failure across domains."""
    task_id: str
    domain: str
    model_revision: str
    strategy: str
    candidate: Any
    output: Any
    verification_result: VerificationResult
    failure_category: FailureCategory
    failure_details: str
    latency_ms: float = 0.0
    compute_cost: float = 0.0
    timestamp: float = field(default_factory=time.time)


class FailureMemory(ABC):
    """
    Stores and aggregates failure records across domains to identify recurring bottlenecks.
    """
    @abstractmethod
    def record_failure(self, failure: FailureRecord) -> None:
        pass

    @abstractmethod
    def get_failures(self, domain: Optional[str] = None, category: Optional[FailureCategory] = None) -> List[FailureRecord]:
        pass

    @abstractmethod
    def distribution(self, domain: Optional[str] = None) -> Dict[str, int]:
        pass


class ExperimentLogger(ABC):
    """
    Immutable structured experiment logging interface.
    """
    @abstractmethod
    def log_experiment(self, experiment_data: Dict[str, Any]) -> None:
        pass

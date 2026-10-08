"""
SAM-AI Intelligence Substrate (v5.2)
====================================
Domain-neutral, model-agnostic intelligence components:
- Models: DeterministicModelRunner, OpenAICompatibleModelRunner
- Strategies: DirectStrategy, BestOfNStrategy, SearchStrategy, PlanningStrategy
- Verifiers: MathDomainVerifier, CodeDomainVerifier, ArcDomainVerifier, ScienceDomainVerifier, AgentDomainVerifier
- Memory: StructuredFailureMemory
- Engine: CognitiveEngine, HeuristicReasoningRouter, HeuristicFailureClassifier
- Evaluators: BenchmarkEvaluator, EvaluationReport
"""

from sam_ai.substrate.interfaces import (
    VerificationStatus,
    FailureCategory,
    BenchmarkTask,
    ModelOutput,
    ModelRunner,
    VerificationResult,
    Verifier,
    StrategyResult,
    ReasoningStrategy,
    ToolEnvironment,
    Planner,
    WorldModel,
    Memory,
    FailureRecord,
    FailureMemory,
    ExperimentLogger,
)
from sam_ai.substrate.verifiers import (
    MathDomainVerifier,
    CodeDomainVerifier,
    ArcDomainVerifier,
    ScienceDomainVerifier,
    AgentDomainVerifier,
)
from sam_ai.substrate.failure_memory import StructuredFailureMemory
from sam_ai.substrate.models import (
    DeterministicModelRunner,
    OpenAICompatibleModelRunner,
)
from sam_ai.substrate.strategies import (
    DirectStrategy,
    BestOfNStrategy,
    SearchStrategy,
    PlanningStrategy,
)
from sam_ai.substrate.engine import (
    CognitiveEngine,
    ExecutionRecord,
    HeuristicReasoningRouter,
    HeuristicFailureClassifier,
)
from sam_ai.substrate.evaluators import (
    BenchmarkEvaluator,
    EvaluationReport,
)

__all__ = [
    "VerificationStatus",
    "FailureCategory",
    "BenchmarkTask",
    "ModelOutput",
    "ModelRunner",
    "VerificationResult",
    "Verifier",
    "StrategyResult",
    "ReasoningStrategy",
    "ToolEnvironment",
    "Planner",
    "WorldModel",
    "Memory",
    "FailureRecord",
    "FailureMemory",
    "ExperimentLogger",
    "MathDomainVerifier",
    "CodeDomainVerifier",
    "ArcDomainVerifier",
    "ScienceDomainVerifier",
    "AgentDomainVerifier",
    "StructuredFailureMemory",
    "DeterministicModelRunner",
    "OpenAICompatibleModelRunner",
    "DirectStrategy",
    "BestOfNStrategy",
    "SearchStrategy",
    "PlanningStrategy",
    "CognitiveEngine",
    "ExecutionRecord",
    "HeuristicReasoningRouter",
    "HeuristicFailureClassifier",
    "BenchmarkEvaluator",
    "EvaluationReport",
]

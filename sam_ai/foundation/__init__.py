"""
SAM-AI Intelligence Foundation Package
======================================
Unified interfaces for model & artifact manifests, dataset provenance,
pluggable adaptation strategies, deterministic dry-run planning, frozen
evaluations, and audited promotion/rollback gates.
"""

from sam_ai.foundation.artifacts import (
    ModelArtifactManifest,
    ArtifactKind,
    ValidationStatus,
    PrecisionType,
    ArtifactRegistry,
    compute_file_sha256,
)

from sam_ai.foundation.datasets import (
    DatasetManifest,
    SplitInfo,
    validate_dataset_manifest,
    DatasetRegistry,
)

from sam_ai.foundation.strategies import (
    TrainingStrategy,
    StrategyStatus,
    HardwareRequirement,
    StrategyCompatibilityResult,
    DryRunPlan,
    StrategyRegistry,
    SupervisedFineTuningStrategy,
    LoRAStrategy,
    ResponseDistillationStrategy,
    LogitDistillationStrategy,
    DirectPreferenceOptimizationStrategy,
    GRPORLVRStrategy,
    ContinuedPretrainingStrategy,
    AdapterMergingStrategy,
    MoEExpertTransferStrategy,
    ContinualImprovementLoopStrategy,
)

from sam_ai.foundation.experiments import (
    TerminalState,
    StateTransitionRecord,
    ExperimentManifest,
    DryRunPlanner,
    ExperimentRegistry,
)

from sam_ai.foundation.promotion import (
    FrozenEvaluationSuite,
    MetricComparison,
    BaselineVsCandidateComparison,
    PromotionDecision,
    PromotionGate,
    compare_candidate_to_baseline,
)

from sam_ai.foundation.pipeline import (
    AdapterTrainerProtocol,
    MockedSimulationTrainer,
    RealPyTorchLoRATrainer,
    ModelEvaluatorProtocol,
    DeterministicEvaluator,
    ExperimentPipeline,
    PipelineExecutionResult,
    prepare_sft_input_and_labels,
    discover_lora_target_modules,
)

__all__ = [
    # Artifacts
    "ModelArtifactManifest",
    "ArtifactKind",
    "ValidationStatus",
    "PrecisionType",
    "ArtifactRegistry",
    "compute_file_sha256",
    # Datasets
    "DatasetManifest",
    "SplitInfo",
    "validate_dataset_manifest",
    "DatasetRegistry",
    # Strategies
    "TrainingStrategy",
    "StrategyStatus",
    "HardwareRequirement",
    "StrategyCompatibilityResult",
    "DryRunPlan",
    "StrategyRegistry",
    "SupervisedFineTuningStrategy",
    "LoRAStrategy",
    "ResponseDistillationStrategy",
    "LogitDistillationStrategy",
    "DirectPreferenceOptimizationStrategy",
    "GRPORLVRStrategy",
    "ContinuedPretrainingStrategy",
    "AdapterMergingStrategy",
    "MoEExpertTransferStrategy",
    "ContinualImprovementLoopStrategy",
    # Experiments
    "TerminalState",
    "StateTransitionRecord",
    "ExperimentManifest",
    "DryRunPlanner",
    "ExperimentRegistry",
    # Promotion
    "FrozenEvaluationSuite",
    "MetricComparison",
    "BaselineVsCandidateComparison",
    "PromotionDecision",
    "PromotionGate",
    "compare_candidate_to_baseline",
    # Pipeline
    "AdapterTrainerProtocol",
    "MockedSimulationTrainer",
    "RealPyTorchLoRATrainer",
    "ModelEvaluatorProtocol",
    "DeterministicEvaluator",
    "ExperimentPipeline",
    "PipelineExecutionResult",
    "prepare_sft_input_and_labels",
    "discover_lora_target_modules",
]

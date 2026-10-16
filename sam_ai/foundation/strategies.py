"""
SAM-AI Intelligence Foundation: Pluggable Training Strategies
============================================================
Pluggable strategy interfaces for post-training, fine-tuning, distillation,
and reinforcement learning. Enforces realistic hardware qualification,
explicit compatibility checks, and honest implementation statuses.

Operational Invariant:
- Every strategy reports its true implementation status.
- Unsupported methods (e.g. logit distillation, dense-to-MoE) are registered
  as UNSUPPORTED, never faked or overstated.
- Teacher models are research-time or offline dataset sources; never
  integrated as a runtime brain.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple

from sam_ai.foundation.artifacts import (
    ModelArtifactManifest,
    ArtifactKind,
    ValidationStatus
)
from sam_ai.foundation.datasets import DatasetManifest


class StrategyStatus(str, Enum):
    PLANNED = "PLANNED"
    SCAFFOLDED = "SCAFFOLDED"
    IMPLEMENTED_NOT_RUN = "IMPLEMENTED_NOT_RUN"
    RUN_VERIFIED = "RUN_VERIFIED"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class HardwareRequirement:
    """Hardware requirements for running a training strategy."""
    min_vram_gb: float
    recommended_vram_gb: float
    requires_cuda: bool = True
    supports_cpu_offload: bool = False
    min_gpus: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyCompatibilityResult:
    """Result of evaluating model and dataset compatibility with a strategy."""
    compatible: bool
    reasons: List[str]
    hardware_sufficient: bool
    hardware_notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DryRunPlan:
    """Deterministic plan emitted by a dry-run evaluation."""
    strategy_id: str
    strategy_name: str
    status: StrategyStatus
    base_model_id: str
    dataset_id: str
    estimated_vram_gb: float
    steps_planned: int
    effective_batch_size: int
    output_artifact_kind: ArtifactKind
    hardware_sufficient: bool
    execution_command_preview: str
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, StrategyStatus) else str(self.status)
        d["output_artifact_kind"] = (
            self.output_artifact_kind.value
            if isinstance(self.output_artifact_kind, ArtifactKind)
            else str(self.output_artifact_kind)
        )
        return d


class TrainingStrategy(ABC):
    """Abstract base class for all SAM-AI training and adaptation strategies."""

    strategy_id: str
    name: str
    description: str
    status: StrategyStatus
    supported_architectures: List[str]
    hardware_req: HardwareRequirement
    output_artifact_kind: ArtifactKind

    def check_compatibility(
        self,
        model_manifest: ModelArtifactManifest,
        dataset_manifest: DatasetManifest,
        available_vram_gb: float = 0.0,
        has_cuda: bool = False
    ) -> StrategyCompatibilityResult:
        """Evaluates model, dataset, and hardware compatibility."""
        reasons: List[str] = []

        # 1. Strategy Support Status
        if self.status == StrategyStatus.UNSUPPORTED:
            reasons.append(f"Strategy '{self.strategy_id}' is marked UNSUPPORTED for this architecture or environment.")

        # 2. Model Architecture Support
        if self.supported_architectures and model_manifest.architecture != "UNKNOWN":
            if not any(arch.lower() in model_manifest.architecture.lower() for arch in self.supported_architectures):
                reasons.append(
                    f"Model architecture '{model_manifest.architecture}' not in supported architectures: {self.supported_architectures}"
                )

        # 3. Model & Dataset License Validation
        is_model_valid, model_reasons = model_manifest.validate_for_training()
        if not is_model_valid:
            reasons.extend(model_reasons)

        if dataset_manifest.license in ("UNKNOWN", "", None):
            reasons.append("Dataset license is UNKNOWN. Cannot proceed with training.")

        # 4. Hardware Checks
        hardware_sufficient = True
        hw_notes = {
            "available_vram_gb": available_vram_gb,
            "has_cuda": has_cuda,
            "min_vram_gb": self.hardware_req.min_vram_gb,
            "requires_cuda": self.hardware_req.requires_cuda,
        }

        if self.hardware_req.requires_cuda and not has_cuda:
            hardware_sufficient = False
            reasons.append("Strategy requires CUDA GPU acceleration, but CUDA is not available.")

        if has_cuda and available_vram_gb < self.hardware_req.min_vram_gb:
            hardware_sufficient = False
            reasons.append(
                f"Available VRAM ({available_vram_gb:.1f} GB) is below minimum requirement ({self.hardware_req.min_vram_gb:.1f} GB)."
            )

        is_compatible = len(reasons) == 0 and hardware_sufficient
        return StrategyCompatibilityResult(
            compatible=is_compatible,
            reasons=reasons,
            hardware_sufficient=hardware_sufficient,
            hardware_notes=hw_notes
        )

    def plan_dry_run(
        self,
        model_manifest: ModelArtifactManifest,
        dataset_manifest: DatasetManifest,
        config: Dict[str, Any],
        available_vram_gb: float = 0.0,
        has_cuda: bool = False
    ) -> DryRunPlan:
        """Generates a deterministic dry run plan without running training."""
        compat = self.check_compatibility(model_manifest, dataset_manifest, available_vram_gb, has_cuda)
        
        batch_size = config.get("per_device_batch_size", 1)
        grad_accum = config.get("gradient_accumulation_steps", 4)
        effective_batch_size = batch_size * grad_accum
        
        total_examples = sum(s.num_examples for s in dataset_manifest.splits.values()) if dataset_manifest.splits else 100
        epochs = config.get("epochs", 1)
        steps_planned = max(1, (total_examples * epochs) // effective_batch_size)

        command_preview = (
            f"python -m sam_ai.training.runner "
            f"--strategy {self.strategy_id} "
            f"--model_id {model_manifest.model_id} "
            f"--dataset_id {dataset_manifest.dataset_id} "
            f"--batch_size {batch_size} "
            f"--epochs {epochs}"
        )

        return DryRunPlan(
            strategy_id=self.strategy_id,
            strategy_name=self.name,
            status=self.status,
            base_model_id=model_manifest.model_id,
            dataset_id=dataset_manifest.dataset_id,
            estimated_vram_gb=self.hardware_req.min_vram_gb,
            steps_planned=steps_planned,
            effective_batch_size=effective_batch_size,
            output_artifact_kind=self.output_artifact_kind,
            hardware_sufficient=compat.hardware_sufficient,
            execution_command_preview=command_preview,
            blockers=compat.reasons
        )


# =====================================================================
# Concrete Strategy Implementations (Honest Status Declarations)
# =====================================================================

class SupervisedFineTuningStrategy(TrainingStrategy):
    """Full parameter supervised fine-tuning."""
    strategy_id = "sft_full"
    name = "Full Supervised Fine-Tuning"
    description = "Full weight fine-tuning across all transformer layers."
    status = StrategyStatus.SCAFFOLDED
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=80.0, recommended_vram_gb=160.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.BASE_MODEL


class LoRAStrategy(TrainingStrategy):
    """PEFT Parameter-Efficient Fine-Tuning via LoRA / QLoRA."""
    strategy_id = "peft_lora"
    name = "PEFT LoRA / QLoRA"
    description = "Low-Rank Adaptation targeting attention projection layers (q, k, v, o, gate, up, down)."
    status = StrategyStatus.IMPLEMENTED_NOT_RUN
    supported_architectures = ["qwen", "llama", "deepseek"]
    # 27B 4-bit weights = 13.5GB + activations/optimizer/CUDA context = 18.0GB minimum. 16GB GPUs will OOM.
    hardware_req = HardwareRequirement(min_vram_gb=18.0, recommended_vram_gb=24.0, requires_cuda=True, supports_cpu_offload=True)
    output_artifact_kind = ArtifactKind.ADAPTER


class ResponseDistillationStrategy(TrainingStrategy):
    """Offline distillation of student model from high-capacity teacher responses."""
    strategy_id = "distillation_response"
    name = "Teacher Response Distillation"
    description = "Supervised adaptation of student model using curated, verified teacher chain-of-thought solutions."
    status = StrategyStatus.SCAFFOLDED
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=16.0, recommended_vram_gb=24.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.ADAPTER


class LogitDistillationStrategy(TrainingStrategy):
    """Logit-level Knowledge Distillation via cross-entropy / KL divergence."""
    strategy_id = "distillation_logit"
    name = "Logit-Level KD (KL Divergence)"
    description = "Concurrent student-teacher forward pass comparing output logit distributions."
    status = StrategyStatus.UNSUPPORTED
    supported_architectures = []
    hardware_req = HardwareRequirement(min_vram_gb=160.0, recommended_vram_gb=320.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.STUDENT_CANDIDATE


class DirectPreferenceOptimizationStrategy(TrainingStrategy):
    """Direct Preference Optimization (DPO) for aligning reasoning models."""
    strategy_id = "dpo_preference"
    name = "Direct Preference Optimization (DPO)"
    description = "Reference-model anchored preference optimization on pairwise chosen/rejected examples."
    status = StrategyStatus.SCAFFOLDED
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=24.0, recommended_vram_gb=48.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.ADAPTER


class GRPORLVRStrategy(TrainingStrategy):
    """Group Relative Policy Optimization with Verifiable Rewards."""
    strategy_id = "grpo_rlvr"
    name = "Group Relative Policy Optimization (GRPO / RLVR)"
    description = "Actor-only RL with group-normalized verifiable reward signals (math, format, test-suite)."
    status = StrategyStatus.IMPLEMENTED_NOT_RUN
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=24.0, recommended_vram_gb=48.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.ADAPTER


class ContinuedPretrainingStrategy(TrainingStrategy):
    """Continued domain pretraining on raw causal language modeling."""
    strategy_id = "cpt_domain"
    name = "Continued Domain Pretraining"
    description = "Unsupervised causal LM on domain corpus (code, math, ARC specifications)."
    status = StrategyStatus.PLANNED
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=48.0, recommended_vram_gb=80.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.BASE_MODEL


class AdapterMergingStrategy(TrainingStrategy):
    """Merging multiple LoRA adapters into base weights or unified adapter."""
    strategy_id = "merge_adapters"
    name = "LoRA Adapter Merging (Ties / DARE / Linear)"
    description = "Deterministic weight merging across specialized LoRA checkpoints."
    status = StrategyStatus.SCAFFOLDED
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=0.0, recommended_vram_gb=16.0, requires_cuda=False, supports_cpu_offload=True)
    output_artifact_kind = ArtifactKind.MERGED_MODEL


class MoEExpertTransferStrategy(TrainingStrategy):
    """Dense to Mixture-of-Experts upcycling and expert routing."""
    strategy_id = "moe_expert_transfer"
    name = "Dense-to-MoE Expert Upcycling"
    description = "Converting dense weights into multi-expert MoE architecture with auxiliary-free routing."
    status = StrategyStatus.UNSUPPORTED
    supported_architectures = []
    hardware_req = HardwareRequirement(min_vram_gb=320.0, recommended_vram_gb=640.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.MERGED_MODEL


class ContinualImprovementLoopStrategy(TrainingStrategy):
    """Evaluation-failure harvesting into targeted self-correction refinement."""
    strategy_id = "continual_loop"
    name = "Continual Improvement & Self-Correction Loop"
    description = "Harvesting benchmark failure traces, generating verified corrections, and updating LoRA."
    status = StrategyStatus.SCAFFOLDED
    supported_architectures = ["qwen", "llama", "deepseek"]
    hardware_req = HardwareRequirement(min_vram_gb=24.0, recommended_vram_gb=48.0, requires_cuda=True)
    output_artifact_kind = ArtifactKind.ADAPTER


# =====================================================================
# Strategy Registry
# =====================================================================

class StrategyRegistry:
    """Registry maintaining all available SAM-AI adaptation and training strategies."""

    def __init__(self):
        self._strategies: Dict[str, TrainingStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            SupervisedFineTuningStrategy(),
            LoRAStrategy(),
            ResponseDistillationStrategy(),
            LogitDistillationStrategy(),
            DirectPreferenceOptimizationStrategy(),
            GRPORLVRStrategy(),
            ContinuedPretrainingStrategy(),
            AdapterMergingStrategy(),
            MoEExpertTransferStrategy(),
            ContinualImprovementLoopStrategy(),
        ]
        for s in defaults:
            self._strategies[s.strategy_id] = s

    def register(self, strategy: TrainingStrategy) -> None:
        """Registers a custom strategy."""
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> Optional[TrainingStrategy]:
        """Retrieves a strategy by its ID."""
        return self._strategies.get(strategy_id)

    def list_strategies(
        self,
        status: Optional[StrategyStatus] = None
    ) -> List[TrainingStrategy]:
        """Lists registered strategies, optionally filtered by status."""
        if status is None:
            return list(self._strategies.values())
        return [s for s in self._strategies.values() if s.status == status]

"""
SAM-AI Intelligence Foundation: Experiment Manifest & Dry-Run Planner
====================================================================
Tracks full lineage, deterministic hyperparameters, environment specs,
and strict state machines for reproducible model training experiments.
Includes an offline DryRunPlanner that audits feasibility without
executing training or touching remote resources.

Operational Invariant:
- Experiments must record git commit, code cleanliness, and environment specs.
- Terminal states are explicit: an un-run experiment cannot claim completion.
- DryRunPlanner executes entirely offline with zero network and zero weight I/O.
"""

from __future__ import annotations
import os
import sys
import json
import platform
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple

from sam_ai.foundation.artifacts import (
    ModelArtifactManifest,
    ArtifactKind,
    ValidationStatus
)
from sam_ai.foundation.datasets import (
    DatasetManifest,
    validate_dataset_manifest
)
from sam_ai.foundation.strategies import (
    StrategyRegistry,
    TrainingStrategy,
    StrategyStatus,
    DryRunPlan
)


class TerminalState(str, Enum):
    PLANNED = "PLANNED"
    VALIDATED = "VALIDATED"
    DRY_RUN_PASSED = "DRY_RUN_PASSED"
    BLOCKED_NO_GPU = "BLOCKED_NO_GPU"
    BLOCKED_MISSING_DATA = "BLOCKED_MISSING_DATA"
    BLOCKED_INCOMPATIBLE_MODEL = "BLOCKED_INCOMPATIBLE_MODEL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    COMPLETED_UNEVALUATED = "COMPLETED_UNEVALUATED"
    EVALUATED_REJECTED = "EVALUATED_REJECTED"
    EVALUATED_ACCEPTED_PENDING_APPROVAL = "EVALUATED_ACCEPTED_PENDING_APPROVAL"
    PROMOTED_MANUALLY = "PROMOTED_MANUALLY"


# Allowed state transitions
VALID_TRANSITIONS: Dict[TerminalState, List[TerminalState]] = {
    TerminalState.PLANNED: [
        TerminalState.VALIDATED,
        TerminalState.DRY_RUN_PASSED,
        TerminalState.BLOCKED_NO_GPU,
        TerminalState.BLOCKED_MISSING_DATA,
        TerminalState.BLOCKED_INCOMPATIBLE_MODEL,
        TerminalState.CANCELLED,
        TerminalState.FAILED
    ],
    TerminalState.VALIDATED: [
        TerminalState.DRY_RUN_PASSED,
        TerminalState.COMPLETED_UNEVALUATED,
        TerminalState.BLOCKED_NO_GPU,
        TerminalState.BLOCKED_MISSING_DATA,
        TerminalState.BLOCKED_INCOMPATIBLE_MODEL,
        TerminalState.CANCELLED,
        TerminalState.FAILED
    ],
    TerminalState.BLOCKED_NO_GPU: [
        TerminalState.PLANNED,
        TerminalState.VALIDATED,
        TerminalState.DRY_RUN_PASSED,
        TerminalState.CANCELLED
    ],
    TerminalState.BLOCKED_MISSING_DATA: [
        TerminalState.PLANNED,
        TerminalState.VALIDATED,
        TerminalState.DRY_RUN_PASSED,
        TerminalState.CANCELLED
    ],
    TerminalState.BLOCKED_INCOMPATIBLE_MODEL: [
        TerminalState.PLANNED,
        TerminalState.VALIDATED,
        TerminalState.DRY_RUN_PASSED,
        TerminalState.CANCELLED
    ],
    TerminalState.DRY_RUN_PASSED: [
        TerminalState.COMPLETED_UNEVALUATED,
        TerminalState.FAILED,
        TerminalState.CANCELLED,
        TerminalState.BLOCKED_NO_GPU
    ],
    TerminalState.COMPLETED_UNEVALUATED: [
        TerminalState.EVALUATED_REJECTED,
        TerminalState.EVALUATED_ACCEPTED_PENDING_APPROVAL,
        TerminalState.PROMOTED_MANUALLY,
        TerminalState.FAILED
    ],
    TerminalState.EVALUATED_REJECTED: [
        TerminalState.PLANNED  # Allows restarting/refining with new manifest
    ],
    TerminalState.EVALUATED_ACCEPTED_PENDING_APPROVAL: [
        TerminalState.PROMOTED_MANUALLY,
        TerminalState.CANCELLED,
        TerminalState.EVALUATED_REJECTED
    ],
    TerminalState.PROMOTED_MANUALLY: [],  # Final state for an experiment run
    TerminalState.FAILED: [TerminalState.PLANNED],
    TerminalState.CANCELLED: [TerminalState.PLANNED],
}


@dataclass
class StateTransitionRecord:
    from_state: str
    to_state: str
    timestamp: str
    reason: str
    operator: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateTransitionRecord:
        return cls(**data)


@dataclass
class ExperimentManifest:
    """
    Complete reproducible manifest for a training or adaptation experiment.
    """
    experiment_id: str
    strategy_id: str
    base_model_artifact_id: str
    dataset_manifest_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    git_commit: str = "UNKNOWN"
    code_state_clean: bool = False
    environment_specs: Dict[str, str] = field(default_factory=dict)
    current_state: TerminalState = TerminalState.PLANNED
    state_history: List[StateTransitionRecord] = field(default_factory=list)
    dry_run_plan: Optional[Dict[str, Any]] = None
    output_artifact_id: Optional[str] = None
    eval_report_id: Optional[str] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(
        self,
        new_state: TerminalState,
        reason: str,
        operator: str = "system"
    ) -> None:
        """Transitions state machine if valid, appending to audit history."""
        allowed = VALID_TRANSITIONS.get(self.current_state, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition from '{self.current_state.value}' to '{new_state.value}'. "
                f"Allowed target states: {[s.value for s in allowed]}"
            )
        
        record = StateTransitionRecord(
            from_state=self.current_state.value,
            to_state=new_state.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            operator=operator
        )
        self.state_history.append(record)
        self.current_state = new_state

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["current_state"] = (
            self.current_state.value
            if isinstance(self.current_state, TerminalState)
            else str(self.current_state)
        )
        d["state_history"] = [
            r.to_dict() if isinstance(r, StateTransitionRecord) else r
            for r in self.state_history
        ]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentManifest:
        payload = dict(data)
        if "current_state" in payload and isinstance(payload["current_state"], str):
            payload["current_state"] = TerminalState(payload["current_state"])
        if "state_history" in payload and isinstance(payload["state_history"], list):
            payload["state_history"] = [
                StateTransitionRecord.from_dict(r) if isinstance(r, dict) else r
                for r in payload["state_history"]
            ]
        return cls(**payload)


class DryRunPlanner:
    """
    Offline Dry-Run Planner.
    Performs full pre-flight compatibility, licensing, and resource audits
    without downloading weights, allocating CUDA tensors, or calling APIs.
    """

    def __init__(self, strategy_registry: Optional[StrategyRegistry] = None):
        self.registry = strategy_registry or StrategyRegistry()

    @staticmethod
    def inspect_system_hardware() -> Tuple[float, bool, Dict[str, Any]]:
        """
        Detects local hardware availability safely without throwing errors.
        Returns: (available_vram_gb, has_cuda, info_dict)
        """
        has_cuda = False
        available_vram_gb = 0.0
        details: Dict[str, Any] = {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count() or 1,
            "cuda_available": False,
            "gpu_count": 0,
            "device_names": [],
        }

        try:
            import torch
            has_cuda = torch.cuda.is_available()
            details["cuda_available"] = has_cuda
            if has_cuda:
                count = torch.cuda.device_count()
                details["gpu_count"] = count
                per_device = []
                single_max = 0.0
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    details["device_names"].append(props.name)
                    vram_gb = props.total_memory / (1024 ** 3)
                    per_device.append(round(vram_gb, 2))
                    available_vram_gb += vram_gb
                    if vram_gb > single_max:
                        single_max = vram_gb
                details["per_device_vram_gb"] = per_device
                details["single_max_vram_gb"] = round(single_max, 2)
                details["total_aggregate_vram_gb"] = round(available_vram_gb, 2)
                details["supports_multi_gpu_sharding"] = False  # Current trainer is single-device
                # Default effective VRAM for non-sharded training is single_max, not aggregate
                available_vram_gb = single_max
        except ImportError:
            details["notes"] = "torch not installed or unavailable"
        except Exception as e:
            details["notes"] = f"Error inspecting CUDA: {e}"

        return available_vram_gb, has_cuda, details

    def plan(
        self,
        experiment: ExperimentManifest,
        model_manifest: ModelArtifactManifest,
        dataset_manifest: DatasetManifest,
        base_dir: Optional[Union[str, Path]] = None,
        force_hardware_vram_gb: Optional[float] = None,
        force_has_cuda: Optional[bool] = None
    ) -> DryRunPlan:
        """
        Audits experiment parameters and manifests, updates experiment state,
        and returns a deterministic DryRunPlan.
        """
        # 1. Resolve Hardware
        detected_vram, detected_cuda, hw_details = self.inspect_system_hardware()
        vram_gb = force_hardware_vram_gb if force_hardware_vram_gb is not None else detected_vram
        has_cuda = force_has_cuda if force_has_cuda is not None else detected_cuda

        # 2. Strategy Lookup
        strategy = self.registry.get(experiment.strategy_id)
        if not strategy:
            reason = f"Strategy '{experiment.strategy_id}' not found in registry."
            experiment.transition_to(TerminalState.FAILED, reason)
            raise ValueError(reason)

        # 3. Model Manifest Integrity
        is_model_valid, model_reasons = model_manifest.validate_for_training()
        if not is_model_valid:
            reason = f"Model validation failed: {'; '.join(model_reasons)}"
            experiment.transition_to(TerminalState.BLOCKED_INCOMPATIBLE_MODEL, reason)
            return strategy.plan_dry_run(
                model_manifest, dataset_manifest, experiment.hyperparameters, vram_gb, has_cuda
            )

        # 4. Dataset Manifest Integrity
        is_dataset_valid, dataset_reasons = validate_dataset_manifest(dataset_manifest, base_dir=base_dir)
        if not is_dataset_valid:
            reason = f"Dataset validation failed: {'; '.join(dataset_reasons)}"
            experiment.transition_to(TerminalState.BLOCKED_MISSING_DATA, reason)
            return strategy.plan_dry_run(
                model_manifest, dataset_manifest, experiment.hyperparameters, vram_gb, has_cuda
            )

        # 5. Plan & Strategy Compatibility Check
        dry_plan = strategy.plan_dry_run(
            model_manifest, dataset_manifest, experiment.hyperparameters, vram_gb, has_cuda
        )
        experiment.dry_run_plan = dry_plan.to_dict()

        if not dry_plan.hardware_sufficient:
            reasons_str = "; ".join(dry_plan.blockers) or "Insufficient hardware"
            experiment.transition_to(TerminalState.BLOCKED_NO_GPU, f"Hardware blocked: {reasons_str}")
        elif dry_plan.status == StrategyStatus.UNSUPPORTED:
            experiment.transition_to(
                TerminalState.BLOCKED_INCOMPATIBLE_MODEL,
                f"Strategy '{strategy.strategy_id}' is marked UNSUPPORTED."
            )
        else:
            experiment.transition_to(
                TerminalState.DRY_RUN_PASSED,
                f"Dry run passed offline verification. Steps planned: {dry_plan.steps_planned}"
            )

        return dry_plan


class ExperimentRegistry:
    """
    Offline local filesystem registry for ExperimentManifest records.
    Persists deterministic JSON records under evidence/ or local storage.
    """

    def __init__(self, registry_dir: Union[str, Path]):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, ExperimentManifest] = {}
        self.reload()

    def reload(self) -> None:
        """Reloads all experiment manifests from disk."""
        self._experiments.clear()
        for manifest_file in self.registry_dir.glob("*.experiment.json"):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    manifest = ExperimentManifest.from_dict(data)
                    self._experiments[manifest.experiment_id] = manifest
            except Exception as e:
                print(f"[Warning] Failed to load experiment from {manifest_file}: {e}")

    def register(self, manifest: ExperimentManifest, overwrite: bool = False) -> Path:
        """Registers and persists an experiment manifest."""
        if manifest.experiment_id in self._experiments and not overwrite:
            raise ValueError(f"Experiment ID '{manifest.experiment_id}' already registered.")
        
        self._experiments[manifest.experiment_id] = manifest
        file_path = self.registry_dir / f"{manifest.experiment_id}.experiment.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        return file_path

    def get(self, experiment_id: str) -> Optional[ExperimentManifest]:
        """Retrieves experiment manifest by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        state: Optional[TerminalState] = None
    ) -> List[ExperimentManifest]:
        """Lists registered experiments, optionally filtered by terminal state."""
        if state is None:
            return list(self._experiments.values())
        return [e for e in self._experiments.values() if e.current_state == state]

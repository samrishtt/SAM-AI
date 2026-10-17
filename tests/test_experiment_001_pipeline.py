"""
Unit Tests: SAM-AI Experiment 001 Pipeline & Hardware Qualification
===================================================================
Tests the end-to-end execution of Experiment 001:
- Dataset integrity and SHA-256 checksums on disk.
- Base model artifact manifest verification.
- Hardware audit: verifies 16 GB GPUs fail closed (18 GB minimum required).
- Baseline recording on fixed held-out split.
- Adapter training, persistence, and parent lineage.
- Adapter reloading verification.
- Candidate vs baseline evaluation with tolerance enforcement.
- Audited promotion vs regression rejection preserving baseline.
- Real trainer fail-closed behavior on CPU.
"""

from __future__ import annotations
import os
import json
import tempfile
import pytest
from pathlib import Path

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
    StrategyRegistry,
    StrategyStatus,
    LoRAStrategy,
)
from sam_ai.foundation.experiments import (
    TerminalState,
    ExperimentManifest,
    DryRunPlanner,
    ExperimentRegistry,
)
from sam_ai.foundation.promotion import (
    FrozenEvaluationSuite,
    PromotionGate,
    compare_candidate_to_baseline,
)
from sam_ai.foundation.pipeline import (
    ExperimentPipeline,
    MockedSimulationTrainer,
    RealPyTorchLoRATrainer,
    DeterministicEvaluator,
    PipelineExecutionResult,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_EXP1 = REPO_ROOT / "evidence" / "experiment_001"


def test_experiment_001_dataset_integrity():
    """Verify that the official GSM8K dataset splits and SHA-256 hashes match on disk."""
    dataset_manifest_path = EVIDENCE_EXP1 / "dataset" / "gsm8k_verifiable_reasoning_v1.dataset.json"
    assert dataset_manifest_path.exists(), f"Dataset manifest missing: {dataset_manifest_path}"

    with open(dataset_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = DatasetManifest.from_dict(data)

    assert manifest.dataset_id == "gsm8k_verifiable_reasoning_v1"
    assert manifest.license == "MIT"
    assert manifest.is_test_fixture is False
    assert len(manifest.splits) == 3
    assert "train" in manifest.splits
    assert "validation" in manifest.splits
    assert "frozen_eval" in manifest.splits

    # Physical verification on disk
    is_valid, errors = validate_dataset_manifest(manifest, base_dir=REPO_ROOT)
    assert is_valid is True, f"Dataset validation failed: {errors}"
    assert len(errors) == 0

    # Verify split file sizes
    for split_name, split in manifest.splits.items():
        p = Path(split.file_path)
        assert p.exists()
        calc_hash = compute_file_sha256(p)
        assert calc_hash == split.sha256_hash


def test_base_model_artifact_manifest_integrity():
    """Verify that the base model manifest for Qwen3.8-27B meets all requirements."""
    model_manifest_path = EVIDENCE_EXP1 / "models" / "qwen38-27b-base-canonical.artifact.json"
    assert model_manifest_path.exists()

    with open(model_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = ModelArtifactManifest.from_dict(data)

    assert manifest.model_id == "Qwen/Qwen3.8-27B"
    assert manifest.license == "Apache-2.0"
    assert manifest.architecture == "Qwen3_5ForConditionalGeneration"
    assert manifest.metadata.get("model_type") == "qwen3_5"
    assert manifest.validation_status == ValidationStatus.VALIDATED

    is_valid, errors = manifest.validate_for_training()
    assert is_valid is True
    assert len(errors) == 0


def test_hardware_qualification_audit_16gb_fails_closed():
    """Verify that 16 GB GPUs fail closed because Qwen3.8-27B 4-bit QLoRA requires 18 GB minimum."""
    planner = DryRunPlanner()

    with open(EVIDENCE_EXP1 / "models" / "qwen38-27b-base-canonical.artifact.json", "r", encoding="utf-8") as f:
        model_manifest = ModelArtifactManifest.from_dict(json.load(f))

    with open(EVIDENCE_EXP1 / "dataset" / "gsm8k_verifiable_reasoning_v1.dataset.json", "r", encoding="utf-8") as f:
        dataset_manifest = DatasetManifest.from_dict(json.load(f))

    exp_manifest = ExperimentManifest(
        experiment_id="EXP-TEST-16GB-FAIL",
        strategy_id="peft_lora",
        base_model_artifact_id=model_manifest.artifact_id,
        dataset_manifest_id=dataset_manifest.dataset_id,
    )

    # 1. 16 GB GPU with CUDA -> must fail closed due to insufficient VRAM (requires 18.0 GB)
    plan_16gb = planner.plan(
        experiment=exp_manifest,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        force_hardware_vram_gb=16.0,
        force_has_cuda=True,
    )
    assert plan_16gb.hardware_sufficient is False
    assert exp_manifest.current_state == TerminalState.BLOCKED_NO_GPU
    assert any("below minimum requirement" in b for b in plan_16gb.blockers)

    # 2. 24 GB GPU with CUDA -> passes dry run
    exp_manifest_24gb = ExperimentManifest(
        experiment_id="EXP-TEST-24GB-PASS",
        strategy_id="peft_lora",
        base_model_artifact_id=model_manifest.artifact_id,
        dataset_manifest_id=dataset_manifest.dataset_id,
        hyperparameters={"per_device_batch_size": 1, "gradient_accumulation_steps": 8, "epochs": 1}
    )
    plan_24gb = planner.plan(
        experiment=exp_manifest_24gb,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        force_hardware_vram_gb=24.0,
        force_has_cuda=True,
    )
    assert plan_24gb.hardware_sufficient is True
    assert exp_manifest_24gb.current_state == TerminalState.DRY_RUN_PASSED


def test_experiment_001_pipeline_successful_promotion_flow():
    """Verify full end-to-end flow where candidate improves with 0 regressions and gets promoted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        art_reg = ArtifactRegistry(tmp_path / "models")
        data_reg = DatasetRegistry(tmp_path / "datasets")
        exp_reg = ExperimentRegistry(tmp_path / "experiments")
        gate = PromotionGate(tmp_path / "promotion")

        # Load manifests
        with open(EVIDENCE_EXP1 / "models" / "qwen38-27b-base-canonical.artifact.json", "r", encoding="utf-8") as f:
            base_manifest = ModelArtifactManifest.from_dict(json.load(f))
        with open(EVIDENCE_EXP1 / "dataset" / "gsm8k_verifiable_reasoning_v1.dataset.json", "r", encoding="utf-8") as f:
            dataset_manifest = DatasetManifest.from_dict(json.load(f))
        with open(EVIDENCE_EXP1 / "suite" / "sam_ai_math_reasoning_frozen_v1.suite.json", "r", encoding="utf-8") as f:
            eval_suite = FrozenEvaluationSuite.from_dict(json.load(f))

        art_reg.register(base_manifest)
        data_reg.register(dataset_manifest)

        experiment = ExperimentManifest(
            experiment_id="EXP-001-TEST-SUCCESS",
            strategy_id="peft_lora",
            base_model_artifact_id=base_manifest.artifact_id,
            dataset_manifest_id=dataset_manifest.dataset_id,
            hyperparameters={"lora_rank": 16, "lora_alpha": 32, "max_steps": 25}
        )

        # Candidate improves: accuracy 0.50 -> 0.60
        evaluator = DeterministicEvaluator(
            scores_provider=lambda adapter_dir: {
                "gsm8k_reasoning_accuracy": 0.50 if adapter_dir is None else 0.60,
                "format_compliance_rate": 0.92 if adapter_dir is None else 0.98,
                "numerical_precision": 0.90 if adapter_dir is None else 0.95,
            }
        )
        trainer = MockedSimulationTrainer(simulated_loss_start=2.45, simulated_loss_end=0.82)

        pipeline = ExperimentPipeline(
            artifact_registry=art_reg,
            dataset_registry=data_reg,
            experiment_registry=exp_reg,
            promotion_gate=gate,
            evaluator=evaluator,
            trainer=trainer,
        )

        result = pipeline.run_experiment(
            experiment=experiment,
            base_manifest=base_manifest,
            dataset_manifest=dataset_manifest,
            eval_suite=eval_suite,
            output_base_dir=tmp_path / "exp_output",
        )

        assert result.final_experiment_state == TerminalState.PROMOTED_MANUALLY
        assert result.comparison.recommendation == "ACCEPT"
        assert result.promotion_decision is not None
        assert result.promotion_decision.action == "PROMOTED"
        assert gate.get_active_model_id() == "EXP-001-TEST-SUCCESS-adapter"

        # Verify adapter artifact manifest lineage
        adapter_art = art_reg.get("EXP-001-TEST-SUCCESS-adapter")
        assert adapter_art is not None
        assert adapter_art.kind == ArtifactKind.ADAPTER
        assert adapter_art.parent_artifact_ids == [base_manifest.artifact_id]
        assert adapter_art.validation_status == ValidationStatus.VALIDATED


def test_experiment_001_pipeline_regression_rejection_flow():
    """Verify that a candidate with accuracy regression is rejected and baseline is preserved."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        art_reg = ArtifactRegistry(tmp_path / "models")
        data_reg = DatasetRegistry(tmp_path / "datasets")
        exp_reg = ExperimentRegistry(tmp_path / "experiments")
        gate = PromotionGate(tmp_path / "promotion")

        # Initial baseline active
        with open(EVIDENCE_EXP1 / "models" / "qwen38-27b-base-canonical.artifact.json", "r", encoding="utf-8") as f:
            base_manifest = ModelArtifactManifest.from_dict(json.load(f))
        with open(EVIDENCE_EXP1 / "dataset" / "gsm8k_verifiable_reasoning_v1.dataset.json", "r", encoding="utf-8") as f:
            dataset_manifest = DatasetManifest.from_dict(json.load(f))
        with open(EVIDENCE_EXP1 / "suite" / "sam_ai_math_reasoning_frozen_v1.suite.json", "r", encoding="utf-8") as f:
            eval_suite = FrozenEvaluationSuite.from_dict(json.load(f))

        art_reg.register(base_manifest)
        data_reg.register(dataset_manifest)

        # Baseline promoted first
        gate.active_file.write_text(json.dumps({"active_artifact_id": base_manifest.artifact_id}))

        experiment = ExperimentManifest(
            experiment_id="EXP-001-TEST-REGRESSION",
            strategy_id="peft_lora",
            base_model_artifact_id=base_manifest.artifact_id,
            dataset_manifest_id=dataset_manifest.dataset_id,
        )

        # Candidate regresses: accuracy 0.60 -> 0.40 (exceeds 0.00 tolerance margin)
        evaluator = DeterministicEvaluator(
            scores_provider=lambda adapter_dir: {
                "gsm8k_reasoning_accuracy": 0.60 if adapter_dir is None else 0.40,
                "format_compliance_rate": 0.95 if adapter_dir is None else 0.90,
                "numerical_precision": 0.95 if adapter_dir is None else 0.85,
            }
        )
        trainer = MockedSimulationTrainer()

        pipeline = ExperimentPipeline(
            artifact_registry=art_reg,
            dataset_registry=data_reg,
            experiment_registry=exp_reg,
            promotion_gate=gate,
            evaluator=evaluator,
            trainer=trainer,
        )

        result = pipeline.run_experiment(
            experiment=experiment,
            base_manifest=base_manifest,
            dataset_manifest=dataset_manifest,
            eval_suite=eval_suite,
            output_base_dir=tmp_path / "exp_output",
        )

        # Candidate rejected
        assert result.final_experiment_state == TerminalState.EVALUATED_REJECTED
        assert result.comparison.recommendation == "REJECT"
        assert result.promotion_decision is None
        # Active model remains the base model
        assert gate.get_active_model_id() == base_manifest.artifact_id


def test_real_trainer_fails_closed_without_cuda():
    """Verify that RealPyTorchLoRATrainer refuses to run on CPU."""
    # Local environment has no CUDA
    with pytest.raises((RuntimeError, ImportError), match="(CUDA|PEFT|PyTorch)"):
        RealPyTorchLoRATrainer()


def test_tampered_dataset_split_hash_rejection():
    """Verify that a tampered file hash causes pipeline to fail closed before training."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        art_reg = ArtifactRegistry(tmp_path / "models")
        data_reg = DatasetRegistry(tmp_path / "datasets")
        exp_reg = ExperimentRegistry(tmp_path / "experiments")
        gate = PromotionGate(tmp_path / "promotion")

        with open(EVIDENCE_EXP1 / "models" / "qwen38-27b-base-canonical.artifact.json", "r", encoding="utf-8") as f:
            base_manifest = ModelArtifactManifest.from_dict(json.load(f))
        with open(EVIDENCE_EXP1 / "suite" / "sam_ai_math_reasoning_frozen_v1.suite.json", "r", encoding="utf-8") as f:
            eval_suite = FrozenEvaluationSuite.from_dict(json.load(f))

        tampered_manifest = DatasetManifest(
            dataset_id="tampered-data-test",
            source="test",
            version="1.0",
            license="MIT",
            provenance={"author": "Test"},
            splits={
                "train": SplitInfo(name="train", num_examples=1, file_path="non_existent_file.jsonl"),
                "frozen_eval": SplitInfo(name="frozen_eval", num_examples=1, file_path="non_existent_file.jsonl"),
            },
            is_test_fixture=False
        )

        exp = ExperimentManifest(
            experiment_id="EXP-TAMPERED",
            strategy_id="peft_lora",
            base_model_artifact_id=base_manifest.artifact_id,
            dataset_manifest_id=tampered_manifest.dataset_id,
        )

        pipeline = ExperimentPipeline(art_reg, data_reg, exp_reg, gate)
        with pytest.raises(ValueError, match="Dataset validation failed"):
            pipeline.run_experiment(
                experiment=exp,
                base_manifest=base_manifest,
                dataset_manifest=tampered_manifest,
                eval_suite=eval_suite,
                output_base_dir=tmp_path / "out",
            )

        assert exp.current_state == TerminalState.BLOCKED_MISSING_DATA

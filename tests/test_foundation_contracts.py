"""
Unit Tests: SAM-AI Intelligence Foundation Contracts & Offline Planning
========================================================================
Validates manifests, provenance verification, strategy registry honesty,
deterministic dry-run planning, frozen evaluation gates, and audited rollback.
"""

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
    GRPORLVRStrategy,
    LogitDistillationStrategy,
    MoEExpertTransferStrategy,
    SupervisedFineTuningStrategy,
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


def test_artifact_manifest_validation():
    """Verify artifact manifest validation fails closed on missing license or architecture."""
    # 1. Valid Manifest
    valid_manifest = ModelArtifactManifest(
        artifact_id="qwen38-27b-base-01",
        kind=ArtifactKind.BASE_MODEL,
        model_id="Qwen/Qwen3.8-27B",
        source_path="/kaggle/input/qwen-3-8-27b",
        license="Apache-2.0",
        architecture="Qwen3ForCausalLM",
        precision=PrecisionType.BFLOAT16,
        validation_status=ValidationStatus.VALIDATED,
    )
    is_valid, errors = valid_manifest.validate_for_training()
    assert is_valid is True
    assert len(errors) == 0

    # 2. Unknown License Fails Closed
    bad_license = ModelArtifactManifest(
        artifact_id="unknown-lic-01",
        kind=ArtifactKind.BASE_MODEL,
        model_id="Custom/Model",
        source_path="/path/to/weights",
        license="UNKNOWN",
        architecture="Qwen3ForCausalLM",
    )
    is_valid, errors = bad_license.validate_for_training()
    assert is_valid is False
    assert any("license is UNKNOWN" in e for e in errors)

    # 3. Unknown Architecture Fails Closed
    bad_arch = ModelArtifactManifest(
        artifact_id="unknown-arch-01",
        kind=ArtifactKind.BASE_MODEL,
        model_id="Custom/Model",
        source_path="/path/to/weights",
        license="Apache-2.0",
        architecture="UNKNOWN",
    )
    is_valid, errors = bad_arch.validate_for_training()
    assert is_valid is False
    assert any("architecture is UNKNOWN" in e for e in errors)

    # 4. Adapter without parent fails closed
    bad_adapter = ModelArtifactManifest(
        artifact_id="orphan-adapter-01",
        kind=ArtifactKind.ADAPTER,
        model_id="SAM-AI/qwen38-lora-pilot",
        source_path="/adapters/lora",
        license="Apache-2.0",
        architecture="Qwen3ForCausalLM",
        parent_artifact_ids=[],
    )
    is_valid, errors = bad_adapter.validate_for_training()
    assert is_valid is False
    assert any("must explicitly declare its parent" in e for e in errors)


def test_artifact_registry_roundtrip():
    """Verify registry serialization and listing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = ArtifactRegistry(tmp_dir)
        manifest = ModelArtifactManifest(
            artifact_id="qwen-art-01",
            kind=ArtifactKind.BASE_MODEL,
            model_id="Qwen/Qwen3.8-27B",
            source_path="/weights",
            license="Apache-2.0",
            architecture="Qwen3ForCausalLM",
        )
        registry.register(manifest)

        reloaded = registry.get("qwen-art-01")
        assert reloaded is not None
        assert reloaded.artifact_id == "qwen-art-01"
        assert reloaded.architecture == "Qwen3ForCausalLM"
        assert len(registry.list_artifacts()) == 1


def test_dataset_manifest_and_sha256_verification():
    """Verify dataset provenance, fixture rules, and SHA-256 file verification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = Path(tmp_dir) / "train.jsonl"
        data_file.write_text('{"prompt": "hello", "response": "world"}\n', encoding="utf-8")
        real_hash = compute_file_sha256(data_file)

        # 1. Valid Production Dataset with matching SHA-256
        prod_manifest = DatasetManifest(
            dataset_id="gsm8k-curated-v1",
            source="huggingface:openai/gsm8k",
            version="1.0.0",
            license="MIT",
            provenance={"author": "OpenAI", "source_url": "https://github.com/openai/grade-school-math"},
            splits={
                "train": SplitInfo(
                    name="train",
                    num_examples=1,
                    file_path=str(data_file),
                    sha256_hash=real_hash
                )
            },
            is_test_fixture=False
        )
        is_valid, errors = validate_dataset_manifest(prod_manifest, base_dir=tmp_dir)
        assert is_valid is True
        assert len(errors) == 0

        # 2. Hash Mismatch Fails Closed
        tampered_manifest = DatasetManifest(
            dataset_id="tampered-v1",
            source="huggingface:openai/gsm8k",
            version="1.0.0",
            license="MIT",
            provenance={"author": "OpenAI"},
            splits={
                "train": SplitInfo(
                    name="train",
                    num_examples=1,
                    file_path=str(data_file),
                    sha256_hash="0000000000000000000000000000000000000000000000000000000000000000"
                )
            },
            is_test_fixture=False
        )
        is_valid, errors = validate_dataset_manifest(tampered_manifest, base_dir=tmp_dir)
        assert is_valid is False
        assert any("SHA-256 mismatch" in e for e in errors)

        # 3. Synthetic without generator fails closed
        synth_manifest = DatasetManifest(
            dataset_id="synth-without-gen",
            source="synthetic",
            version="1.0.0",
            license="Apache-2.0",
            is_synthetic=True,
            generator_id=None,
            is_test_fixture=True
        )
        is_valid, errors = validate_dataset_manifest(synth_manifest)
        assert is_valid is False
        assert any("Synthetic dataset must declare generator_id" in e for e in errors)


def test_strategy_registry_completeness_and_honest_statuses():
    """Verify registry contains 10 strategies with honest implementation statuses."""
    registry = StrategyRegistry()
    strategies = registry.list_strategies()
    assert len(strategies) >= 10

    strat_map = {s.strategy_id: s for s in strategies}

    # Verify honest declarations
    assert strat_map["peft_lora"].status == StrategyStatus.IMPLEMENTED_NOT_RUN
    assert strat_map["grpo_rlvr"].status == StrategyStatus.IMPLEMENTED_NOT_RUN
    assert strat_map["sft_full"].status == StrategyStatus.SCAFFOLDED
    assert strat_map["distillation_logit"].status == StrategyStatus.UNSUPPORTED
    assert strat_map["moe_expert_transfer"].status == StrategyStatus.UNSUPPORTED
    assert strat_map["cpt_domain"].status == StrategyStatus.PLANNED


def test_dry_run_planner_offline_execution():
    """Verify DryRunPlanner runs deterministically offline without network or CUDA."""
    planner = DryRunPlanner()

    model_manifest = ModelArtifactManifest(
        artifact_id="qwen-base-01",
        kind=ArtifactKind.BASE_MODEL,
        model_id="Qwen/Qwen3.8-27B",
        source_path="/kaggle/input/qwen-3-8-27b",
        license="Apache-2.0",
        architecture="Qwen3ForCausalLM",
        precision=PrecisionType.BFLOAT16,
    )

    dataset_manifest = DatasetManifest(
        dataset_id="arc-train-v1",
        source="local",
        version="1.0",
        license="Apache-2.0",
        is_test_fixture=True,
        splits={"train": SplitInfo(name="train", num_examples=200)}
    )

    # 1. On CPU-only local system without CUDA -> blocks with BLOCKED_NO_GPU
    exp_manifest = ExperimentManifest(
        experiment_id="exp-lora-test-01",
        strategy_id="peft_lora",
        base_model_artifact_id="qwen-base-01",
        dataset_manifest_id="arc-train-v1",
        hyperparameters={"per_device_batch_size": 2, "epochs": 1}
    )

    plan = planner.plan(
        experiment=exp_manifest,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        force_hardware_vram_gb=0.0,
        force_has_cuda=False
    )
    assert plan.hardware_sufficient is False
    assert exp_manifest.current_state == TerminalState.BLOCKED_NO_GPU
    assert len(exp_manifest.state_history) == 1

    # 2. Under simulated adequate GPU (e.g. 24GB VRAM) -> passes dry run
    exp_manifest_gpu = ExperimentManifest(
        experiment_id="exp-lora-gpu-01",
        strategy_id="peft_lora",
        base_model_artifact_id="qwen-base-01",
        dataset_manifest_id="arc-train-v1",
        hyperparameters={"per_device_batch_size": 2, "gradient_accumulation_steps": 4, "epochs": 1}
    )
    plan_gpu = planner.plan(
        experiment=exp_manifest_gpu,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        force_hardware_vram_gb=24.0,
        force_has_cuda=True
    )
    assert plan_gpu.hardware_sufficient is True
    assert exp_manifest_gpu.current_state == TerminalState.DRY_RUN_PASSED
    assert plan_gpu.steps_planned == 25  # (200 * 1) // (2 * 4) = 25 steps


def test_experiment_state_machine_transitions():
    """Verify state transitions enforce allowed paths and record audit history."""
    exp = ExperimentManifest(
        experiment_id="exp-sm-01",
        strategy_id="peft_lora",
        base_model_artifact_id="base",
        dataset_manifest_id="data"
    )
    assert exp.current_state == TerminalState.PLANNED

    # Valid transition
    exp.transition_to(TerminalState.VALIDATED, "Schema verified")
    assert exp.current_state == TerminalState.VALIDATED
    assert len(exp.state_history) == 1
    assert exp.state_history[0].from_state == "PLANNED"
    assert exp.state_history[0].to_state == "VALIDATED"

    # Invalid transition raises ValueError
    with pytest.raises(ValueError, match="Invalid state transition"):
        exp.transition_to(TerminalState.PROMOTED_MANUALLY, "Illegal skip")


def test_frozen_evaluation_and_regression_detection():
    """Verify FrozenEvaluationSuite enforces tolerances and detects regressions."""
    suite = FrozenEvaluationSuite(
        suite_id="sam-ai-v1-core",
        name="SAM-AI Core Eval Suite",
        version="1.0.0",
        metric_thresholds={"arc_accuracy": 0.30, "reasoning_score": 0.50},
        tolerance_margins={"arc_accuracy": 0.02, "reasoning_score": 0.01}
    )

    baseline_scores = {"arc_accuracy": 0.35, "reasoning_score": 0.60}

    # Case 1: Improvement
    cand_scores_good = {"arc_accuracy": 0.40, "reasoning_score": 0.62}
    comp_good = compare_candidate_to_baseline(baseline_scores, cand_scores_good, suite, "base", "cand1")
    assert comp_good.overall_improved is True
    assert comp_good.recommendation == "ACCEPT"
    assert len(comp_good.critical_regressions) == 0

    # Case 2: Acceptable small regression within tolerance
    cand_scores_slight_drop = {"arc_accuracy": 0.34, "reasoning_score": 0.61}  # -0.01 on arc (tolerance is 0.02)
    comp_slight = compare_candidate_to_baseline(baseline_scores, cand_scores_slight_drop, suite, "base", "cand2")
    assert comp_slight.metrics["arc_accuracy"].status == "REGRESSED_ACCEPTABLE"
    assert comp_slight.recommendation == "ACCEPT"

    # Case 3: Critical regression beyond tolerance -> REJECT
    cand_scores_bad = {"arc_accuracy": 0.31, "reasoning_score": 0.62}  # -0.04 on arc (exceeds 0.02)
    comp_bad = compare_candidate_to_baseline(baseline_scores, cand_scores_bad, suite, "base", "cand3")
    assert comp_bad.recommendation == "REJECT"
    assert len(comp_bad.critical_regressions) == 1


def test_promotion_gate_and_rollback():
    """Verify PromotionGate promotes only validated candidates and executes reversible rollbacks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gate = PromotionGate(tmp_dir)
        assert gate.get_active_model_id() is None

        # Prepare Candidate
        candidate = ModelArtifactManifest(
            artifact_id="sam-ai-v1-checkpoint",
            kind=ArtifactKind.BASE_MODEL,
            model_id="SAM-AI/qwen38-adapted",
            source_path="/models/v1",
            license="Apache-2.0",
            architecture="Qwen3ForCausalLM",
            validation_status=ValidationStatus.VALIDATED
        )

        suite = FrozenEvaluationSuite(
            suite_id="suite-1",
            name="Suite 1",
            version="1.0",
            tolerance_margins={"acc": 0.01}
        )
        comp = compare_candidate_to_baseline({"acc": 0.50}, {"acc": 0.55}, suite, "base-0", candidate.artifact_id)

        # 1. Promote
        decision = gate.promote(
            candidate_manifest=candidate,
            comparison=comp,
            operator="lead_architect",
            rationale="Demonstrated 5% accuracy gain with 0 regressions."
        )
        assert decision.action == "PROMOTED"
        assert gate.get_active_model_id() == "sam-ai-v1-checkpoint"

        # 2. Reject unvalidated candidate
        unverified_cand = ModelArtifactManifest(
            artifact_id="unverified-cand",
            kind=ArtifactKind.BASE_MODEL,
            model_id="SAM-AI/bad",
            source_path="/models/bad",
            license="Apache-2.0",
            architecture="Qwen3ForCausalLM",
            validation_status=ValidationStatus.UNVERIFIED
        )
        with pytest.raises(ValueError, match="Only VALIDATED artifacts may be promoted"):
            gate.promote(unverified_cand, comp, "lead_architect", "Should fail")

        # 3. Rollback
        rollback_decision = gate.rollback(
            target_artifact_id="base-0",
            operator="lead_architect",
            reason="Canary alert triggered on latency."
        )
        assert rollback_decision.action == "ROLLED_BACK"
        assert gate.get_active_model_id() == "base-0"

        history = gate.get_history()
        assert len(history) == 2
        assert history[0].action == "PROMOTED"
        assert history[1].action == "ROLLED_BACK"

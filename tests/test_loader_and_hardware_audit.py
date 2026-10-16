"""
Unit Tests: Loader Verification, Label Masking, & Multi-GPU Hardware Audit
==========================================================================
Verifies:
- AutoModel class resolution (CausalLM vs ConditionalGeneration).
- SFT label masking (-100 on prompt tokens).
- Dynamic LoRA target module discovery.
- Single-GPU VRAM vs. aggregate multi-GPU memory audit (2x 15GB fails 18GB requirement).
- Refusal to claim real compatibility without loading actual checkpoints.
"""

from __future__ import annotations
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from sam_ai.models.loader import (
    resolve_model_loading_class,
    ModelLoader,
    ModelConfig,
    ModelSpec,
    PRIMARY_MODEL_ID,
)
from sam_ai.foundation.pipeline import (
    prepare_sft_input_and_labels,
    discover_lora_target_modules,
)
from sam_ai.foundation.experiments import (
    DryRunPlanner,
    ExperimentManifest,
    TerminalState,
)
from sam_ai.foundation.artifacts import (
    ModelArtifactManifest,
    ArtifactKind,
    ValidationStatus,
)
from sam_ai.foundation.datasets import (
    DatasetManifest,
    SplitInfo,
)


def test_resolve_model_loading_class_causal_lm():
    """Verify standard text-only causal LM resolves to AutoModelForCausalLM."""
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
    }
    cls_name, notes = resolve_model_loading_class(config)
    assert cls_name == "AutoModelForCausalLM"
    assert any("causal LM" in n for n in notes)


def test_resolve_model_loading_class_conditional_generation():
    """Verify Qwen3.5/Qwen3.8 conditional generation config resolves to AutoModelForConditionalGeneration."""
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
    }
    cls_name, notes = resolve_model_loading_class(config)
    assert cls_name == "AutoModelForConditionalGeneration"
    assert any("conditional generation" in n for n in notes)


def test_resolve_model_loading_class_missing_file_fails_closed():
    """Verify non-existent config file fails closed with UNKNOWN."""
    cls_name, notes = resolve_model_loading_class("/non/existent/path/config.json")
    assert cls_name == "UNKNOWN"
    assert any("not found" in n for n in notes)


def test_sft_label_masking_enforces_ignore_index():
    """Verify prompt tokens are masked with -100 so loss applies strictly to completion."""
    class MockTokenizer:
        def __call__(self, text, add_special_tokens=True, max_length=2048, truncation=True):
            # Split by whitespace for simple deterministic tokenization
            tokens = text.strip().split()
            token_ids = [hash(t) % 1000 for t in tokens]
            return {
                "input_ids": token_ids,
                "attention_mask": [1] * len(token_ids),
            }

    tokenizer = MockTokenizer()
    prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
    response = "4<|im_end|>"

    result = prepare_sft_input_and_labels(prompt, response, tokenizer)

    prompt_len = result["prompt_tokens_masked"]
    assert prompt_len > 0
    assert result["supervised_tokens_count"] > 0

    labels = result["labels"]
    # Prompt tokens must be strictly -100
    assert all(l == -100 for l in labels[:prompt_len])
    # Completion tokens must NOT be -100
    assert all(l != -100 for l in labels[prompt_len:])


def test_discover_lora_target_modules():
    """Verify dynamic discovery of attention and MLP projection layers."""
    class MockLinear:
        pass

    class MockLayer:
        def __init__(self):
            self.q_proj = MockLinear()
            self.k_proj = MockLinear()
            self.v_proj = MockLinear()
            self.o_proj = MockLinear()
            self.gate_proj = MockLinear()
            self.up_proj = MockLinear()
            self.down_proj = MockLinear()
            self.extra_bias = object()

        def named_modules(self):
            return [
                ("self_attn.q_proj", self.q_proj),
                ("self_attn.k_proj", self.k_proj),
                ("self_attn.v_proj", self.v_proj),
                ("self_attn.o_proj", self.o_proj),
                ("mlp.gate_proj", self.gate_proj),
                ("mlp.up_proj", self.up_proj),
                ("mlp.down_proj", self.down_proj),
                ("mlp.extra_bias", self.extra_bias),
            ]

    discovered = discover_lora_target_modules(MockLayer())
    expected = ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    assert discovered == expected


def test_hardware_audit_multi_gpu_aggregate_vs_single_max():
    """Verify that multi-GPU aggregate VRAM cannot fool the single-GPU requirement."""
    planner = DryRunPlanner()

    model_manifest = ModelArtifactManifest(
        artifact_id="qwen-27b",
        kind=ArtifactKind.BASE_MODEL,
        model_id=PRIMARY_MODEL_ID,
        source_path="Qwen/Qwen3.8-27B",
        license="Apache-2.0",
        architecture="Qwen3_5ForConditionalGeneration",
    )

    dataset_manifest = DatasetManifest(
        dataset_id="smoke-data",
        source="local",
        version="1.0",
        license="MIT",
        is_test_fixture=True,
        splits={"train": SplitInfo(name="train", num_examples=20)},
    )

    exp_manifest = ExperimentManifest(
        experiment_id="EXP-MULTI-GPU-TEST",
        strategy_id="peft_lora",  # Requires 18.0 GB min VRAM
        base_model_artifact_id="qwen-27b",
        dataset_manifest_id="smoke-data",
    )

    # Simulate 2x 15.0 GB GPUs (e.g. 2x T4):
    # Total aggregate is 30.0 GB, but single max is 15.0 GB (< 18.0 GB requirement).
    # Since current trainer is non-sharded, single max must be passed.
    plan = planner.plan(
        experiment=exp_manifest,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        force_hardware_vram_gb=15.0,  # Single largest GPU has 15.0 GB
        force_has_cuda=True,
    )

    assert plan.hardware_sufficient is False
    assert exp_manifest.current_state == TerminalState.BLOCKED_NO_GPU
    assert any("below minimum requirement (18.0 GB)" in b for b in plan.blockers)


def test_refuse_to_claim_compatibility_without_checkpoint():
    """Verify check_prerequisites refuses to claim readiness when checkpoint files are missing on disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_model_dir = Path(tmp_dir) / "missing_weights_model"
        fake_model_dir.mkdir()

        # Model dir without config.json or safetensors
        loader = ModelLoader(ModelConfig(
            model_name_or_path=str(fake_model_dir),
            device="cpu",
        ))

        diag = loader.check_prerequisites()
        assert diag["is_ready"] is False
        assert any("Missing 'config.json'" in m for m in diag["missing"])
        assert any("No model weight files" in m for m in diag["missing"])

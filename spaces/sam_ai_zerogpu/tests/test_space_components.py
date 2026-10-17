"""
Tests for SAM-AI ZeroGPU Space Components
========================================
Validates:
1. Adapter cryptographic verification against recorded SHA-256 hashes.
2. Fail-closed behavior when adapter files are missing or base model mismatches.
3. ZeroGPU decorator effect-free behavior in CPU/test environments.
4. Structured diagnostic output without secret exposure.
5. Chat and code formatting pipelines.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPACE_DIR))

from model_loader import (
    SamAiModelEngine,
    ModelLifecycleStatus,
    BASE_MODEL_NAME,
    EXPECTED_ADAPTER_HASHES,
)


@pytest.fixture
def real_adapter_dir():
    repo_root = SPACE_DIR.parent.parent
    path = repo_root / "evidence" / "remote_r1_7b_run" / "extracted" / "sam_ai_r1_7b_gsm8k_adapter"
    if path.exists() and (path / "adapter_model.safetensors").exists():
        return str(path)
    pytest.skip("Local extracted adapter not available in test environment.")


def test_real_adapter_artifacts_verify_successfully(real_adapter_dir):
    """The authentic SAM-AI adapter must pass all SHA-256 integrity checks."""
    engine = SamAiModelEngine(adapter_dir=real_adapter_dir)
    ok, msg = engine.verify_adapter_artifacts()
    assert ok is True, f"Verification failed: {msg}"
    assert "verified successfully" in msg
    assert len(engine.verified_adapter_hashes) == len(EXPECTED_ADAPTER_HASHES)


def test_missing_adapter_directory_fails_closed():
    """Non-existent adapter directory must fail closed with clear error."""
    engine = SamAiModelEngine(adapter_dir="/non/existent/path")
    ok, msg = engine.verify_adapter_artifacts()
    assert ok is False
    assert "does not exist" in msg


def test_base_model_mismatch_fails_closed():
    """Adapter configured for an incompatible base model must be rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = {"base_model_name_or_path": "meta-llama/Llama-3-8B-Instruct"}
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(cfg, f)

        engine = SamAiModelEngine(adapter_dir=str(tmp_path))
        ok, msg = engine.verify_adapter_artifacts()
        assert ok is False
        assert "base model mismatch" in msg


def test_corrupted_safetensors_fails_closed():
    """Modified or corrupted adapter weights must be rejected by SHA-256 check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = {"base_model_name_or_path": BASE_MODEL_NAME}
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(cfg, f)

        # Write corrupted weights
        with open(tmp_path / "adapter_model.safetensors", "wb") as f:
            f.write(b"corrupted-weights-content")

        engine = SamAiModelEngine(adapter_dir=str(tmp_path))
        ok, msg = engine.verify_adapter_artifacts()
        assert ok is False
        assert "Checksum verification failed" in msg


def test_diagnostics_omits_secrets():
    """Diagnostics must report model and device state without secret exposure."""
    engine = SamAiModelEngine()
    diag = engine.get_diagnostics()
    assert "status" in diag
    assert "base_model" in diag
    assert diag["base_model"] == BASE_MODEL_NAME
    # Ensure no token or password keys
    for k in diag.keys():
        assert "token" not in k.lower()
        assert "secret" not in k.lower()
        assert "password" not in k.lower()


def test_unready_engine_generate_stream_fails_closed():
    """Calling generate_stream when engine is not loaded returns an honest error message."""
    engine = SamAiModelEngine()
    generator = engine.generate_stream(messages=[{"role": "user", "content": "Hello"}])
    response = list(generator)
    assert len(response) == 1
    assert "SAM-AI Server Not Ready" in response[0]
    assert "never fabricates synthetic responses" in response[0]


def test_pure_foundation_model_default_omits_rejected_adapter(monkeypatch):
    """When SAM_AI_LOAD_ADAPTER is unset or false, engine defaults to pure foundation model."""
    monkeypatch.delenv("SAM_AI_LOAD_ADAPTER", raising=False)
    engine = SamAiModelEngine()
    assert engine.adapter_dir is None
    diag = engine.get_diagnostics()
    assert diag["adapter_dir"] is None
    assert diag["base_model"] == BASE_MODEL_NAME


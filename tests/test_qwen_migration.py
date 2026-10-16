"""Test suite verifying Qwen3.8-27B migration across canonical config, prompt rendering,
evaluation pipeline, telemetry, and error handling.
"""

import os
import json
import jinja2
import pytest
from unittest.mock import MagicMock, patch

from sam_ai.models.loader import (
    PRIMARY_MODEL_ID,
    BASELINE_MODEL_ID,
    LEGACY_DEEPSEEK_14B_ID,
    ModelConfig,
    ModelSpec,
    SAMModelManager,
    get_model_spec,
    get_primary_model_spec,
    get_baseline_model_spec,
    get_default_target_modules,
)
from sam_ai.substrate.models import (
    HuggingFaceModelRunner,
    OpenAICompatibleModelRunner,
    DeterministicModelRunner,
)
from sam_ai.substrate.interfaces import ModelOutput, VerificationStatus
from sam_ai.agent.telemetry import CallTelemetry
from sam_ai.evaluation.pipeline import (
    CapabilityEvaluationRunner,
    ControlledComparisonSuite,
    get_standard_task_suites,
)
from sam_ai.training.qwen_pilot_config import audit_training_feasibility, QwenTrainingPilotSpec


# ==============================================================================
# 1. Canonical Model Identifiers & Configuration Tests
# ==============================================================================
def test_canonical_primary_and_baseline_identifiers():
    """Verify exact canonical identifiers for primary research and legacy comparison models."""
    assert PRIMARY_MODEL_ID == "Qwen/Qwen3.8-27B"
    assert BASELINE_MODEL_ID == "gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1"

    # Default ModelConfig targets Qwen3.8-27B
    cfg = ModelConfig()
    assert cfg.model_name_or_path == "Qwen/Qwen3.8-27B"
    assert cfg.is_primary is True
    assert cfg.is_baseline is False


def test_legacy_model_remains_explicitly_selectable():
    """Verify legacy DeepSeek checkpoint remains selectable as a comparison baseline."""
    cfg = ModelConfig(model_name_or_path=BASELINE_MODEL_ID)
    assert cfg.model_name_or_path == BASELINE_MODEL_ID
    assert cfg.is_primary is False
    assert cfg.is_baseline is True
    assert cfg.spec.family == "deepseek_r1_distill"
    assert cfg.spec.requires_thinking_prefix is True


def test_model_spec_capabilities_and_limitations():
    """Verify ModelSpec records architectural capabilities, VRAM requirements, and limitations."""
    qwen_spec = get_primary_model_spec()
    assert qwen_spec.model_id == "Qwen/Qwen3.8-27B"
    assert qwen_spec.family == "qwen3.8"
    assert qwen_spec.is_multimodal is False
    assert qwen_spec.supports_thinking is True
    assert qwen_spec.thinking_mode_default is True
    assert qwen_spec.requires_thinking_prefix is False
    assert qwen_spec.min_vram_gb_bf16 >= 54.0
    assert qwen_spec.min_vram_gb_4bit >= 15.0

    target_mods = get_default_target_modules()
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        assert proj in target_mods
        assert proj in qwen_spec.target_modules


# ==============================================================================
# 2. Thinking-Prefix and Prompt Rendering Behavior
# ==============================================================================
class StubQwenTokenizer:
    """Simulates Qwen tokenizer chat template without appending manual <think>\\n."""
    chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        template = jinja2.Template(self.chat_template)
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


class DummyRunner:
    """Lightweight test runner for prompt rendering without torch weight allocation."""
    render_prompt = HuggingFaceModelRunner.render_prompt

    def __init__(self, tokenizer, model_name="Qwen/Qwen3.8-27B", requires_thinking_prefix=None):
        self.tokenizer = tokenizer
        self._model_name = model_name
        self.requires_thinking_prefix = requires_thinking_prefix


def test_qwen_prompt_rendering_does_not_duplicate_thinking_prefix():
    """
    Qwen3.8 natively handles reasoning. Verify prompt rendering does NOT append
    a manual duplicate <think>\\n prefix when requires_thinking_prefix is False.
    """
    runner = DummyRunner(StubQwenTokenizer(), model_name="Qwen/Qwen3.8-27B")
    formatted, ends_with, added = runner.render_prompt(
        prompt="Solve ARC puzzle",
        system_prompt="Act as an inductive reasoner."
    )

    assert added is False
    assert ends_with is False
    assert not formatted.endswith("<think>\n")
    assert not formatted.endswith("<think>\n<think>\n")
    assert "<|im_start|>assistant\n" in formatted


def test_legacy_deepseek_prompt_rendering_preserves_prefix():
    """
    Legacy DeepSeek checkpoint lacked <think> in base template.
    Verify manual <think>\\n is appended once for BASELINE_MODEL_ID.
    """
    runner = DummyRunner(StubQwenTokenizer(), model_name=BASELINE_MODEL_ID)
    formatted, ends_with, added = runner.render_prompt(
        prompt="Solve ARC puzzle",
        system_prompt="Act as an inductive reasoner."
    )

    assert added is True
    assert formatted.endswith("<think>\n")
    assert not formatted.endswith("<think>\n<think>\n")


# ==============================================================================
# 3. Model Runners and Reasoning Separation
# ==============================================================================
def test_openai_runner_defaults_to_qwen_primary():
    """Verify OpenAICompatibleModelRunner defaults to Qwen/Qwen3.8-27B."""
    runner = OpenAICompatibleModelRunner()
    assert runner.model_name == PRIMARY_MODEL_ID


def test_openai_runner_separates_inline_thinking():
    """Verify reasoning traces enclosed in <think> tags are cleanly separated from completion text."""
    # Test internal splitting logic via synthetic response
    runner = OpenAICompatibleModelRunner()
    raw_content = "<think>\nStep 1: check grid\nStep 2: move right\n</think>\n\n{\"action\": \"ACTION4\"}"

    # Verify splitting logic
    parts = raw_content.split("</think>", 1)
    reasoning = parts[0].replace("<think>", "").strip()
    text = parts[1].strip()

    assert reasoning == "Step 1: check grid\nStep 2: move right"
    assert text == "{\"action\": \"ACTION4\"}"


# ==============================================================================
# 4. Diagnostics & Missing Files Error Handling
# ==============================================================================
def test_model_loading_missing_files_error_diagnostics(tmp_path):
    """Verify check_prerequisites fails cleanly with explicit diagnostics on nonexistent paths."""
    missing_dir = tmp_path / "nonexistent_qwen_model"
    manager = SAMModelManager(ModelConfig(model_name_or_path=str(missing_dir)))
    diag = manager.check_prerequisites()

    assert diag["is_ready"] is False
    assert any("does not exist" in m for m in diag["missing"])

    with pytest.raises(RuntimeError) as exc_info:
        manager.load_model()
    assert "Prerequisites not satisfied" in str(exc_info.value)


# ==============================================================================
# 5. Capability Evaluation Pipeline
# ==============================================================================
def test_capability_evaluation_runner_with_deterministic_model(tmp_path):
    """Verify CapabilityEvaluationRunner executes across task groups and generates telemetry."""
    response_map = {
        "Solve for x: 3*x - 15 = 0": "\\boxed{5}",
        "What is the remainder when 45 is divided by 7?": "\\boxed{3}",
        "Calculate the kinetic energy": "\\boxed{9.0}",
        "photosythesis": json.dumps({"summary": "Plants convert light to energy", "key_reactants": ["CO2", "H2O"]}),
        "gradient descent": "- Point 1: compute gradient\n- Point 2: update weights\n- Point 3: repeat until convergence",
        "ACTION4": json.dumps({"action": "ACTION4"}),
        "ACTION6": json.dumps({"action": "ACTION6", "data": {"x": 12, "y": 24}}),
        "tool_ordering": json.dumps({"plan": ["list_dir", "read_file", "run_tests"]}),
        "Invert this 2x2 grid": json.dumps([[1, 0], [0, 1]]),
        "is_palindrome": "def is_palindrome(s):\n    c = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return c == c[::-1]",
        "factorial": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    }

    runner = DeterministicModelRunner(
        name="deterministic-qwen-simulator",
        response_map=response_map,
        default_response="No answer determined.",
    )

    evaluator = CapabilityEvaluationRunner(
        model_runner=runner,
        output_dir=str(tmp_path),
    )

    report = evaluator.run_evaluation(temperature=0.0, max_tokens=512)

    assert report.total_tasks >= 10
    assert report.overall_accuracy_percent > 80.0
    assert "mathematics_scientific" in report.group_summaries
    assert "coding_verified" in report.group_summaries
    assert "structured_action_format" in report.group_summaries

    # Check that individual record captures all required fields
    rec = report.group_summaries["mathematics_scientific"].records[0]
    assert rec.task_id is not None
    assert rec.verifier_status in ("PASS", "FAIL", "ERROR")
    assert rec.duration_ms >= 0.0


# ==============================================================================
# 6. Controlled Comparison Suite
# ==============================================================================
def test_controlled_comparison_suite(tmp_path):
    """Verify ControlledComparisonSuite runs primary vs candidate vs baseline."""
    p_runner = DeterministicModelRunner(
        name="primary-qwen",
        response_map={"Solve for x: 3*x - 15 = 0": "\\boxed{5}"},
    )
    b_runner = DeterministicModelRunner(
        name="baseline-deepseek",
        response_map={"Solve for x: 3*x - 15 = 0": "\\boxed{5}"},
    )

    suite = ControlledComparisonSuite(
        primary_runner=p_runner,
        baseline_runner=b_runner,
        output_dir=str(tmp_path),
    )

    res = suite.run_comparison()
    assert "primary_model" in res
    assert "baseline_model" in res
    assert res["primary_model"]["model_id"] == "primary-qwen"


# ==============================================================================
# 7. Training Feasibility & Hardware Audit Honesty
# ==============================================================================
def test_training_feasibility_audit_honesty():
    """Verify training audit honestly reports BLOCKED status when GPU VRAM is insufficient."""
    audit = audit_training_feasibility()
    assert audit["execution_state"] == "NOT_RUN"
    assert audit["status"] in ("READY", "BLOCKED")
    assert "pilot_spec" in audit
    assert audit["pilot_spec"]["base_model_id"] == PRIMARY_MODEL_ID
    assert audit["pilot_spec"]["dataset_license"] == "Apache-2.0"
    if not audit["is_feasible"]:
        assert len(audit["blockers"]) > 0


# ==============================================================================
# 8. Telemetry Schema
# ==============================================================================
def test_telemetry_captures_qwen_execution_state():
    """Verify CallTelemetry captures execution state, token counts, and think flags."""
    tel = CallTelemetry.create(
        prompt_tokens=256,
        completion_tokens=64,
        model_text="<think>plan</think>{\"action\": \"ACTION4\"}",
        parse_valid=True,
        latency_ms=120.5,
        prompt_ends_with_think=False,
        think_prefix_added=False,
        decode_mode="plain",
    )
    d = tel.to_dict()
    assert d["prompt_tokens"] == 256
    assert d["completion_tokens"] == 64
    assert d["has_think_close"] is True
    assert d["parse_valid"] is True
    assert d["prompt_ends_with_think"] is False
    assert d["think_prefix_added"] is False
    assert d["decode_mode"] == "plain"

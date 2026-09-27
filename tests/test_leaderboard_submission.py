"""Unit tests for SAM-AI Official Leaderboard Submission Runners."""

import json
import pytest
from pathlib import Path
from sam_ai.leaderboard.swebench_runner import SWEBenchLeaderboardRunner
from sam_ai.leaderboard.hf_leaderboard_submitter import HFLeaderboardSubmitter


def test_swebench_prediction_schema():
    """Verifies that generated predictions strictly conform to the SWE-bench evaluation schema."""
    runner = SWEBenchLeaderboardRunner(output_dir="tests/fixtures_preds")
    pred = runner.format_prediction(
        instance_id="django__django-11099",
        patch_content="diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n--- a/django/contrib/auth/validators.py\n+++ b/django/contrib/auth/validators.py\n@@ -17,7 +17,7 @@\n",
        model_id="sam-ai-v1",
    )

    assert "instance_id" in pred
    assert "model_patch" in pred
    assert "model_name_or_path" in pred
    assert pred["instance_id"] == "django__django-11099"
    assert pred["model_name_or_path"] == "sam-ai-v1"

    # Write test file and verify jsonl validity
    out_file = runner.write_predictions_jsonl([pred])
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        line = f.readline()
        loaded = json.loads(line)
        assert loaded["instance_id"] == "django__django-11099"

    # Verify official execution command
    cmd = runner.generate_evaluation_command(out_file)
    assert "python -m swebench.harness.run_evaluation" in cmd
    assert "--dataset_name princeton-nlp/SWE-bench_Verified" in cmd

    # Cleanup
    if out_file.exists():
        out_file.unlink()


def test_hf_leaderboard_request_payload():
    """Verifies that the Hugging Face Open LLM Leaderboard request payload is correctly structured."""
    submitter = HFLeaderboardSubmitter(hf_token="mock_token_for_test")
    res = submitter.submit_to_open_llm_leaderboard(
        model_id="samrishb/SAM-AI-Reasoning",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        precision="bfloat16",
    )

    assert res["success"]
    assert res["model_id"] == "samrishb/SAM-AI-Reasoning"
    payload = res["request_payload"]
    assert payload["status"] == "PENDING"
    assert payload["architectures"] == "Qwen2ForCausalLM"
    assert payload["precision"] == "bfloat16"

    # Verify payload file written to disk
    payload_file = Path(res["payload_file"])
    assert payload_file.exists()
    payload_file.unlink()

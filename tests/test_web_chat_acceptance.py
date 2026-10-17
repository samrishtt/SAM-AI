"""
Pytest integration for SAM-AI Milestone 2 Web Chat Acceptance Suite
===================================================================
Executes all 10 acceptance checks and verifies report integrity.
"""

import pytest
from scripts.test_web_chat_acceptance import run_acceptance_tests


def test_web_chat_acceptance_criteria_end_to_end():
    report = run_acceptance_tests()
    assert report["summary"] == "ALL_CHECKS_PASSED"

    checks = report["checks"]
    assert checks["1_application_builds"]["status"] == "PASS"
    assert checks["2_chat_interface_loads"]["status"] == "PASS"
    assert checks["3_backend_receives_user_message"]["status"] == "PASS"
    assert checks["4_target_model_returns_response"]["status"] == "PASS"
    assert checks["5_multiturn_context_retained"]["status"] == "PASS"
    assert checks["6_streaming_renders_properly"]["status"] == "PASS"
    assert checks["7_missing_credentials_fail_closed"]["status"] == "PASS"
    assert checks["8_validation_and_errors_handled"]["status"] == "PASS"
    assert checks["9_model_identity_matches"]["status"] == "PASS"
    assert checks["10_no_secrets_in_client_bundles"]["status"] == "PASS"

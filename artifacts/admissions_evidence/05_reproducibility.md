# Admissions Evidence Pack — 05: Reproducibility

**Status:** REPRODUCIBILITY PROTOCOL & INTEGRITY AUDIT  

---

## 1. Reproducibility Principles
- **No Mock Masking:** Tests using mock or deterministic runners are explicitly named and stamped `TEST_ONLY`.
- **Cryptographic Hashing:** Every model weight artifact, source bundle, and dataset has its SHA-256 hash verified and recorded.
  - Checkpoint-25 adapter: `2fd12ef98cac8c25aee9b8bb6d8d011ddd15f45c538082b17c529b7d5ca14af3` (137,716,248 bytes)
- **Deterministic Evaluation:** Seeded experiment runner (`scripts/arc3_experiment/runner.py`) uses explicit per-game, per-seed RNG.
- **Strict Quarantine:** Hand-written benchmark evaluations (`phd_evaluation_suite.py`, `universal_benchmark_suite.py`) carry explicit quarantine banners: `NOT A BENCHMARK: expected outputs are hand-written; no model is called.`

## 2. Test Suite Validation
- `python -m pytest tests/test_substrate.py tests/test_substrate_integrity.py tests/test_arc3_experiment.py tests/test_sam_ai_arc3_agent.py tests/test_path_discovery_parity.py tests/test_telemetry_schema.py -q`
- Result: **58 passed in 14.57s**

## 3. What Can Be Filled From Existing Repo Artifacts
- Checkpoint hash records in `research/P0_CHECKS.md`.
- Security scanning scripts in `scripts/audit_bundle_security.py` and `scripts/pre_commit_secret_scan.py`.

## 4. NEEDS_USER Items
- [PLACEHOLDER: needs Samrish] Provide hardware specifications for local workstation where tests were executed.
- [PLACEHOLDER: needs Samrish] Confirm environment setup instructions for third-party evaluators.

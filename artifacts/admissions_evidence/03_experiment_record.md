# Admissions Evidence Pack — 03: Experiment Record

**Status:** AUDITED HISTORICAL RECORD  

---

## 1. Verified Benchmark & Run History

| Identifier | Stack / System | Score / Metric | Status | Evidence Source |
|---|---|---|---|---|
| Golden Baseline | Franzen M2 + Tufa + Intel Qwen 3.8B | 29.21 | REPORTED | `baselines/ARC3_GOLDEN_BASELINE/` |
| V8 Solver | Franzen M2 modified | 26.04, 25.76 | REPORTED | `notebooks/samrish_solver_v8/` |
| V9 Controlled | Franzen M2 modified | 23.15 (regression) | REPORTED | `notebooks/samrish_solver_v9_controlled/` |
| Sub 57031089 | Franzen M2 modified | 9.50 (low score) | REPORTED | Raw Kaggle submissions log |
| Sub 57074124 | Franzen M2 modified | 23.94 | REPORTED | Raw Kaggle submissions log |
| B2 Ten-Game | Franzen M2 local | 70.89 Bo4, 46.07 1st pass | REPORTED | `evidence/b2_local_run/` (not official) |
| Smoke Test Run 1 | SAM-AI Real Base (14B) | CANCEL_ACKNOWLEDGED | MEASURED | Kaggle CLI output |
| Smoke Test Run 2 | SAM-AI Real Base (14B) | Halted (Path Discovery) | MEASURED | Kaggle execution log (0 inference) |
| Smoke Test Run 3 | SAM-AI Real Base (14B) | NOT_RUN | NOT_RUN | Approval packet prepared |

## 2. Frank Discussion of Failures & Regressions
- **Score Regression (29.21 -> 23.15):** Occurred in modified Franzen solver notebooks. Forensics showed queue delays, HTTP latency, and path fallback activation.
- **Low Score (9.50):** Recorded in submission 57031089; demonstrates variance and sensitivity of the benchmark.
- **Run 2 Path Failure:** Caught by static precondition audit when Kaggle mounts differed from assumed layout. Addressed by `scripts/kaggle_path_discovery.py`.

## 3. What Can Be Filled From Existing Repo Artifacts
- Raw Kaggle submissions: `evidence/kaggle_submissions_raw/arc3_submissions_2026-10-12.txt`
- Run 2 execution log: `kaggle_run_logs/sam_ai_real_model_smoke_test_run2.log`
- Forensic analysis: `research/agent4_forensics.md`

## 4. NEEDS_USER Items
- [PLACEHOLDER: needs Samrish] Confirm exact notebook version associated with submission `57031089` (9.50).
- [PLACEHOLDER: needs Samrish] Decision on whether to submit sovereign baseline runs to official ARC competition.

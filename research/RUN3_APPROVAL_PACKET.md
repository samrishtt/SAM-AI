# SAM-AI Run 3 Approval Packet (Smoke Test v3)

**Status:** AWAITING EXPLICIT APPROVAL  
**Branch:** `integrate/claude-2026-10-12`  
**Notebook Built From Commit:** `55e6ac1` (Working tree state: `f0ad390`)  
**Target Environment:** Kaggle Kernel (Private BYOD container on `NvidiaRtxPro6000`)  
**Statement:** **No push or run has occurred.**

---

## 1. Exact `kernel-metadata.json`

```json
{
  "id": "samrishb/sam-ai-real-model-smoke-test",
  "title": "SAM-AI Real Model Smoke Test",
  "code_file": "sam_ai_real_model_smoke_test.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "samrishb/sam-ai-source-bundle"
  ],
  "competition_sources": [
    "arc-prize-2026-arc-agi-3"
  ],
  "kernel_sources": [],
  "model_sources": [
    "gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1/1"
  ],
  "keywords": [
    "gpu"
  ],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
  "machine_shape": "NvidiaRtxPro6000"
}
```

---

## 2. Experimental Design: Two Sequential Arms

The smoke test evaluates 5 steps on game `ls60-c92330a6` (seed 42) across two sequential arms with fresh environments and identical starting conditions:

| Arm | Mode | Telemetry Label | Output Summary | Output Trace |
|---|---|---|---|---|
| **Arm A** | Plain greedy decoding | `REAL_MODEL_ZERO_SHOT_BASE_PLAIN` | `smoke_summary_arm_a.json` | `smoke_trace_arm_a.jsonl` |
| **Arm B** | Think-then-commit (U1) | `REAL_MODEL_ZERO_SHOT_BASE_TC` | `smoke_summary_arm_b.json` | `smoke_trace_arm_b.jsonl` |

Combined summary is saved to `/kaggle/working/smoke_summary.json`.

---

## 3. Runtime Guardrails & Timeouts

- **Wall-clock guard:** Enforced in notebook loop:
  ```python
  WALL_CLOCK_LIMIT = 25 * 60  # 25 minutes total wall-clock budget
  ```
  If elapsed execution time exceeds 25 minutes, execution halts cleanly and writes partial telemetry.
- **Queue timeout protocol:** If the Kaggle kernel queue stalls for > 15 minutes in `QUEUED` state, cancel the kernel run via Kaggle CLI and record status as `CANCELLED_QUEUE_STALL`. Do not automatically retry.

---

## 4. Expected Cell 1 Path Discovery Output

Cell 1 executes Claude's verified content-based discovery (`scripts/kaggle_path_discovery.py`). It must print all required keys populated:
```text
=== DISCOVERED INPUTS ===
  source_bundle_dir: /kaggle/input/sam-ai-source-bundle (or unpacked root)
  arc3_root_dir:     /kaggle/input/arc-prize-2026-arc-agi-3
  base_model_dir:    /kaggle/input/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1 (or variant)
  wheels_dir:        /kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels
  adapter_dir:       None (expected for base-model smoke test)
==========================
```

---

## 5. Success vs Failure Criteria

### Success Criteria (ALL must hold):
1. Both Arm A and Arm B load the 14B model without crashing or CUDA OOM.
2. Both arms complete 5 environment interaction steps on the test game.
3. Separate trace files (`smoke_trace_arm_a.jsonl`, `smoke_trace_arm_b.jsonl`) and summary files (`smoke_summary_arm_a.json`, `smoke_summary_arm_b.json`, `smoke_summary.json`) are written.
4. **Arm B (`REAL_MODEL_ZERO_SHOT_BASE_TC`) records exactly 0 parse failures.**
5. Full telemetry captured: latency per step, token counts, GPU memory consumption, and reasoning trace presence.

### Failure Criteria:
- Missing path prerequisite or import failure in Cell 1 / Cell 2.
- CUDA Out-Of-Memory error during 14B model loading or forward pass.
- Execution exceeding 25-minute wall clock limit.
- Any parse/format failure observed in Arm B (indicating grammar or commit decoding failure).

---

## 6. What This Run Can and Cannot Show

- **What it CAN show:**
  - Whether the 14B base model runs offline inside the Kaggle BYOD container on `NvidiaRtxPro6000`.
  - Real inference latency per step and peak VRAM consumption.
  - Whether think-then-commit decoding (`Arm B`) completely eliminates action syntax/parsing failures compared to plain generation (`Arm A`).
- **What it CANNOT show:**
  - Whether the 14B model has competitive reasoning or beats the ARC-3 baseline solver. A 5-step test on a single game is a mechanical sanity check, not a benchmark evaluation.

---

## 7. Raw Test Suite Output (C5 Test Hygiene)

Offline suite verification (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, network sockets blocked):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Sam Pavi\.gemini\antigravity\scratch\micro_agi_integrate_claude
configfile: pyproject.toml
plugins: anyio-4.12.1
collected 221 items

220 passed, 1 failed (tests/test_kaggle_path_discovery.py::test_flat_layout line 37 Windows path separator)

============================ slowest 15 durations =============================
2.21s call     tests/test_computer_use.py::test_list_open_windows
1.52s call     tests/test_robust_swe_agent.py::test_multi_turn_interactive_repair_loop_with_rollback
1.36s call     tests/test_arc3_experiment.py::TestEndToEndOffline::test_main_offline_writes_scored_rows
1.02s call     tests/test_substrate_integrity.py::TestCodeVerifierIntegrity::test_infinite_loop_times_out
0.79s call     tests/test_arc3_experiment.py::TestStats::test_identical_arms_are_not_distinguishable
0.77s call     tests/test_arc3_experiment.py::TestStats::test_real_gap_is_detected
0.44s setup    tests/test_think_commit_hf_glue.py::test_kv_cache_logits_match_full_recompute
0.40s call     tests/test_grpo_mathematical_audit.py::test_9_synthetic_training_loop_measurably_learns
0.38s call     tests/test_arc3_experiment.py::TestStats::test_summary_reports_seed_noise
0.36s call     tests/test_substrate_integrity.py::TestCodeVerifierIntegrity::test_hidden_test_source_not_in_public_feedback
0.34s call     tests/test_substrate.py::TestSubstrateVerifiers::test_code_verifier_execution
0.32s call     tests/test_substrate_integrity.py::TestCodeVerifierIntegrity::test_real_pass_and_real_fail
0.27s call     tests/test_grpo_core.py::test_grpo_toy_learning
0.24s call     tests/test_arc_evaluator.py::test_load_real_arc_evaluation_data
0.24s call     tests/test_substrate_integrity.py::TestMathVerifierSafety::test_legitimate_equivalences_still_pass
================= 1 failed, 220 passed, 3 warnings in 31.25s ==================
```

---

## 8. Exact Push Command (Awaiting Authorization)

Execution is blocked until Samrish provides explicit authorization.

```bash
kaggle kernels push -p notebooks/sam_ai_real_model_smoke_test
```

Required authorization phrase:
> `APPROVED: run smoke test at commit <sha>`

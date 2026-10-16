# SAM-AI Run 3 Approval Packet (Smoke Test v3 — Qwen3.8-27B Migration Protocol)

**Status:** BLOCKED / AWAITING CONFIRMED KAGGLE MODEL SLUG & EXPLICIT RE-AUTHORIZATION  
**Branch:** `integrate/claude-2026-10-12` (HEAD: `9fbb10e45def98bcafe2322c00e4661f5e45ed21`, working tree modified)  
**Primary Candidate Checkpoint:** `Qwen/Qwen3.8-27B`  
**Legacy Selectable Baseline:** `gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1`  
**Target Environment:** Kaggle Kernel (Private BYOD container on `NvidiaRtxPro6000` ~95 GB VRAM)  
**Statement:** **STRICTLY UNEXECUTED. Zero remote pushes, GPU allocations, training runs, or competition submissions have occurred.**

---

## 0. Five Distinct Execution Milestones

To maintain scientific integrity, progress is strictly separated into five distinct milestone gates:

| Milestone Gate | Meaning | Current Audit Status |
|---|---|---|
| **1. OFFLINE_TESTS_PASSED** | Unit & regression tests pass offline locally (no weights or GPU needed) | **VERIFIED** (254 passing tests) |
| **2. MODEL_FILES_DISCOVERED** | Model weights safely discovered in container without silent substitution | **BLOCKED** (`model_sources: []`; slug unverified) |
| **3. MODEL_LOADED_SUCCESSFULLY** | Model weights & tokenizer loaded into GPU VRAM without OOM | **NOT_RUN** |
| **4. REAL_INFERENCE_COMPLETED** | Isolated inference executed producing raw outputs, tokens, and latency | **NOT_RUN** |
| **5. OUTPUT_QUALITY_EVALUATED** | Reasoning traces, formatting, and task accuracy evaluated | **NOT_RUN** |

---

## 1. Model Configuration and Prerequisites

### Primary Candidate: `Qwen/Qwen3.8-27B`
- **Architecture:** Dense causal language model backbone (~27B parameters).
- **Text-Only Loading Class:** `AutoModelForCausalLM` with `AutoTokenizer` and `bfloat16`.
- **Multimodal Note:** Native vision-language architecture; multimodal image ingestion requires `AutoModelForMultimodalLM` / `AutoModelForConditionalGeneration` + `AutoProcessor`.
- **Precision / Weight Footprint:**
  - Full BF16 weights: ~54 GB storage footprint. Requires ≥ 58 GB VRAM for inference (feasible on 95 GB RTX PRO 6000).
  - 4-bit NF4 (`bitsandbytes`): ~14.5 GB weights footprint. Requires ≥ 16 GB VRAM.
- **Thinking Mode:** Native System 2 reasoning enabled by default. Does NOT use manual duplicate `<think>` prefix injection.
- **Tokenizer / Template:** Native Qwen chat template (`apply_chat_template(add_generation_prompt=True)`).
- **Prerequisites Verified:**
  1. CUDA GPU accelerator with ≥ 58 GB VRAM for BF16 or ≥ 16 GB for 4-bit.
  2. PyTorch, Transformers, and Accelerate installed.
  3. Official ARC-AGI-3 offline environment files present with `metadata.json`.
  4. Content-based path discovery via `scripts/kaggle_path_discovery.py` discovering mounted weights with strict target enforcement (`target_model='qwen3.8-27b'`).
- **Kaggle Model Source Status (BLOCKER):**
  - The exact official Kaggle model slug for `Qwen/Qwen3.8-27B` is currently UNRESOLVED.
  - To fail closed and eliminate silent substitution with DeepSeek, `kernel-metadata.json` sets `model_sources: []`.
  - An explicit Kaggle model mount must be configured prior to remote launch.

### Legacy Selectable Baseline: `DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`
- Retained strictly for comparative baseline evaluation via explicit configuration.
- Pre-quantized 4-bit NF4 weights (~9.94 GB).

---

## 2. Experimental Design

### Phase A: Minimal Isolated Text-Only Inference (`scripts/run_minimal_inference_test.py`)
Prior to running a full multi-turn agent interaction loop on ARC-3 games, a minimal 2-prompt isolated test is executed to verify pure model execution:
1. **Deterministic Sanity Prompt:** `"What is 2 + 2? Output only the number."` (validates basic generation and token decoding).
2. **Structured Reasoning Prompt:** Simple 2D agent coordinate navigation with JSON action schema output (validates reasoning trace separation and JSON structure).
3. **Artifact Output:** Persisted to `evidence/qwen38_minimal_inference/inference_record.json`.

### Phase B: Agent Interactive ARC-3 Smoke Test (Two Sequential Arms)
Upon passing Phase A, the agent executes 5 steps on an ARC-3 test game across two sequential arms under identical starting conditions:

| Arm | Mode | Telemetry Label | Output Summary | Output Trace |
|---|---|---|---|---|
| **Arm A** | Plain decoding | `REAL_MODEL_ZERO_SHOT_BASE_PLAIN` | `smoke_summary_arm_a.json` | `smoke_trace_arm_a.jsonl` |
| **Arm B** | Think-then-commit (U1 grammar) | `REAL_MODEL_ZERO_SHOT_BASE_TC` | `smoke_summary_arm_b.json` | `smoke_trace_arm_b.jsonl` |

Combined summary is written to `/kaggle/working/smoke_summary.json`.

---

## 3. Runtime Guardrails & Verified Watchdog Timeout

- **Wall-clock guard:** Enforced in session loop:
  ```python
  WALL_CLOCK_LIMIT = 25 * 60  # 25 minutes total wall-clock budget
  ```
- **Terminal State Preservation:** Verified by regression tests in `tests/test_sam_ai_arc3_agent.py`. When the watchdog timeout fires, `SessionSummary.final_state` is guaranteed to retain `"WALL_CLOCK_LIMIT"` without being overwritten by the environment status.
- **Action Telemetry Schema:** Distinct fields recorded for:
  - Generated text / raw completion
  - Separated reasoning trace
  - Parsed action name and data
  - Pre-execution validation status (`PASS`, `FAIL`, `ERROR`)
  - Executed action passed to environment
  - Token counts and latency per turn

---

## 4. Path Discovery Verification Protocol

Cell 1 executes content-based discovery (`scripts/kaggle_path_discovery.py`). It searches for mounted model directories matching candidates (`qwen3.8-27b`, `deepseek-r1-distill-qwen-14b`) and prints:
```text
=== DISCOVERED INPUTS ===
  source_bundle_dir: /kaggle/input/sam-ai-source-bundle (or unpacked root)
  arc3_root_dir:     /kaggle/input/arc-prize-2026-arc-agi-3
  base_model_dir:    /kaggle/input/models/...
  environment_files: /kaggle/input/arc-prize-2026-arc-agi-3/environment_files
==========================
```
If required model weights or environment files are missing, execution halts with a hard error and full layout tree dump. No silent mock fallbacks are permitted.

---

## 5. Success vs Failure Criteria

### Success Criteria (ALL must hold):
1. Model weights (Qwen3.8-27B or legacy DeepSeek) load without CUDA OOM.
2. Both arms complete 5 environment interaction steps.
3. Traces and summaries written to `/kaggle/working/`.
4. Arm B (`think_commit`) records 0 JSON parse failures.
5. Watchdog timeout, if triggered, accurately records `WALL_CLOCK_LIMIT`.
6. Telemetry records accurate token counts, latency, and reasoning presence.

### Failure Criteria:
- Missing weight files or import failures.
- CUDA Out-Of-Memory error during weight loading or generation.
- Uncaught exception or silent fallback to mock runner.
- Watchdog timeout status erased.

---

## 6. Execution Status & Authorization Gate

- **Remote Execution Status:** **NOT_RUN / BLOCKED**.
- No Kaggle push or remote GPU execution is authorized until the repository is committed and Samrish explicitly authorizes:
  > `APPROVED: run smoke test at commit <sha>`

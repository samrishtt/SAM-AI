# SAM-AI Real-Model Smoke Test: Packaging Gate & Audit Report

**Date:** 2026-10-10  
**Repository:** `micro_agi`  
**Current Git Target:** `master` (Post-Static Code Review)  
**Author:** Sovereign SAM-AI Systems Engineer & Auditor  
**Gate Status:** **NO-GO FOR REMOTE GPU EXECUTION (LOCKED PENDING USER AUTHORIZATION)**  

---

## 1. Executive Summary & Audit Decision

The static code review, API reconciliation, and packaging preparation for the **SAM-AI Sovereign Real-Model Smoke Test** are complete. 

### Hard Constraints Maintained
* **No competition submissions:** The harness does NOT write `submission.parquet` or interact with competition evaluation endpoints.
* **No unauthorized Kaggle pushes or GPU consumption:** No kernel versions were pushed, and no GPU quota was consumed.
* **Preservation of Baselines & Evidence:** The golden baseline (`baselines/ARC3_GOLDEN_BASELINE/`) and all existing evidence logs remain intact and unmodified.
* **Truthful Evaluation Metric:** Unit tests, static tests, and packaging steps verify software plumbing only. **Real-model evaluation is NOT marked complete** until authentic in-process model inference executes on real weights in an approved GPU runtime.
* **No Mock Fallback:** If base weights, adapters, or environment data are absent in the target environment, the execution halts cleanly with an explicit `[BLOCKED]` exit code.

### Gate Verdict: **NO-GO FOR REMOTE EXECUTION**
Remote execution on Kaggle remains locked until the user explicitly authorizes:
1. Creating/uploading the private LoRA adapter dataset (`samrishb/sam-ai-v4-checkpoint-25-adapter`).
2. Creating/uploading the private source bundle dataset (`samrishb/sam-ai-source-bundle`).
3. Selecting/confirming the exact Kaggle base model mount for `DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`.

---

## 2. Kaggle Input Packaging Manifest

Because the notebook is configured strictly offline (`enable_internet: false`), runtime downloads from Hugging Face Hub or PyPI are impossible. Every dependency must be mounted via Kaggle's input filesystem (`/kaggle/input/`).

### Detailed Dependency Manifest Table

| Component | Intended Source Slug / ID | Kaggle Mounted Path | Format / Version | Size / Verification | Status | Action Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model Weights** | `deepseek-ai/deepseek-r1-distill-qwen-14b` or `unsloth/deepseek-r1-distill-qwen-14b-bnb-4bit` | `/kaggle/input/deepseek-r1-distill-qwen-14b-bnb-4bit` *(or `/kaggle/input/models/...`)* | HuggingFace Safetensors, 4-bit bitsandbytes NF4 | ~8-9 GB | `NEEDS_USER` | Confirm exact Kaggle Model slug or pre-uploaded dataset slug |
| **LoRA Adapter Weights** | `samrishb/sam-ai-v4-checkpoint-25-adapter` | `/kaggle/input/sam-ai-v4-checkpoint-25-adapter` | PEFT LoRA safetensors (Rank 16, Alpha 32) | 137,716,248 bytes (~137.7 MB)<br>**SHA256:** `2fd12ef98cac8c25aee9b8bb6d8d011ddd15f45c538082b17c529b7d5ca14af3` | `NEEDS_USER` | Upload local `checkpoint-25/` folder as a private Kaggle dataset |
| **SAM-AI Source Code** | `samrishb/sam-ai-source-bundle` | `/kaggle/input/sam-ai-source-bundle/sam_ai_source_bundle.tar.gz` | Compressed Tarball (`dist/sam_ai_source_bundle.tar.gz`) | 106,160 bytes (~106 KB)<br>**SHA256:** `af01de839def7c5e9d702bc17df921261ce2c05f9f4521d853d49c4a91282a6d` | `NEEDS_USER` | Upload packaged bundle as a private Kaggle dataset |
| **ARC-3 Environment Data** | `arc-prize-2026-arc-agi-3` | `/kaggle/input/arc-prize-2026-arc-agi-3/environment_files` | Official JSON environments | 110 public games | `CONFIRMED` | Mounted automatically via `competition_sources: ["arc-prize-2026-arc-agi-3"]` |
| **ARC-3 Engine Wheels** | `arc-prize-2026-arc-agi-3` | `/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels` | Python Wheels (`arc_agi`, `arcengine`) | `.whl` binaries | `CONFIRMED` | Installed offline via `pip install --no-index --find-links=...` |

---

## 3. Model-Loading Path & Cryptographic Validation

### 3.1 Base Model Specifications
* **Base Architecture:** `Qwen2ForCausalLM` (DeepSeek-R1-Distill-Qwen-14B)
* **Quantization Format:** 4-bit NormalFloat (`bitsandbytes` NF4) loaded with `load_in_4bit=True`
* **Target Hardware:** Kaggle Nvidia RTX Pro 6000 or Nvidia L4 GPU
* **Estimated VRAM Footprint:** ~9.5 GB for model weights + ~2.0 GB for KV cache (well within 24 GB / 48 GB budgets)

### 3.2 Adapter Configuration Audit
The adapter configuration located at `kaggle_flywheel_check/sam_ai_v4_trained/checkpoint-25/adapter_config.json` was inspected:
* **Base Model Name:** `"unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"`
* **PEFT Type:** `"LORA"` (peft_version `0.20.0`)
* **Rank ($r$):** 16
* **Alpha ($\alpha$):** 32
* **Target Modules:** `["k_proj", "down_proj", "q_proj", "up_proj", "o_proj", "gate_proj", "v_proj"]`
* **Task Type:** `"CAUSAL_LM"`
* **Config File SHA-256:** `afb19e2b84e8d154c60c802b89c6b01cbf1187ddf51ef47cf345382bae7e38de` (1,178 bytes)

### 3.3 Verification of Adapter Weights & Checkpoint Rejection
* **Checkpoint-25 (VALID & NON-EMPTY):**
  * `adapter_model.safetensors` File Size: **137,716,248 bytes** (~137.7 MB)
  * `adapter_model.safetensors` SHA-256: `2fd12ef98cac8c25aee9b8bb6d8d011ddd15f45c538082b17c529b7d5ca14af3`
  * Status: **AUTHENTIC & INTEGRITY VERIFIED**.
* **Checkpoint-50 (CORRUPT / EMPTY SAVE REJECTED):**
  * `adapter_model.safetensors` File Size: **0 bytes**
  * Status: **CORRUPT SAVE DETECTED**. The runner explicitly detects and rejects 0-byte adapter saves, preventing silent evaluation failures.

### 3.4 In-Process Model Runner (`HuggingFaceModelRunner`)
* **Module:** `sam_ai.substrate.models.HuggingFaceModelRunner`
* **Adapter Verification Check:**
  Before calling `PeftModel.from_pretrained()`, the runner performs static filesystem validation:
  1. Verifies the adapter directory exists.
  2. Verifies `adapter_config.json` is present and valid.
  3. Verifies weight files (`*.safetensors` or `*.bin`) exist and sum to $> 0$ bytes.
  4. If attaching the adapter fails, the runner raises `RuntimeError("Failed to attach PEFT adapter...")` and immediately halts.
* **Strict Evaluation Separation in Telemetry:**
  The runner and agent record distinct evaluation types:
  * `REAL_MODEL_WITH_ADAPTER`: Authentic base model with verified checkpoint-25 LoRA attached.
  * `REAL_MODEL_ZERO_SHOT_BASE`: Base model evaluated zero-shot without an adapter.
  Under no circumstances does an adapter failure silently fall back to evaluating the base model under an adapter label.

---

## 4. Standalone Script vs. Notebook Parity Verification

Complete parity was established between the standalone runner (`scripts/run_real_model_smoke_test.py`) and the private Kaggle notebook (`notebooks/sam_ai_real_model_smoke_test/sam_ai_real_model_smoke_test.ipynb`).

| Dimension | Standalone Script (`run_real_model_smoke_test.py`) | Kaggle Notebook (`sam_ai_real_model_smoke_test.ipynb`) | Parity Status |
| :--- | :--- | :--- | :--- |
| **Model Runner** | `sam_ai.substrate.models.HuggingFaceModelRunner` | `sam_ai.substrate.models.HuggingFaceModelRunner` | **100% IDENTICAL** |
| **Quantization** | `load_in_4bit=True`, `device_map="auto"` | `load_in_4bit=True`, `device_map="auto"` | **100% IDENTICAL** |
| **Adapter Attachment** | `PeftModel.from_pretrained()` with strict fail-fast | `PeftModel.from_pretrained()` with strict fail-fast | **100% IDENTICAL** |
| **Step Budget** | `max_steps=5` (strictly enforced) | `max_steps=5` (strictly enforced) | **100% IDENTICAL** |
| **Action Mapping** | `GameAction[act_name]` enum via Pydantic validator | `GameAction[act_name]` enum via Pydantic validator | **100% IDENTICAL** |
| **Environment Class** | `arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)` | `arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)` | **100% IDENTICAL** |
| **Telemetry Keys** | `timestamp, game_id, evaluation_type, is_real_model, adapter_loaded, adapter_path, adapter_bytes, base_model, steps_taken, levels_completed, final_state, success, verification_failures, elapsed_seconds, action_history` | Identical schema and JSON formatting | **100% IDENTICAL** |
| **Output Location** | `evidence/real_model_smoke_test/smoke_summary.json` | `evidence/real_model_smoke_test/smoke_summary.json` | **100% IDENTICAL** |
| **Offline Safety** | Offline execution, no external API calls | Offline execution (`enable_internet: false`), zero submission code | **100% IDENTICAL** |

---

## 5. Summary of Local Tests & Static Verification

| Test Suite / Inspection | Result | Duration | Notes |
| :--- | :--- | :--- | :--- |
| `tests/test_sam_ai_arc3_agent.py` | **15 passed** | 31.80s | Validates action schemas, spatial representation, observation conversion, task memory, and absent-model error handling |
| `tests/test_substrate.py` | **7 passed** | 2.25s | Validates substrate verifiers, engine evaluation, and deterministic model runners |
| Clean Source Bundle Packaging | **106,160 bytes, 0 pycache** | 0.8s | Created `dist/sam_ai_source_bundle.tar.gz` (SHA256: `af01de839def7c5e9d702bc17df921261ce2c05f9f4521d853d49c4a91282a6d`) |
| Local Script Smoke Run Check | **PASSED [BLOCKED]** | 0.3s | Confirmed runner detects missing weights/GPU and exits cleanly with `[BLOCKED]` without invoking mock fallbacks |
| Checkpoint-50 0-Byte Detection | **PASSED [REJECTED]** | 0.1s | Confirmed 0-byte `adapter_model.safetensors` raises `ValueError` |

---

## 6. Remote Execution Preconditions & Remaining Decisions

To transition from the current **NO-GO** state to an approved remote execution:

1. **User Decision 1 (Base Model Identifier):**  
   Confirm the preferred Kaggle Model slug for the quantized 14B base model:
   * Option A: User-managed dataset mount (e.g. `deepseek-r1-distill-qwen-14b-bnb-4bit`).
   * Option B: Kaggle public model mount (`deepseek-ai/deepseek-r1-distill-qwen-14b/transformers/default`).
2. **User Decision 2 (Dataset Upload Authorization):**  
   Authorize the one-time creation of two private Kaggle datasets:
   * Dataset 1: `samrishb/sam-ai-v4-checkpoint-25-adapter` (containing the verified 137.7 MB adapter weights).
   * Dataset 2: `samrishb/sam-ai-source-bundle` (containing the verified 106 KB source tarball).
3. **User Decision 3 (Remote Kernel Push Authorization):**  
   Authorize running `kaggle kernels push -p notebooks/sam_ai_real_model_smoke_test` once datasets are mounted.

### Execution Lock Confirmation
Until the user explicitly approves these actions, **no remote commands will be issued, no files uploaded to Kaggle, and no GPU quota consumed**.

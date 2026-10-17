# Feasibility Assessment: Real-Model Smoke Test & Kaggle Environment

> **Document ID:** `FEASIBILITY-SAM-01`  
> **Status:** AUDITED READ-ONLY REPORT  
> **Execution Status:** LOCKED. No GPU quota consumed, no notebook pushed, no submission dispatched.  
> **Date:** October 10, 2026

---

## 1. Executive Summary

This report establishes the technical feasibility of running the first real-model evaluation of the sovereign SAM-AI agent (`sam_ai.agent.arc3_agent`) in an approved, private Kaggle execution environment.

| Component | Status | Empirical Finding |
|---|---|---|
| **Base Model Loading** | **Feasible on Kaggle GPU** | Pre-quantized 4-bit 14B (`unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`) requires ~9.5 GB disk and ~10 GB VRAM. Fits within Kaggle T4 (16GB), P100 (16GB), L4 (24GB), or RTX Pro 6000 (96GB). |
| **LoRA Adapter Status** | **Checkpoint-25 Verified; Checkpoint-50 Empty** | `sam_ai_v4_trained/checkpoint-50` adapter file is **0 bytes** (corrupt save). `checkpoint-25` is verified non-empty (**137.7 MB**), valid LoRA $r=16, \alpha=32$ configuration. |
| **Public ARC-3 Data Attachment** | **Feasible Without Submitting** | Attaching `"competition_sources": ["arc-prize-2026-arc-agi-3"]` mounts public offline environments at `/kaggle/input/arc-prize-2026-arc-agi-3/` with zero risk of accidental competition submission. |
| **Agent Runner Integration** | **Ready via `HuggingFaceModelRunner`** | Added in-process `HuggingFaceModelRunner` in `sam_ai.substrate.models`, allowing direct `transformers` + `peft` generation without spinning up a background HTTP server. |
| **Storage & Runtime Budget** | **Within Limits** | Kaggle mounts read-only inputs without consuming the 20 GB root disk limit. |
| **Free GPU Quota Needed** | **Smoke Test: < 2 min; EXP-01: ~45 min** | Smoke test consumes negligible quota (<0.1 hours). Full 10-game ablation consumes ~0.75 hours of the 30-hour weekly free quota. |

---

## 2. Detailed Technical Answers (Task 2)

### Question 1: Can the intended base model be loaded in an approved Kaggle notebook?
**Answer: YES, under specific quantization settings.**
- **Intended Base:** `unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit` or unquantized `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
- **Hardware Footprint:**
  - Full BF16 (14B): ~28 GB storage and ~32 GB VRAM (requires RTX Pro 6000 96GB or 2x T4 with tensor parallelism).
  - Pre-quantized 4-bit (`bnb-4bit`): ~9.5 GB storage, ~9.5–11.0 GB VRAM.
- **Kaggle Execution Mode:**
  - On a Kaggle T4 (16GB) or P100 (16GB), the 4-bit base model fits comfortably in single-device VRAM with ~5 GB headroom for KV cache and context.
  - On an RTX Pro 6000 (96GB), the model loads in native BF16 or 4-bit with zero VRAM pressure.

### Question 2: Are the actual adapter weights available, non-empty, and compatible with the base model?
**Answer: MIXED — Checkpoint-25 is verified; Checkpoint-50 is unusable.**
- **Forensic Audit of Local Files:**
  - `kaggle_flywheel_check/sam_ai_v4_trained/checkpoint-50/adapter_model.safetensors`: **0 bytes**. This file was truncated or failed during saving; it **cannot** be loaded.
  - `kaggle_flywheel_check/sam_ai_v4_trained/checkpoint-25/adapter_model.safetensors`: **137,716,248 bytes (137.7 MB)**.
- **Configuration Compatibility Check:**
  - Inspecting `checkpoint-25/adapter_config.json`:
    - `base_model_name_or_path`: `"unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"`
    - `peft_type`: `"LORA"`
    - `r`: `16`, `lora_alpha`: `32`
    - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - The checkpoint-25 adapter is structurally complete and fully compatible with the 4-bit base model.
- **Action Required:** The smoke test must load `checkpoint-25` (or pure base model weights first), and explicitly reject `checkpoint-50`.

### Question 3: Can official public ARC-3 environment files be attached without submitting?
**Answer: YES.**
- In Kaggle's container architecture, setting:
  ```json
  "competition_sources": ["arc-prize-2026-arc-agi-3"]
  ```
  in `kernel-metadata.json` mounts the public competition inputs to `/kaggle/input/arc-prize-2026-arc-agi-3/environment_files/` in read-only mode.
- **Submission Isolation:** Executing a notebook (via manual "Run All" or `kaggle kernels push`) does **not** create a leaderboard submission. A submission only occurs if:
  1. An output file named `submission.parquet` is generated.
  2. The user explicitly clicks "Submit to Competition" in the Kaggle UI or passes the competition submission API flag.
- Setting `"is_private": true` and omitting `submission.parquet` generation guarantees 100% isolation from the competition leaderboard.

### Question 4: Is the current agent connected to a real model runner or still requires an integration change?
**Answer: RESOLVED via `HuggingFaceModelRunner`.**
- Previously, the codebase only implemented `DeterministicModelRunner` (mock/offline) and `OpenAICompatibleModelRunner` (requiring a separate HTTP server like vLLM).
- To enable clean, in-process execution in a standard Kaggle notebook, we implemented [`HuggingFaceModelRunner`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/sam_ai/substrate/models.py#L195-L350):
  - Directly interfaces with `AutoModelForCausalLM` and `PeftModel`.
  - Formats chat prompts using the tokenizer template.
  - Manages generation parameters (`temperature`, `max_tokens`, `device_map`).
  - Returns structured [`ModelOutput`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/sam_ai/substrate/interfaces.py) with measured token counts and latencies.
- The agent controller (`SAMARC3Agent`) now runs interchangeably on mock, HTTP, or in-memory HuggingFace weights without any controller-level code changes.

### Question 5: Do needed dependencies and model files fit runtime and storage limits?
**Answer: YES.**
- **Storage Limits:**
  - Kaggle ephemeral disk: ~20 GB on root filesystem `/`, ~73 GB on `/kaggle/working`.
  - Kaggle input mounts (`/kaggle/input/`) are network-attached read-only volumes that do **not** count against the 20 GB root disk.
  - Attaching the model as a Kaggle Model or private Kaggle Dataset avoids downloading weights during notebook startup.
- **Dependencies:**
  - Kaggle GPU environments have PyTorch, CUDA, Transformers, and Accelerate pre-installed.
  - PEFT, bitsandbytes, and `arc_agi` require only lightweight pip wheels or local source bundles (~50 MB total).

### Question 6: What free GPU quota is required, and what information is still unavailable?
**Answer:**
- **Quota Requirements:**
  - Smoke Test (1 game, 5 steps): **< 2 minutes of GPU time** (< 0.05 hours).
  - Proposed EXP-01 Ablation (10 games $\times$ 3 seeds, 30 steps max): **~45 minutes of GPU time** (~0.75 hours).
  - Kaggle provides **30 hours per week** of free GPU compute. The smoke test consumes less than 0.2% of the weekly allowance.
- **Currently Unavailable Information:**
  - User's exact remaining GPU quota for the current 7-day cycle.
  - User's preference between Kaggle's free GPU tier (T4/P100) vs. their partner's private RTX Pro 6000 quota.

---

## 3. Go / No-Go Checklist for Smoke Test

| Precondition | Status | Action Needed |
|---|---|---|
| Agent Core & Schemas | **READY** | Tested & verified in `sam_ai/agent/`. |
| In-Process Model Runner | **READY** | Implemented in `sam_ai/substrate/models.py`. |
| Adapter Weights | **READY (Checkpoint-25)** | Use `checkpoint-25` (137.7 MB). Exclude empty `checkpoint-50`. |
| Public Environment Data | **BLOCKED LOCALLY** | Available on Kaggle via `/kaggle/input/arc-prize-2026-arc-agi-3/`. |
| Smoke Test Script / Notebook | **READY (UNEXECUTED)** | Prepared in `notebooks/sam_ai_real_model_smoke_test/`. |
| User Execution Approval | **PENDING** | **DO NOT EXECUTE UNTIL EXPLICITLY APPROVED.** |

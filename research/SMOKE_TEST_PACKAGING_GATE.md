# SAM-AI Real-Model Smoke Test: Packaging Gate & Audit Report

**Date:** 2026-10-10  
**Repository:** `micro_agi`  
**Current Git Target:** `master` (Post-Read-Only Packaging Verification)  
**Author:** Sovereign SAM-AI Systems Engineer & Security Auditor  
**Gate Status:** **NO-GO FOR REMOTE GPU EXECUTION (LOCKED PENDING USER AUTHORIZATION)**  

---

## 1. Executive Summary & Audit Decision

The final read-only validation of the packaging gate for the **SAM-AI Sovereign Real-Model Smoke Test** has concluded successfully.

### Hard Constraints Maintained
* **Zero remote uploads or execution:** No datasets have been uploaded, no Kaggle notebook versions were pushed, and no GPU quota was consumed.
* **Zero competition submissions:** The harness does not generate `submission.parquet` or invoke submission APIs.
* **Integrity of Golden Baseline & Evidence:** All baselines (`baselines/ARC3_GOLDEN_BASELINE/`) and logs remain intact.
* **Truthful Evaluation Metric:** Unit tests, static tests, and packaging steps verify software plumbing only. **Real-model evaluation is NOT marked complete** until authentic in-process model inference executes on real weights in an approved GPU runtime.
* **No Mock Fallback:** If base weights, adapters, or environment data are absent in the target environment, the execution halts cleanly with an explicit `[BLOCKED]` exit code.

### Gate Verdict: **NO-GO FOR REMOTE EXECUTION**
Remote execution on Kaggle remains strictly locked. We request explicit user authorization for Stage 1 (dataset creation) before proceeding. Stage 2 (kernel push & smoke test execution) will require a separate, independent approval.

---

## 2. Accessible Kaggle Base Model Verification

The PEFT LoRA adapter was fine-tuned on `unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`. We performed an empirical catalog query using the Kaggle API to verify accessible base models.

### Exact Accessible Kaggle Model Instance
* **Kaggle Model Slug:** `gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit`
* **Model ID:** `245083` | **Instance Slug:** `v1` | **Version:** `1`
* **Exact Instance Identifier:** `gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1`
* **Framework:** `transformers`
* **Upstream Hugging Face Commit:** `c73c785bf86971fa5bfd0912ca383a714e925210`
* **Quantization Format:** Pre-quantized 4-bit NormalFloat (`bitsandbytes` NF4)
* **Total Size:** `9,943,996,348 bytes` (~9.94 GB)
* **Status:** **CONFIRMED & READY** (Publicly accessible Kaggle Model)
* **Kaggle Mount Path:** `/kaggle/input/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1`

### Analysis of Alternative Models (Documented & Rejected)
To avoid silent substitution, we audited other candidate models in the Kaggle catalog:
1. `deepseek-ai/deepseek-r1` / `gowillgo/deepseek-ai-deepseek-r1-distill-qwen-14b` (BF16 Unquantized):
   * *Status:* **REJECTED.** Requires ~28 GB storage and ~32 GB VRAM. Exceeds standard single-GPU limits (T4/P100/L4) unless dynamically quantized at load time.
2. `konstantinboyko/deepseek-r1-distill-qwen-14b-awq-corianas` (AWQ Quantized):
   * *Status:* **REJECTED.** AWQ quantized weights are structurally incompatible with PEFT LoRA linear projection layers trained on bitsandbytes NF4 format.
3. `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (3.8B AutoRound):
   * *Status:* **REJECTED.** Different base model architecture and parameter count (3.8B vs 14B). LoRA adapter weights cannot attach.

---

## 3. Security, Secret Scanning & Dependency Audit of Source Bundle

The distribution bundle [`dist/sam_ai_source_bundle.tar.gz`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/dist/sam_ai_source_bundle.tar.gz) was audited using [`scripts/audit_bundle_security.py`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/scripts/audit_bundle_security.py):

* **File Size:** 106,160 bytes (~106 KB)
* **Archive Members:** 66 files/directories
* **SHA-256 Digest:** `af01de839def7c5e9d702bc17df921261ce2c05f9f4521d853d49c4a91282a6d`
* **Secret Scanning Result:** **0 VIOLATIONS DETECTED.**
  * Scanned against regex signatures for Hugging Face tokens (`hf_...`), OpenRouter keys, Kaggle keys/configs, OpenAI keys, Anthropic keys, GitHub tokens, AWS keys, generic private keys, and credential assignments.
* **File Cleanliness:** **0 FORBIDDEN FILES.**
  * Zero `.pyc` or `__pycache__` directories.
  * Zero dotfiles, hidden files, logs, database files, or scratch artifacts.
* **Package Dependencies:**
  * Standard Library (19 modules): `__future__`, `abc`, `ast`, `bisect`, `collections`, `copy`, `dataclasses`, `enum`, `json`, `math`, `os`, `pathlib`, `random`, `re`, `subprocess`, `sys`, `time`, `typing`, `urllib`.
  * External Packages (8 modules): `arcengine`, `torch`, `transformers`, `peft`, `numpy`, `sympy`, `huggingface_hub`, `z3`.
  * *Runtime Feasibility:* In Kaggle GPU environments, `torch`, `transformers`, `peft`, `numpy`, and `sympy` are pre-installed. `arcengine` and `arc_agi` are provided via offline competition wheels.
* **Licensing:**
  * SAM-AI source code is sovereign property of Parallax (Samrish B).
  * External dependencies follow open licenses (Apache 2.0 / MIT).

---

## 4. Reconfirmation of Checkpoint-25 vs. Checkpoint-50

Both local checkpoints were re-scanned and cryptographically reconfirmed:

### Checkpoint-25 (TARGET — VERIFIED NON-EMPTY)
* Path: `kaggle_flywheel_check/sam_ai_v4_trained/checkpoint-25/`
* `adapter_model.safetensors`: **137,716,248 bytes** (~137.7 MB)  
  **SHA-256:** `2fd12ef98cac8c25aee9b8bb6d8d011ddd15f45c538082b17c529b7d5ca14af3`
* `adapter_config.json`: **1,178 bytes**  
  **SHA-256:** `afb19e2b84e8d154c60c802b89c6b01cbf1187ddf51ef47cf345382bae7e38de`
* `README.md`: **2,428 bytes**  
  **SHA-256:** `bcdc47e8959dba91f3803dfbdb3c5af92ac9242f16885345aa7928b1562f0b12`
* **Status:** **INTEGRITY VERIFIED & READY FOR STAGING.**

### Checkpoint-50 (CORRUPT — UNUSABLE)
* Path: `kaggle_flywheel_check/sam_ai_v4_trained/checkpoint-50/`
* `adapter_model.safetensors`: **0 bytes**  
  **SHA-256:** `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty)
* **Status:** **PERMANENTLY MARKED AS UNUSABLE.**
* Guard Verification: [`HuggingFaceModelRunner.load_model()`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/sam_ai/substrate/models.py#L248-L300) explicitly validates total weight bytes $> 0$ and raises `ValueError` if a 0-byte adapter is selected.

---

## 5. Final Kaggle Input Manifest

| Component | Intended Source Slug / ID | Kaggle Mounted Path | Format / Version | Size / Verification | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model Weights** | `gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1` | `/kaggle/input/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1` | 4-bit NF4 (`bitsandbytes`) Safetensors | 9,943,996,348 bytes (~9.94 GB)<br>Upstream: `c73c785bf86971fa5bfd0912ca383a714e925210` | **CONFIRMED & READY** |
| **LoRA Adapter Weights** | `samrishb/sam-ai-v4-checkpoint-25-adapter` | `/kaggle/input/sam-ai-v4-checkpoint-25-adapter` | PEFT LoRA safetensors ($r=16, \alpha=32$) | 137,716,248 bytes (~137.7 MB)<br>**SHA256:** `2fd12ef98cac8c25aee9b8bb6d8d011ddd15f45c538082b17c529b7d5ca14af3` | **STAGED LOCALLY**<br>*(Pending Upload)* |
| **SAM-AI Source Code** | `samrishb/sam-ai-source-bundle` | `/kaggle/input/sam-ai-source-bundle/sam_ai_source_bundle.tar.gz` | Clean tarball (0 pycache) | 106,160 bytes (~106 KB)<br>**SHA256:** `af01de839def7c5e9d702bc17df921261ce2c05f9f4521d853d49c4a91282a6d` | **STAGED LOCALLY**<br>*(Pending Upload)* |
| **ARC-3 Environment Data** | `arc-prize-2026-arc-agi-3` | `/kaggle/input/arc-prize-2026-arc-agi-3/environment_files` | Official JSON environments | 110 public games | **CONFIRMED** |
| **ARC-3 Engine Wheels** | `arc-prize-2026-arc-agi-3` | `/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels` | Python Wheels (`arc_agi`, `arcengine`) | `.whl` binaries | **CONFIRMED** |

---

## 6. Staged Dataset Creation Commands & Metadata

The two private dataset folders have been prepared and verified locally via [`scripts/stage_kaggle_datasets.py`](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi/scripts/stage_kaggle_datasets.py):

### Dataset 1: `samrishb/sam-ai-v4-checkpoint-25-adapter`
* **Local Folder:** `kaggle_datasets/sam_ai_v4_checkpoint_25_adapter/`
* **Contents:**
  * `adapter_model.safetensors` (137,716,248 B)
  * `adapter_config.json` (1,178 B)
  * `README.md` (2,428 B)
  * `dataset-metadata.json` (175 B)
* **Metadata Schema:**
  ```json
  {
    "title": "SAM-AI v4 LoRA Adapter (Checkpoint-25)",
    "id": "samrishb/sam-ai-v4-checkpoint-25-adapter",
    "licenses": [{"name": "apache-2.0"}]
  }
  ```
* **Command to Execute (Once Approved):**
  ```bash
  kaggle datasets create -p kaggle_datasets/sam_ai_v4_checkpoint_25_adapter
  ```

### Dataset 2: `samrishb/sam-ai-source-bundle`
* **Local Folder:** `kaggle_datasets/sam_ai_source_bundle/`
* **Contents:**
  * `sam_ai_source_bundle.tar.gz` (106,160 B)
  * `dataset-metadata.json` (156 B)
* **Metadata Schema:**
  ```json
  {
    "title": "SAM-AI Sovereign Source Bundle",
    "id": "samrishb/sam-ai-source-bundle",
    "licenses": [{"name": "apache-2.0"}]
  }
  ```
* **Command to Execute (Once Approved):**
  ```bash
  kaggle datasets create -p kaggle_datasets/sam_ai_source_bundle
  ```

---

## 7. Next Steps & Approval Protocol

1. **STAGE 1 APPROVAL (CURRENT GATE):**  
   We request your explicit authorization to execute the two dataset creation commands listed above.
2. **STAGE 2 APPROVAL (FUTURE GATE):**  
   Once the two datasets are confirmed active and mounted on Kaggle, we will present the updated `kernel-metadata.json` and request a **separate, independent authorization** before pushing the notebook and executing the 5-step smoke test.

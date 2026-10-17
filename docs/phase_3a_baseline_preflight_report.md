# SAM-AI Phase 3A: Multi-Domain Intelligence Baseline Preflight Report

**Document Version:** 1.0.0  
**Date:** October 17, 2026  
**Status:** COMPLETE (STOP GATE REACHED — READY FOR REVIEW)  
**Target Environment:** Kaggle GPU (NVIDIA Tesla T4 x 2 or P100)  
**Foundation Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Base Only)  
**Adapter State:** **STRICTLY NONE** (Experiment 001 adapter is rejected; no LoRA attached)  

---

## 1. Executive Summary & Objective

In accordance with the **SAM-AI Phase 3 Roadmap**, Phase 3A prepares a rigorous, reproducible, multi-domain evaluation of the untouched base foundation model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) before any further fine-tuning or reinforcement learning (e.g., Experiment 002).

In the historical Experiment 001, the untouched base model achieved **19/20 (95.0%)** on the 20-sample GSM8K pilot, while the candidate GSM8K adapter degraded performance to **16/20 (80.0%)** through reasoning-chain collapse. The historical artifacts have been audited and proven reproducible. To prevent regressions across non-mathematical capabilities, this Phase 3A benchmark expands evaluation to **95 frozen challenge samples across 7 distinct cognitive domains**.

> [!IMPORTANT]
> **Strict Stop Gate Enforced:** No remote Kaggle session has been launched, no cloud GPU compute has been consumed, and no production services have been altered. This preflight report, self-contained notebook, frozen fixtures, and offline test suite are prepared for user inspection and authorization.

---

## 2. Benchmark Architecture & Fixtures

### 2.1 Frozen Multi-Domain Dataset (`curriculum/phase_3a_baseline_fixtures.json`)
- **Total Domains:** 7
- **Total Challenge Samples:** 95
- **Fixtures SHA-256:** `b7fc8795fa64c82a97a871efaee68491fc7aa93f3bb557a31197b710892caf3d`
- **Configuration Path:** `curriculum/phase_3a_baseline_config.json`

| Domain Identifier | Sample Count | Primary Evaluation Source | Metric / Verification Method | Target Output Format |
|---|---|---|---|---|
| `mathematics_gsm8k` | 20 | GSM8K Validation Pilot (`gsm8k_val_pilot_00` to `pilot_19`) | Exact numeric matching with currency/comma cleanup | `#### <num>` or `\boxed{<num>}` |
| `mathematics_math500` | 20 | Competition Math (Algebra, Number Theory, Geometry, Probability, Precalculus) | Symbolic & fractional equivalence with tolerance | LaTeX `\boxed{...}` or fraction `A/B` |
| `coding_humaneval` | 20 | HumanEval Functional Coding (`humaneval_00` to `humaneval_19`) | Isolated subprocess execution with 3.0s timeout and assertion test suites | Executable Python function matching entry point |
| `science_reasoning` | 10 | Physics, Chemistry, Biology quantitative challenges | Exact numeric / tolerance comparison | Physical quantities, molar mass, pH, kinematics |
| `logic_constraints` | 10 | Knights & Knaves, syllogisms, relative order, logic puzzles | Text containment & logical exact match | Boolean, agent identity, middle object |
| `instruction_following` | 10 | Strict constraint compliance (JSON schema, word limits, letter bans, format) | Schema validation, word count, character filters, line breaks | Valid JSON, exactly 6 words, no 'e', ALL CAPS, 3 bullets |
| `multiturn_context` | 5 (10 turns) | Multi-turn conversational memory and attribute retention | Context target extraction across multi-turn dialog | Entity and attribute recall in Turn 2 |
| **TOTAL** | **95** | **Comprehensive Frontier Multi-Domain Benchmark** | **Domain-Specific Verifiers & Sandboxing** | **Deterministic Evaluation** |

---

## 3. Kaggle Execution Package & Notebook Design

### 3.1 Kernel Location & Artifacts
- **Notebook File:** `notebooks/sam_ai_multidomain_baseline/sam_ai_multidomain_baseline.ipynb`
- **Kernel Metadata:** `notebooks/sam_ai_multidomain_baseline/kernel-metadata.json`
  - Kernel Slug: `samrishb/sam-ai-multidomain-baseline`
  - Accelerator: GPU Enabled (Tesla T4 x 2 or P100)
  - Internet: Enabled (for Hugging Face model weight fetching)
  - Kernel Sources: `[]` (Guarantees zero upstream adapter artifacts are linked)
  - Dataset Sources: `["samrishb/sam-ai-source-bundle"]`

### 3.2 Cell Structure & Safety Guardrails

```
Cell 0 [Markdown]: Benchmark Protocol, Guarantees, Domain Breakdown
  │
Cell 1 [Code]: Hardware & Storage Preflight Audit (Fail-Closed)
  ├─ Validates CUDA availability
  ├─ Checks Cluster VRAM (Requires >= 14.0 GB)
  ├─ Checks Disk Space (Requires >= 10.0 GB free)
  └─ Pip installs transformers>=4.45.0, accelerate>=0.34.0, bitsandbytes>=0.43.0
  │
Cell 2 [Code]: Benchmark Fixtures & Domain Extraction / Verification Engine
  ├─ Inlines all 95 samples with SHA-256 integrity verification
  ├─ extract_boxed_content: recursive brace matcher supporting nested LaTeX
  ├─ extract_numerical_answer: ####, \boxed{}, natural phrases, trailing tokens
  ├─ extract_symbolic_math: fractions, signs, and algebraic forms
  ├─ execute_humaneval_in_sandbox: isolated subprocess with 3.0s timeout
  ├─ verify_instruction_constraint: JSON schema, word limits, character bans
  ├─ verify_multiturn_context: multi-turn entity recall
  └─ verify_prediction: master dispatcher
  │
Cell 3 [Code]: Fail-Closed Base Model Loader (DeepSeek-R1-Distill-Qwen-7B)
  ├─ Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  ├─ Quantization: 4-bit NF4 (compute_dtype=float16) or sharded float16
  ├─ Anti-Adapter Assertion 1: assert not hasattr(base_model, "peft_config")
  ├─ Anti-Adapter Assertion 2: assert "PeftModel" not in type(base_model).__name__
  └─ Sets base_model.eval()
  │
Cell 4 [Code]: Resumable Execution Loop with Streaming Persistence
  ├─ Checkpoint: /kaggle/working/predictions.jsonl
  ├─ Resumption: Scans completed example_ids and skips on restart
  ├─ Streaming Persistence: flushes each record immediately to disk
  ├─ Decoding: greedy (temperature=0.0, do_sample=False, max_new_tokens=2048)
  └─ No artificial truncation: allows full <think>...</think> reasoning traces
  │
Cell 5 [Code]: Aggregation, Summary Report & Artifact Packaging
  ├─ Computes per-domain and aggregate benchmark metrics
  ├─ Generates /kaggle/working/baseline_multidomain_report.json
  ├─ Compresses results into /kaggle/working/sam_ai_baseline_multidomain_results.zip
  └─ Prints clean ASCII summary table
```

---

## 4. Hardware, Runtime & Memory Footprint

### 4.1 Resource Budgeting on Kaggle Free-Tier GPU

| Parameter | Specification | Preflight Budget / Safety Margin |
|---|---|---|
| **GPU Target** | 2x NVIDIA Tesla T4 (15.0 GB each = 30.0 GB total) | CC 7.5 (Turing) |
| **Model Footprint** | ~7.6 Billion Parameters | 4-bit NF4: ~4.5 GB VRAM; Native FP16: ~15.2 GB sharded |
| **Context & Generation** | Input: ~100–300 tokens; Output: up to 2048 tokens | KV Cache: ~1.5 GB per GPU |
| **Peak VRAM Expected** | ~6.5 GB on GPU 0, ~5.5 GB on GPU 1 (4-bit NF4) | Well below 13.0 GB per-device allocation limit |
| **Disk Space Needed** | Model cache (~15 GB) + working files (< 50 MB) | Requires >= 10 GB free in `/kaggle/working` |
| **Per-Sample Latency** | ~15–35 seconds per reasoning generation (at ~30–45 tok/s) | ~25s average across 95 items |
| **Estimated Total Runtime** | $95 \text{ items} \times 25\text{s} \approx 2,375\text{s} \approx 40\text{ minutes}$ | Well within Kaggle's 9-hour (540 min) execution cap |

---

## 5. Resumption, Error Recovery & Sandboxing

1. **Incremental Streaming Persistence:**
   - Every inference record is appended to `/kaggle/working/predictions.jsonl` and flushed immediately.
   - If a Kaggle session disconnects, times out, or restarts, Cell 4 reads all completed `example_id` entries and resumes exactly at the next unfinished sample.
2. **Coding Sandbox Security:**
   - Model-generated Python code for HumanEval is written to a temporary file and executed in an isolated child process (`subprocess.run`).
   - A strict **3.0-second timeout** is enforced. Infinite loops, hangs, or blocking operations are aborted without impacting the evaluation runner.
   - The sandbox runs with offline environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), preventing unauthorized external socket connections.
3. **Fail-Closed Anti-Adapter Gate:**
   - Explicit runtime assertions verify that no LoRA adapter or `PeftModel` wrapper is present. Any attempt to mount adapter weights triggers an immediate fatal exception.

---

## 6. Local Test & Validation Evidence

All components have been verified locally in the workspace:

1. **Fixtures Generator (`scripts/build_phase_3a_fixtures.py`):**
   - Produced `curriculum/phase_3a_baseline_fixtures.json` (95 samples, 7 domains).
   - Produced `curriculum/phase_3a_baseline_config.json`.
   - Verified fixtures SHA-256: `b7fc8795fa64c82a97a871efaee68491fc7aa93f3bb557a31197b710892caf3d`.
2. **Notebook Syntax & Static Analysis (`scripts/verify_multidomain_notebook_syntax.py`):**
   - Validated `notebooks/sam_ai_multidomain_baseline/sam_ai_multidomain_baseline.ipynb`.
   - All 5 code cells (747 lines total) compile cleanly with `compile(..., "exec")`.
   - Confirmed presence of hardware fail-closed checks, sandbox timeouts, streaming persistence, and base-model assertions.
3. **Multi-Domain Unit Test Suite (`tests/test_multidomain_baseline.py`):**
   - Executed and passed all 13 test cases:
     - `test_clean_numeric_str` [PASS]
     - `test_extract_numerical_answer` [PASS]
     - `test_extract_symbolic_math` (including nested LaTeX braces and negative signs) [PASS]
     - `test_compare_math_answers` (exact, float tolerance, fractional equivalence) [PASS]
     - `test_extract_code_block` [PASS]
     - `test_execute_humaneval_in_sandbox_success` [PASS]
     - `test_execute_humaneval_in_sandbox_assertion_failure` [PASS]
     - `test_execute_humaneval_in_sandbox_syntax_error` [PASS]
     - `test_execute_humaneval_in_sandbox_timeout` [PASS]
     - `test_verify_instruction_constraint` (all 10 types) [PASS]
     - `test_verify_multiturn_context` [PASS]
     - `test_fixtures_integrity` [PASS]
     - `test_resumability_checkpoint_loop` [PASS]

---

## 7. Risks, Limitations & Mitigation Strategies

| Risk | Impact | Mitigation Strategy Implemented |
|---|---|---|
| **GPU Out of Memory (OOM)** | Medium | 4-bit NF4 quantization configured with float16 compute dtype; fallback to sharded float16 with explicit `max_memory={0: '13.0GB', 1: '13.0GB'}`. |
| **Model Reasoning Truncation** | High | `max_new_tokens` set to 2048 with greedy decoding (`temperature=0.0`); `is_truncated` flag tracks any instances reaching the token limit. |
| **Network or Container Interruption** | Medium | Incremental JSONL streaming to `/kaggle/working/predictions.jsonl` with automatic resumption skipping completed IDs. |
| **Malicious or Broken Code Execution** | High | Subprocess isolation with strict 3.0s timeout, isolated environment, and capture of `AssertionError` vs `RuntimeError`. |
| **Adapter Contamination** | Critical | Explicit assertions in Cell 3 fail closed if `peft_config` or `PeftModel` is detected; kernel metadata has `kernel_sources=[]`. |

---

## 8. Stop Gate & Authorization Command

> [!NOTE]
> **Current Status:** STOP GATE REACHED. No remote execution has been started.

When the user authorizes remote GPU execution, the following sequence should be initiated:

1. **Push Kernel to Kaggle:**
   ```bash
   kaggle kernels push -p notebooks/sam_ai_multidomain_baseline
   ```
2. **Monitor Execution:**
   ```bash
   kaggle kernels status samrishb/sam-ai-multidomain-baseline
   ```
3. **Retrieve Benchmark Artifacts Upon Completion:**
   ```bash
   kaggle kernels output samrishb/sam-ai-multidomain-baseline -p evidence/phase_3a_baseline/
   ```

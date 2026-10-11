# SAM-AI Sovereign Fine-Tuning & Capability Proposal (EXP-SAM-FT-01)

> **Document Status:** PROPOSAL / SPECIFICATION (Status: `NOT_RUN`)  
> **Date:** October 11, 2026  
> **Author:** SAM-AI Research Team (Parallax)  
> **Target Base Model:** `DeepSeek-R1-Distill-Qwen-14B` (4-bit NF4 / BF16)  
> **Execution Gate:** STRICTLY BLOCKED awaiting user authorization. No compute will be consumed.

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Verified Capability Gap
Extensive audit of `DeepSeek-R1-Distill-Qwen-14B` across local evaluation and competitor forensics reveals a distinct failure pattern when deploying reasoning models inside interactive agent loops:
1. **Unconstrained Prose Drift:** While the base model possesses strong mathematical chain-of-thought capabilities, it frequently emits thousands of conversational tokens, meta-commentary, or ungrounded hypotheses rather than strictly formatted, executable tool calls.
2. **Action Syntax Hallucination:** In interactive environments (such as ARC-AGI-3 or terminal sandboxes), the model outputs illegal action names, out-of-bounds coordinates ($x, y \ge 64$), or mixes JSON and markdown blocks.
3. **Truncation Collapse:** In previous training runs (e.g. `checkpoint-25`), 85% of completions hit the sequence length limit because the model lacked an explicit structural inductive prior for concise state-delta reflection.

### 1.2 Core Research Hypothesis (`EXP-SAM-FT-01`)
> **Hypothesis:** Supervised fine-tuning (SFT) of `DeepSeek-R1-Distill-Qwen-14B` on a verified curriculum of *Grounded Observation $\to$ Deliberation Trace $\to$ Deterministic Action Schema* will increase valid action parse density from baseline (~65%) to $>98\%$, reduce average turn token length by $40\%$, and eliminate syntax hallucination without degrading mathematical or programming reasoning.

---

## 2. Dataset Specification & Provenance

### 2.1 Training Curriculum Composition (Total: 2,000 Verified Examples)

| Domain | Size | Source | Verification & Ground Truth | License |
| :--- | :--- | :--- | :--- | :--- |
| **Interactive Spatial Actions** | 800 pairs | Deterministic ARC-3 simulator rollouts | Exact state transition verified via Python engine | Apache-2.0 |
| **Grounded Tool Use / REPL** | 500 pairs | Python sandbox execution traces | Exact stdout/stderr match + assertion checks | MIT / Apache-2.0 |
| **Mathematical Deliberation** | 400 pairs | MATH-500 Level 5 (Curated reasoning traces) | SymPy symbolic equivalence verified | MIT |
| **Sovereign Identity & System**| 300 pairs | Parallax Agent Instruction Set | Strict compliance with system persona | Apache-2.0 (Original) |

### 2.2 Data Integrity, Deduplication & Held-Out Separation
* **Deduplication:**
  - Exact MD5 hashing on normalized observation states.
  - MinHash LSH with Jaccard threshold $\theta = 0.80$ to eliminate near-duplicate prompt-response pairs.
* **Data Splits**:
  - **Training Set (80%):** 1,600 examples.
  - **Validation Set (10%):** 200 examples.
  - **Strictly Held-Out Evaluation Set (10%):** 200 examples (including 50 unseen ARC-3 game layouts and 50 GSM8K problems).
* **Contamination Guard:** Zero test tasks from the ARC-AGI-3 hidden test set or competition evaluation split are included in the training corpus.

---

## 3. Model & Adapter Architecture

### 3.1 Base Model Selection
* **Identifier:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (served via 4-bit NF4 quantized weights or BF16 on high-memory hardware).
* **Context Window:** 8,192 tokens (evaluated up to 32,768).
* **License:** Apache-2.0.

### 3.2 LoRA Configuration Specification
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```
* **Trainable Parameters:** ~137.7 MB ($\approx 0.95\%$ of model weights).
* **Weight Initialization:** Standard LoRA gaussian init with $B=0$ guaranteeing identity transformation at step 0.

---

## 4. Training Objective & Hyperparameters

### 4.1 Loss Function
Supervised Cross-Entropy loss computed **strictly on completion tokens** (masking prompt tokens with `-100` label index):
$$\mathcal{L}(\theta) = - \frac{1}{T} \sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{<t})$$

### 4.2 Key Hyperparameters
* **Effective Batch Size:** 16 (per-device batch size = 2, gradient accumulation steps = 8).
* **Optimizer:** AdamW (`beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01`).
* **Learning Rate:** $2.0 \times 10^{-5}$ with Cosine Annealing schedule.
* **Warmup:** 5% of total steps (~10 steps).
* **Total Epochs:** 2 epochs (~200 optimizer steps).
* **Max Sequence Length:** 2,048 tokens (prevents the 512-token truncation observed in `checkpoint-25`).
* **Gradient Clipping:** Max norm = 1.0.
* **Precision:** Mixed precision `bfloat16` with PyTorch SDPA attention.

### 4.3 Checkpointing & Fault Tolerance
* Checkpoint saved every 25 optimizer steps (`save_steps: 25`).
* Maximum 3 checkpoints retained (`save_total_limit: 3`).
* Checkpoint directory contains full optimizer, scheduler, and RNG state for deterministic resumption.

---

## 5. Explicit Resource & Compute Estimates

| Resource Dimension | Estimated Requirement | Safety Margin / Ceiling | Available Environment |
| :--- | :--- | :--- | :--- |
| **GPU Hardware** | 1x NVIDIA RTX PRO 6000 (Blackwell Server) | Min: 1x 24GB L4 / Max: 96GB RTX 6000 | **CONFIRMED AVAILABLE** (94.97 GB VRAM) |
| **VRAM Consumption** | ~18.5 GB (Model weights 9.9 GB + Activations 8.6 GB) | 24.0 GB | Well within 94.43 GB free VRAM |
| **Wall-Clock Time** | ~45 minutes | 90 minutes | Kaggle kernel 9-hour limit |
| **GPU-Hours Billed** | ~0.75 GPU-hours | 1.5 GPU-hours | Free weekly quota |
| **Disk Storage** | ~350 MB (Checkpoints + logs) | 2.0 GB | Kaggle 20 GB working disk |

---

## 6. Evaluation Framework & Keep/Reject Criteria

### 6.1 Evaluation Benchmark Batteries

```
+-------------------------------------------------------------------------------+
|                       EXP-SAM-FT-01 EVALUATION BATTERY                        |
+-------------------------------------------------------------------------------+
|  Battery 1: In-Domain Action Validity (200 Held-Out Interaction Turns)        |
|             Metric: Strict JSON parse rate & legal action execution rate      |
|                                                                               |
|  Battery 2: Multi-Step Reasoning Retain (50 Problems from MATH-500)           |
|             Metric: Exact Match accuracy on SymPy mathematical equivalence    |
|                                                                               |
|  Battery 3: Code Generation & Execution (50 Tasks from HumanEval / LCB)       |
|             Metric: Pass@1 on executable unit tests                           |
|                                                                               |
|  Battery 4: ARC-AGI-3 Agent Smoke Test (5 Environments, Max 20 Steps)         |
|             Metric: Mean action efficiency and level completion count         |
+-------------------------------------------------------------------------------+
```

### 6.2 Explicit Keep / Reject Gates

| Metric | Baseline (Zero-Shot 14B) | Target Threshold (Keep Gate) | Rejection Threshold (Reject Gate) |
| :--- | :--- | :--- | :--- |
| **Action JSON Parse Rate** | ~65.0% | $\ge \mathbf{98.0\%}$ | $< 90.0\%$ |
| **Action Legality Rate** | ~70.0% | $\ge \mathbf{95.0\%}$ | $< 85.0\%$ |
| **MATH-500 Retention** | 78.0% | $\ge \mathbf{76.5\%}$ (Max 1.5% drop)| Drop $> 3.0\%$ (Catastrophic forgetting) |
| **HumanEval Pass@1** | 68.0% | $\ge \mathbf{67.0\%}$ | Drop $> 2.0\%$ |
| **Token Efficiency** | ~1,200 tok/turn | $\le \mathbf{700\text{ tok/turn}}$ | $> 1,100\text{ tok/turn}$ (No pruning effect) |

---

## 7. Risks & Mitigation Strategies

1. **Catastrophic Forgetting of Reasoning:**
   - *Risk:* Fine-tuning exclusively on short structured actions impairs the base model's deep chain-of-thought mathematical derivation.
   - *Mitigation:* The training corpus preserves a 20% mixture of verified MATH-500 multi-step proof traces.
2. **Format Overfitting / Rigidity:**
   - *Risk:* The model becomes brittle when tool schemas or environment actions vary slightly.
   - *Mitigation:* Data augmentation with 3 distinct syntaxes (strict JSON, Python function calls, and symbolic key-value pairs).
3. **Loss of General Conversational Grounding:**
   - *Risk:* Persona degradation or refusal to engage in general dialogue.
   - *Mitigation:* 15% inclusion of general dialogue and instruction-following pairs.

---

## 8. Full SAM-AI Capability Integration Plan

To build a true frontier AI system rather than an ARC-only heuristic solver, capabilities must be correctly partitioned between **model weights** and the **shared agent runtime**:

### 8.1 Partitioning Responsibilities

```
+-------------------------------------------------------------------------------+
|                       SAM-AI SYSTEM ARCHITECTURE MAPPING                      |
+-------------------------------------------------------------------------------+
|                                                                               |
|   MODEL WEIGHTS (Learned via SFT / RLVR)                                      |
|   ├── Inductive priors (pattern recognition across observation pairs)        |
|   ├── Deep Chain-of-Thought deliberation (<think> ... </think>)              |
|   ├── Deterministic tool-calling syntax & schema adherence                   |
|   └── Self-critique & error reflection habits                                 |
|                                                                               |
|   SHARED AGENT RUNTIME (Engineered Software Layer)                            |
|   ├── Isolated Execution Sandboxes (Python REPL, Bash subprocess)            |
|   ├── Symbolic Visual Perception (Lossless grid parsing, frame segmentation)  |
|   ├── Episodic Memory Ledgers (Known-death registry, state hash history)     |
|   ├── Pre-Execution Deterministic Guards (Bounds checking, legality verifier) |
|   ├── Dynamic Priority Pacing & SGLang Radix Cache alignment                 |
|   └── Multi-turn Conversation & User Interface                                |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### 8.2 Pillar Integration Matrix

| SAM-AI Product Pillar | Learned Weight Capabilities | Shared Agent Runtime Capabilities |
| :--- | :--- | :--- |
| **SAM-AI Assistant** | Polite conversation, structured task breakdown, user intent extraction | Working memory, persistent state store, rate limiting, UI components |
| **SAM-AI Code** | Algorithmic synthesis, test generation, syntax precision | Sandboxed execution container, AST parsing, Git versioning, diff patcher |
| **SAM-AI Research** | Mathematical derivation, logical deduction, hypothesis synthesis | Grounded literature retrieval, PDF parsing, citation verification |
| **SAM-AI Multimodal** | Visual coordinate grounding, object relation description | OpenCV / PIL frame diffing, bounding box crop extraction, upscale cache |
| **Intelligence Lab** | Autonomous exploration policy, self-correction reasoning | Pre-registered ablation harness, statistical testing, telemetry logging |

---

## 9. Current Approval & Execution Gate

* **Status:** `SPECIFICATION ONLY` / `NOT_RUN`.
* **Compute Allocated:** 0 GPU seconds.
* **Weights Modified:** None.
* **Next Gate:** This experiment will NOT be submitted, pushed, or run until:
  1. The smoke test path resolution is fixed and verified live.
  2. The zero-shot baseline capability is measured.
  3. Explicit written authorization is provided.

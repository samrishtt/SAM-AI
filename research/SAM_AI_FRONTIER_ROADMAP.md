# SAM-AI Frontier Intelligence Roadmap

> **Author:** SAM-AI Research Team (Parallax)  
> **Date:** October 10, 2026  
> **Status:** ARCHITECTURAL SPECIFICATION & LONG-TERM RESEARCH PLAN  
> **Guiding Principle:** Truthful evidence over inflated metrics. Frontier capabilities cannot be faked with heuristic wrappers, speculative claims, or test-time evaluation hacks. Progress must be validated through reproducible benchmarks, honest baselines, and clear attribution.

---

## Executive Summary & Foundational Posture

SAM-AI is designed as an autonomous, sovereign agent capable of fluid reasoning, inductive abstraction, tool use, and continuous learning. 

Recent forensic audits established a critical distinction in the codebase:
1. **Third-Party Solvers (e.g. B2 / Franzen Milestone 2):** High-throughput specialized interactive pipelines running external quantized models (Intel Qwen 3.8B + Albucino drafter) on SGLang. While valuable for empirical study and competition reproducibility, they do **not** represent SAM-AI's own reasoning weights or sovereign architecture.
2. **SAM-AI Sovereign Architecture (`sam_ai.agent` & `sam_ai.substrate`):** An independent, decoupled agent system integrating strict action schemas, lossless symbolic visual representations, task-specific failure memory, domain-specific verifiers, and modular foundation model runners.

This roadmap outlines the eight sequential research milestones required to advance SAM-AI from a clean local scaffold to a competitive frontier-grade reasoning agent. Long-term milestones are strictly non-active research tracks; each milestone requires rigorous empirical proof before the subsequent phase may begin.

```
+----------------------------------------------------------------------------------------------------+
|                                    SAM-AI RESEARCH MILESTONE LADDER                                |
+----------------------------------------------------------------------------------------------------+
|  M1: Truthful Model Execution & Evaluation (Local/Remote Provenance & Strict Evaluation)          |
|                                         |                                                          |
|                                         v                                                          |
|  M2: Formal Reasoning & Dynamic Verification (Pre-Execution Guard & Test-Time Search)               |
|                                         |                                                          |
|                                         v                                                          |
|  M3: Learning from Failures (Curriculum Synthesis, RLVR / GRPO with Verifiable Rewards)             |
|                                         |                                                          |
|                                         v                                                          |
|  M4: Episodic Memory & World-State Tracking (Provenance Graphs & Latent Transition Models)         |
|                                         |                                                          |
|                                         v                                                          |
|  M5: Tool-Using Autonomous Agents (Sandboxed REPL, System Actuation & Error Recovery)              |
|                                         |                                                          |
|                                         v                                                          |
|  M6: Cross-Domain Generalization (Math, Competitive Code, Formal Logic, Interactive Science)      |
|                                         |                                                          |
|                                         v                                                          |
|  M7: Multimodal Grounding (Modular Pretrained Vision/Audio Ingestion & Spatial Coordinate Binding)  |
|                                         |                                                          |
|                                         v                                                          |
|  M8: Frontier Scaling & Compute Realities (Pretraining Physics, Cluster Scaling, Post-Training)    |
+----------------------------------------------------------------------------------------------------+
```

---

## Milestone 1: Reliable Model Execution and Truthful Evaluation

### 1.1 Research Hypothesis
An agent cannot improve if its evaluation harness leaks benchmark data, conflates multi-pass maximums with live single-pass capability, or misattributes model outputs. Decoupling the controller from the model runner and recording cryptographic configuration hashes, exact git commits, random seeds, and raw step-by-step traces will eliminate evaluation drift and establish a dependable baseline ($N=1$).

### 1.2 Required Implementation
- Complete instrumentation of `sam_ai.agent.arc3_agent` and `scripts/evaluate_sam_ai_agent.py`.
- Automated detection of runtime environment: CPU local testing must fail cleanly or run in explicit `TEST_ONLY` mode; GPU runs must verify SHA256 hashes of base model weights and LoRA adapters.
- Machine-readable raw evidence logging (`runs_raw.jsonl`) recording prompt, completion, parsed action, verification outcome, environment state delta ($\Delta$), reward, and per-step latency.
- Aggregation engine reporting mean, median, standard deviation, and 95% bootstrap confidence intervals across single-pass runs ($N=1$).

### 1.3 Evaluation Benchmark
- **Test-Bed:** 10 offline ARC-AGI-3 environments (5 development, 5 held-out).
- **Control:** Zero-shot unconstrained base model (`unsloth/DeepSeek-R1-Distill-Qwen-14B` or `Qwen/Qwen2.5-Coder-14B-Instruct`).
- **Telemetry Verification:** Automated schema validation of raw logs verifying 100% trace completeness.

### 1.4 Success Criteria
- Zero silent fallbacks or unrecorded failures across 30 consecutive episodes.
- Re-running evaluation on identical seeds yields zero variance in action trajectories ($\Delta \text{score} = 0.0$).
- Raw log artifacts enable exact reconstruction of all summary tables.

### 1.5 Compute and Data Prerequisites
- **Compute:** 1x GPU with $\ge 24\text{ GB}$ VRAM (NVIDIA L4 or RTX Pro 6000) for FP8/4-bit 14B inference.
- **Data:** 10 official offline ARC-AGI-3 environment metadata files (`metadata.json`).
- **Dependencies:** `vllm` or `sglang` offline engine, `transformers`, `torch`.

### 1.6 Major Risks
- GPU memory exhaustion during long-horizon context accumulation (>16k tokens).
- Rate limits or network interruptions if calling external inference endpoints.

### 1.7 Evidence Required Before Proceeding
- A committed `summary.json` and `runs_raw.jsonl` from a 30-episode run on genuine 14B weights, with zero missing fields and explicit seed reproducibility confirmed.

---

## Milestone 2: Reasoning and Verification

### 2.1 Research Hypothesis
Unconstrained autoregressive generation frequently outputs illegal actions, syntax violations, or hallucinations when faced with complex game boards. Introducing structured multi-stage deliberation (Chain-of-Thought $\to$ Plan $\to$ Action) coupled with a deterministic pre-execution verification guard will increase valid action density and eliminate wasted exploratory steps.

### 2.2 Required Implementation
- **Deliberation Prompting:** Two-stage generation prompting the model to produce `<think>` (spatial hypotheses, object tracking) followed by `<plan>` (short 2-step intended path) and `<action>` (strictly formatted JSON).
- **Pre-Execution Verifier:**
  - Semantic legality verification (e.g. enforcing `RESET` during `GAME_OVER`).
  - Coordinate bounding ($0 \le x, y < 64$).
  - Fast simulation check: If an action can be proven locally to produce a zero-delta without changing state, request an immediate local reflection before stepping the environment.
- **Calibrated Verification Retries:** Bounded reflection loop (maximum 2 retries) with structured diagnostic feedback.

### 2.3 Evaluation Benchmark
- **Ablation Comparison:** Pre-registered `EXP-SAM-01-SCAFFOLDING-EVAL` (Arm A: Unconstrained ReAct vs. Arm B: Deliberation + Verification Guard).
- **Environment:** 10 ARC-AGI-3 offline games $\times$ 3 seeds (30 episodes per arm).

### 2.4 Success Criteria
- $\ge 30\%$ relative reduction in zero-delta (no-op/wall bump) actions ($p < 0.05$).
- Zero illegal/out-of-bounds actions reaching the environment.
- Single-pass win rate ($N=1$) equal to or greater than Arm A baseline.

### 2.5 Compute and Data Prerequisites
- Identical to Milestone 1 (1x 24GB GPU).
- Pre-registered experiment protocol approved by user.

### 2.6 Major Risks
- Model gets caught in degenerate retry loops if the verifier rejects plausible but imperfect actions.
- Verification feedback prompt exceeds context budget.

### 2.7 Evidence Required Before Proceeding
- Completed statistical analysis of `EXP-SAM-01-SCAFFOLDING-EVAL` showing statistically significant gains in action efficiency without regression on held-out tasks.

---

## Milestone 3: Learning from Failures (Curriculum & Verifiable RL)

### 3.1 Research Hypothesis
Static foundation models cannot discover subtle environment mechanics without task-specific adaptation. Training on synthetic interaction trajectories with Verifiable Rewards (RLVR) or Group Relative Policy Optimization (GRPO) will teach the model to explore systematically, interpret spatial differences ($\Delta$), and backtrack upon encountering failure states.

### 3.2 Required Implementation
- **Failure Trajectory Extraction:** Filter all episodes where an action caused immediate `GAME_OVER` or zero-delta, pairing them with the correct recovery step.
- **Synthetic Inductive Curriculum:** Procedural generation of 2D grid mini-games with verifiable rules (mazes, key-doors, push-blocks, color matching) generating 10,000+ verifiable transitions.
- **Targeted SFT / LoRA Post-Training:** Fine-tune LoRA adapter (rank 32, alpha 64) on validated trajectory pairs.
- **GRPO / RLVR Pipeline (Phase 3b):**
  - Group size $G = 4$ rollouts per prompt.
  - Verifiable reward function:
    $$R = R_{\text{format}} + R_{\text{validity}} + R_{\text{spatial\_delta}} + R_{\text{level\_solve}} - R_{\text{repetition\_penalty}}$$
  - Rule-based verifiers eliminate reward hacking.

### 3.3 Evaluation Benchmark
- **Validation Tasks:** Held-out procedural grid puzzles and unseen ARC-AGI-3 environments.
- **Metrics:** Level solve rate, average actions to goal, recovery rate from adverse states.

### 3.4 Success Criteria
- Base Model + SAM-AI LoRA outperforms unadapted Base Model by $\ge 20\%$ in levels solved per 50 actions.
- Model demonstrates zero-shot recovery behavior (executing `RESET` or reverse move immediately after fatal state).

### 3.5 Compute and Data Prerequisites
- **Compute:** Minimum 2x NVIDIA A100 (80GB) or 4x RTX Pro 6000 for GRPO rollouts and policy gradient updates; or Kaggle GPU quota (if approved).
- **Data:** 10,000 synthetically generated, verified puzzle transitions.

### 3.6 Major Risks
- Policy collapse or length hacking under GRPO if format reward is uncalibrated.
- Distribution shift: model overfits synthetic mini-games and fails on authentic ARC-3 environments.

### 3.7 Evidence Required Before Proceeding
- Reproducible training loss curves, KL-divergence stability charts, and benchmark evaluation showing strictly positive transfer on unseen tasks.

---

## Milestone 4: Memory and World-State Tracking

### 4.1 Research Hypothesis
In partially observable or multi-room environments, reactive agents suffer from amnesia, repeatedly revisiting depleted rooms or attempting already-disproven hypotheses. Maintaining an explicit spatial memory map and a directed transition graph with cryptographic state provenance will enable long-horizon planning across multi-level games.

### 4.2 Required Implementation
- **Symbolic World Graph:** A directed state-action graph tracking:
  $$G = (V, E), \quad v_t = \text{Hash}(\text{Grid}_t), \quad e = (a_t, \Delta_t, r_t)$$
- **Episodic Failure Memory:** Indexing known dead ends, lethal entities, and immovable obstacles.
- **Spatial Object Tracker:** Tracking persistent entities across frame transitions using connected-component analysis and Hungarian matching.
- **Context Injection Policy:** Dynamic selection of relevant graph neighbors and past failures to include in the context window without exceeding token limits.

### 4.3 Evaluation Benchmark
- Multi-room navigation tasks and complex ARC-AGI-3 games (e.g. `ft09`, `ls20`) requiring backtrack exploration.
- Memory ablation: Graph Memory Enabled vs. Linear Context Sliding Window.

### 4.4 Success Criteria
- $\ge 50\%$ reduction in duplicate state visits ($v_t \in V_{\text{visited}}$).
- Successful completion of at least one multi-room game where linear sliding window consistently fails.

### 4.5 Compute and Data Prerequisites
- Local CPU compute for graph operations (`networkx`, `scipy`).
- Same GPU requirements as Milestone 1 for agent execution.

### 4.6 Major Risks
- Graph state explosion in environments with continuous animations or floating-point elements.
- Excessive prompt overhead if graph summarization is verbose.

### 4.7 Evidence Required Before Proceeding
- Empirical demonstration of graph-guided backtracking resolving a multi-room task in $< 50$ actions on held-out environments.

---

## Milestone 5: Tool-Using Autonomous Agents

### 5.1 Research Hypothesis
Pure autoregressive token generation cannot perform deep search, matrix algebra, or simulation. Providing the agent with an isolated Python REPL and discrete spatial tools (e.g., connected-component labeling, BFS pathfinding, symmetry detection) allows the model to offload computational complexity to exact deterministic algorithms.

### 5.2 Required Implementation
- **Sandboxed Execution Container:** Secure Python runtime with strict timeouts (1.0s limit per execution), memory caps (512MB), and network disabled.
- **Standard Spatial Toolset:**
  - `find_objects(grid)`: Connected components and bounding boxes.
  - `pathfind(grid, start, goal)`: Breadth-First Search avoiding obstacles.
  - `detect_symmetries(grid)`: D4 dihedral group transformations.
  - `simulate_action(grid, action)`: Internal rule prediction.
- **Tool Error Handling:** Structured exception capture returning stdout/stderr directly into the deliberation context.

### 5.3 Evaluation Benchmark
- Complex ARC-AGI-1 and ARC-AGI-2 program synthesis benchmarks.
- Tool invocation accuracy and execution error rates.

### 5.4 Success Criteria
- Agent successfully invokes pathfinding or object extraction tools in $\ge 80\%$ of relevant puzzle contexts.
- Code execution runtime overhead remains $< 200\text{ ms}$ per step.
- Zero security breaches, filesystem escapes, or unhandled exceptions.

### 5.5 Compute and Data Prerequisites
- Local or containerized execution environment (Docker or Python `subprocess` with resource limits).

### 5.6 Major Risks
- Model generates slow, non-terminating Python scripts.
- Agent relies excessively on brittle hand-crafted tools rather than general reasoning (a known failure mode documented in prior art).

### 5.7 Evidence Required Before Proceeding
- Benchmark comparison demonstrating that tool-augmented reasoning improves accuracy over pure prompt reasoning on the ARC-AGI-2 offline set.

---

## Milestone 6: Broader Generalization (Math, Coding, Science)

### 6.1 Research Hypothesis
True intelligence is general, not narrow. If SAM-AI's reasoning architecture is sound, its deliberation, verification, and failure memory mechanisms must transfer directly to domain-general tasks: formal mathematical problem-solving, competitive code generation, and hypothesis-driven scientific deduction.

### 6.2 Required Implementation
- **Domain Adapters:**
  - *Mathematics:* LaTeX / SymPy expression parser, formal proof verifier, step-by-step calculator.
  - *Coding:* Unit-test driven test runner, syntax linter, edge-case generator.
  - *Science:* Tabular hypothesis tester, statistical significance evaluator.
- **Multi-Domain Benchmark Pipeline:** Unified evaluation runner supporting standard benchmark formats without code modification.

### 6.3 Evaluation Benchmark
1. **Mathematics:** OpenAI MATH (Level 3-5 subset), AIME 2024/2025 benchmark.
2. **Coding:** HumanEval+, LiveCodeBench (Easy/Medium subsets).
3. **Interactive Reasoning:** ALFWorld, WebArena (lite subset).

### 6.4 Success Criteria
- Positive cross-domain transfer: Scaffolding improvements developed on ARC transfer to $\ge 5\%$ absolute gain on coding/math verification benchmarks without domain-specific re-tuning.
- Zero regressions on ARC-AGI-3 baselines when domain verifiers are loaded modularly.

### 6.5 Compute and Data Prerequisites
- Evaluation datasets (MATH, HumanEval, LiveCodeBench).
- Dedicated evaluation server or local runner with SymPy and pytest.

### 6.6 Major Risks
- Domain overfitting: heuristics optimized for 2D grids degrade natural language or code comprehension.
- Benchmark data contamination: foundation models may have memorized classic benchmarks.

### 6.7 Evidence Required Before Proceeding
- Published evaluation report on fresh, verified benchmark problems (e.g. post-2025 competition math and code) demonstrating verifiable reasoning improvements.

---

## Milestone 7: Multimodal Capabilities

### 7.1 Research Hypothesis
Biological and frontier intelligence processes visual, auditory, and spatial signals natively. Attempting to train a general omnimodal foundation model from scratch under current resource constraints is unfeasible; however, integrating frozen, high-fidelity pretrained vision-language backbones (e.g. Qwen2-VL, PaliGemma) with SAM-AI's symbolic representation layer will unlock rich visual scene understanding.

### 7.2 Required Implementation
- **Vision-Language Adapter:** Interface connecting pretrained vision encoders to SAM-AI's symbolic state representation.
- **Dual Representation Bridge:** Combining raw pixel tensors with symbolic ASCII / coordinate matrices, allowing the model to cross-attend between visual saliency and symbolic logic.
- **Multimodal Tool Grounding:** Bounding box prediction, visual click actuation, and optical character recognition for arbitrary GUI environments.

### 7.3 Evaluation Benchmark
- Visual puzzle benchmarks (VQA, Core-Knowledge ARC, OSWorld desktop tasks).
- Visual action accuracy and click precision ($F_1$ score on target objects).

### 7.4 Success Criteria
- Dual-representation agent achieves higher solve rates on visually complex ARC-3 environments than either purely symbolic or purely pixel-based approaches.
- Accurate spatial grounding on GUI tasks with $< 5\%$ coordinate deviation.

### 7.5 Compute and Data Prerequisites
- 1x GPU with $\ge 32\text{ GB}$ VRAM for running 7B-14B Vision-Language Models.
- Public pretrained weights (e.g. `Qwen/Qwen2-VL-7B-Instruct`).

### 7.6 Major Risks
- Vision encoders miss single-pixel changes critical to discrete grid puzzles.
- Latency increases significantly ($\ge 3\text{ s}$ per forward pass).

### 7.7 Evidence Required Before Proceeding
- Ablation study showing that visual embeddings resolve ambiguities where pure ASCII symbolic representations lose fidelity.

---

## Milestone 8: Scaling Research and Frontier Realities

### 8.1 Research Hypothesis & Honest Assessment
Frontier-grade AI (OpenAI o1/o3, DeepMind Gemini 2.0/Argon, Claude 3.7 Sonnet) is produced by massive scaling: clusters of thousands of H100/B200 GPUs, hundreds of billions of parameters, trillions of pretraining tokens, and millions of dollars in test-time RL compute. 

A small independent lab cannot compete on raw brute-force pretraining FLOPs. However, through **algorithmic efficiency, dense verification, high-signal synthetic curricula, and test-time compute scaling**, a small team can match or exceed frontier models on specialized inductive reasoning tasks.

### 8.2 Architectural Prerequisites for Frontier Scaling
To scale beyond 14B parameters toward frontier reasoning, SAM-AI requires:
1. **Multi-Head Latent Attention (MLA):** Compressing KV-cache memory footprints by $4-6\times$, enabling 64k+ context windows on constrained hardware.
2. **Deep Mixture-of-Experts (MoE):** Activating 10-20B parameters out of 100B+ total capacity for high computational throughput per FLOP.
3. **Test-Time Tree Search (MCTS / Best-of-$N$ with Verifiers):** Spending dynamic compute at inference time—allocating more tokens to harder problems and pruning unproductive branches early.
4. **Decentralized & Community Compute:** Leveraging compute pooling (e.g. Prime Intellect, TRC TPU grants, academic compute clusters).

### 8.3 Evaluation Benchmark
- Grand ARC Prize 2026 Challenge (85% milestone).
- Frontier reasoning benchmarks: FrontierMath, Humanity's Last Exam (HLE), SWE-bench Verified.

### 8.4 Success Criteria
- Demonstrating state-of-the-art reasoning efficiency per parameter on public inductive reasoning benchmarks.
- Published peer-reviewed research accepted at top venues (NeurIPS, ICML, ICLR).

### 8.5 Compute and Capital Realities
- **Pretraining a 30B MoE model:** Requires $\sim \$500,000 - \$2,000,000$ in compute (128-512 H100s for several months).
- **Post-Training & RLVR on Open Weights:** Requires $\sim \$10,000 - \$50,000$ in GPU cluster compute.
- **Current Position:** Operating on local CPU prototyping + Kaggle/subsidized GPU allocations.
- **Strategic Imperative:** Maximize software efficiency, published academic credibility, and competition prize capital before attempting large-scale compute acquisitions.

### 8.6 Major Risks
- Premature capital burn on inefficient model runs before scaffolding is verified.
- Overpromising timeline deliverables to external stakeholders.

### 8.7 Evidence Required Before Proceeding
- Validated Milestones 1 through 5, documented reproducible paper, and secured dedicated compute funding (e.g. IndiaAI Mission, TRC grant, or prize capital).

---

## Summary Protocol & Current Milestone Status

### Precise Current Status Audit (October 10, 2026)
* **Agent Code & Test Plumbing:** **IMPLEMENTED** (in `sam_ai/agent/` and `sam_ai/substrate/`).
* **Unit Tests:** **15/15 PASSED** (offline smoke tests in `tests/test_sam_ai_arc3_agent.py`, strictly stamped `TEST_ONLY`).
* **Real 14B Model Execution:** **BLOCKED** until model weight files and a CUDA GPU runtime are available.
* **Adapter Evaluation:** **BLOCKED** until verified non-empty weights are loaded. (Forensics: `sam_ai_v4_trained/checkpoint-25` is verified non-empty at 137.7 MB; `checkpoint-50` is 0 bytes and unusable).
* **Official ARC-3 Execution:** **BLOCKED LOCALLY** until official environment files (`metadata.json`) are mounted in the execution environment (e.g. via Kaggle input).
* **Real Capability Improvement:** **NOT YET DEMONSTRATED**. (Passing unit tests verifies plumbing only; real intelligence gains have not yet been evaluated).

### Milestone Phasing Matrix

| Milestone | Focus | Stage | Current Status |
|---|---|---|---|
| **M1: Execution & Evaluation** | Truthful Telemetry & Provenance | **PLUMBING READY / REAL MODEL BLOCKED** | Plumbing & `HuggingFaceModelRunner` implemented. Real-model execution BLOCKED awaiting GPU and weight files. |
| **M2: Reasoning & Verification** | Single-Variable Verification Guard | **PRE-REGISTERED / NOT_RUN** | Single-variable specification (`EXP-SAM-01`) pre-registered. NOT RUN pending user approval. |
| **M3: Learning from Failures** | Failure Memory & Synthetic GRPO | **ROADMAP SPECIFICATION** | Failure Memory (Arm C / EXP-SAM-02) deferred until M2 is evaluated. |
| **M4: Memory & World Tracking** | Provenance Graphs & State Maps | **ROADMAP SPECIFICATION** | Non-active research track. |
| **M5: Tool-Using Agents** | Sandboxed REPL & Spatial Tools | **ROADMAP SPECIFICATION** | Non-active research track. |
| **M6: Broader Generalization** | Math, Code, Science Transfers | **ROADMAP SPECIFICATION** | Non-active research track. |
| **M7: Multimodal Grounding** | Vision Encoders + Symbolic Dual | **ROADMAP SPECIFICATION** | Non-active research track. |
| **M8: Frontier Scaling** | MLA, MoE, Cluster Training | **LONG-TERM VISION** | Non-active research track. |

> **Directive:** Passing offline plumbing tests does NOT constitute a completed real-model evaluation. All execution remains strictly locked. No GPU quota consumption, Kaggle submission, or automated code execution will occur without explicit, written user authorization.


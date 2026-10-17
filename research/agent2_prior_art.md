# ARC-AGI-3 & ARC Prize 2026: Comprehensive Prior Art Deep-Dive

**Document ID:** `RESEARCH-AGENT2-PRIOR-ART-2026-10-09`  
**Author:** Agent 2 (ARC Prior Art Deep-Dive Specialist)  
**Date of Assessment:** October 9, 2026  
**Scope:** ARC-AGI-3 (Interactive Track) Milestone 1, Milestone 2, Baseline Notebooks, SGLang Serving Runtimes, and Agent Harness Architectures  
**Destination:** `research/agent2_prior_art.md`  

---

## 1. Executive Summary & Interactive ARC Landscape

ARC-AGI-3 represents a fundamental paradigm shift from static 2D grid input-output program synthesis (ARC-1 and ARC-2) to interactive, dynamic, video-game-like environments where agents perceive, explore, hypothesize, plan, and act over extended multi-turn horizons. Environments feature hidden state dynamics, lack textual rule descriptions, penalize action inefficiency relative to human baselines, and require dynamic goal inference.

Across the two 2026 milestone prizes ($37.5k for Milestone 1 on June 30, 2026; $37.5k for Milestone 2 on September 30, 2026), empirical results converged on a definitive technical hierarchy:
1. **Interactive Python REPL beats discrete action selection:** Systems that allow language models to inspect grids programmatically via a Python REPL (Tufa Labs' *The Duck*, Daniel Franzen's *Milestone 2 Solution*) outperformed systems restricted to raw JSON action emission (Reki, Forge, official template).
2. **Harness engineering and high-throughput inference dominate score ceilings:** Frontier model capability bounds the solvability of complex puzzles, but under Kaggle's single-GPU 9-hour limit across 110 private competition games, token generation throughput (550–850 tokens/sec), prefix KV-cache reuse (>93%), and priority scheduling dictate total game completions.
3. **Hand-crafted heuristic tools hurt generalization:** Rigid macro-pathfinders, hardcoded shape extractors, and prescriptive policy prompts consistently degraded performance across multiple independent competitors by constraining model adaptation, introducing edge-case bugs, and burning action budgets.

---

## 2. Deep-Dive into Documented Solutions & Milestone Write-Ups

### 2.1 ARC Prize Official GPT-OSS-120B Template
* **Author / Provider:** Greg Kamradt (ARC Prize Official Team)  
* **Source:** Kaggle Notebook `gregkamradt/arc-agi-3-gpt-oss-120b` [Primary Code]  
* **Architecture:** Vision-language policy utilizing OpenAI's open-weight `GPT-OSS-120B` (117B parameter Mixture-of-Experts with MXFP4 4.25-bit quantization) served locally on a single 80GB GPU container.
* **Mechanism:**
  - Evaluates games turn-by-turn by feeding labeled RGB grid images and ASCII arrays into the model.
  - Generates single JSON actions per step with system-level constraints enforcing valid environment action schemas (`ACTION1`–`ACTION6`).
  - Serves as the canonical baseline harness that demonstrated offline local model hosting within Kaggle's BYOD container.
* **Limitations:**
  - Emits one action at a time without multi-step simulation or computational reflection.
  - Slower inference compared to highly optimized speculative MTP runtimes.
  - High token cost per step due to repeated prefill of full game history on each turn.

### 2.2 Milestone 1 Winner: Tufa Labs — "The Duck"
* **Authors:** Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tešnar, Stefano Viel (Tufa Labs)  
* **Score & Rank:** 1st Place Milestone 1 (June 30, 2026), $25,000 Prize; Mean score across 25 public games: $1.60 \pm 0.45$ [Official]  
* **Sources:**
  - Kaggle Notebook: `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` [Primary Code]
  - Kaggle Discussion: Topic 717133 (`tufa-labs-duck-harness-writeup`) [Community]
  - GitHub: `Tufalabs/duck-harness` [Primary Code]
  - Blog: `tufalabs.ai/research/duck-harness/` [Community]
  - Video / Discussion: *Machine Learning Street Talk* (MLST) with Tim Scarfe [Community]
* **Architecture & Mechanics:**
  - **REPL-as-Harness:** Encodes ARC game states into Python variables inside an interactive Read-Eval-Print Loop. The agent writes and executes Python code to inspect the grid, evaluate hypothesis loops, and dispatch action batches.
  - **Multimodal Perception:** Combines rendered RGB board images, ASCII matrices, and a coordinate-level segmentation tool.
  - **Infinite Play via Context Eviction:** Continuously pops the oldest conversational messages from the context window, retaining only the immutable system prompt and the most recent history.
  - **Local Model:** Served `Qwen3.6-27B-FP8` locally on a single Kaggle GPU.
  - **Comparative Benchmark:** Evaluated against GPT-5.4 on Executable World Models; demonstrated equivalent level completion profiles at an order of magnitude lower token cost.
* **Key Findings:**
  - **Improvisation beats Hand-Crafted Tools:** Tufa Labs explicitly reported that providing hand-crafted domain tools hurt the model. Allowing the LLM to write raw numpy/Python code to inspect slices and test assumptions worked significantly better than rigid abstractions.
  - **Model scaling is the primary lever:** Solvability of difficult games scaled directly with base model reasoning capability.

### 2.3 Milestone 1 2nd Place: Reki
* **Author:** Ruichard Liu (`ruichardliu`)  
* **Score & Rank:** 2nd Place Milestone 1 (June 30, 2026), $10,000 Prize [Official]  
* **Sources:**
  - Kaggle Notebook: `ruichardliu/milestone1-2nd-solution` [Primary Code]
  - ARC Prize Blog: `arcprize.org/blog/arc-prize-2026-milestone-1` [Official]
* **Architecture & Mechanics:**
  - **Vision-LLM-as-Policy:** Forked from Greg Kamradt's official GPT-OSS-120B template, replacing the model with `Gemma-4-31B`.
  - **Structured Action Planning:** Model analyzes recent labeled board images and outputs a structured JSON object containing:
    1. Visual differential summary (what changed).
    2. Short reasoning hypothesis.
    3. Action plan queue of 1–4 sequential moves.
  - **Reflection Memory:** Updates a rolling textual memory bank every ~10 environment steps.
  - **Numpy Click Heuristics (CPU-only):**
    - Fallback/exploratory click heuristic: Automatically prioritizes small, rare-colored, button-like contours rather than uniform background pixels.
    - Dead-signature tracking: Automatically detects when clicking a specific object type produces zero board transition, immediately masking that object signature from future clicks on that level.
  - **Ablation Toggles:** Every component (click heuristics, dead-signatures, memory refresh, plan queue length) was wrapped in environment variable toggles for empirical testing.

### 2.4 Milestone 1 3rd Place: Md Boktiar Mahbub Murad — "Forge"
* **Author:** Md Boktiar Mahbub Murad (`mbmmurad`)  
* **Score & Rank:** 3rd Place Milestone 1 (June 30, 2026), $2,500 Prize [Official]  
* **Sources:**
  - Kaggle Notebook: `mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone` [Primary Code]
  - ARC Prize Blog: `arcprize.org/blog/arc-prize-2026-milestone-1` [Official]
* **Architecture & Mechanics:**
  - Built on the official template with `Gemma-4-31B`.
  - Developed "Forge," a profile-driven execution framework featuring candidate action generators, multi-action scoring arbiters, and confidence-gated fallback prompts.
* **Key Finding (Arbiter Failure):**
  - **Disabling complex machinery produced the winning run:** Murad discovered that the top-scoring submission occurred when the multi-action generator, candidate arbiter, and confidence gating were **disabled**.
  - **Evaluation Discrepancy:** Noted that local public game performance did not reliably correlate with private leaderboard scores.

### 2.5 Milestone 2 Winner: Daniel Franzen — Building on the Duck Harness
* **Author:** Daniel Franzen (`dfranzen` / `da-fr`, "The ARChitects")  
* **Score & Rank:** 1st Place Milestone 2 (September 30, 2026), $25,000 Prize; Official Public Score: **27.89** (Highest verified open-source score on the leaderboard) [Official / Primary Code]  
* **Sources:**
  - Kaggle Notebook: `dfranzen/arc-agi-3-milestone-2-solution` [Primary Code]
  - GitHub Repository: `da-fr/arc-agi-3-solution` [Primary Code]
  - Technical Documentation: `da-fr/arc-agi-3-solution/WRITEUP.md` [Primary Code]
  - Upstream Runtime: Pennyroyal SGLang fork `jpezzulli/sglang-rtxpro6000` (v2.5.3, commit `d00d88e`) [Primary Code]
* **Architecture & Innovations:**
  1. **Model Stack & Quantization:**
     - Base Model: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (INT4 weights, W4A16).
     - Drafter Model: `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` (speculative decoding via Multi-Token Prediction draft checkpoints).
     - Full unquantized BF16 PLE embedding table offloaded to host system RAM on Kaggle's `NvidiaRtxPro6000` (48GB VRAM).
     - SGLang engine configured with FP8 E4M3 KV cache, 60 Mamba cache slots, FlashInfer linear backend, and low-M BF16 GEMM kernel patches.
  2. **Serving & Throughput Performance:**
     - 609 to 836+ decode tokens/sec, 10,550+ prefill tokens/sec.
     - **93.43% weighted prompt prefix-cache reuse** across thousands of evaluation requests.
     - Long-prompt cache miss rate reduced to 0.03% (1 out of 3,248 requests longer than 62k tokens).
  3. **Hysteresis-Driven Blockwise History Trimming:**
     - Maximum context window: 131,072 tokens (128 Ki).
     - Instead of trimming small increments (which destroys the shared KV prefix on every request), trims a large block down to ~57,000–59,000 tokens once the limit is approached.
     - Allows the retained prefix to stay pinned in cache across dozens of subsequent turns.
  4. **Bounded Concurrent Admission (Cache Protection):**
     - Bounded concurrent active games to exactly 10 streams (matching GPU KV-cache capacity).
     - Games hold admission slots across sub-second tool calls; context eviction serves as the handover point to swap games without thrashing VRAM.
  5. **Priority-Governed Compute Scheduling:**
     - Replaced equal time allocation with dynamic priority scheduling across all games:
       $$P = (A + B) \cdot C$$
     - $A$: Level completion value penalized by action count: $A = \ell \cdot \left(\frac{25}{25 + a}\right)^2 \cdot \frac{55}{N(N+1)/2}$.
     - $B$: Continuation opportunity lookup based on remaining levels, linearly faded to 0 in the final 20% of runtime ($B_{\text{effective}} = B \cdot \min(1, \frac{r}{0.2T})$).
     - $C$: Empirically calibrated Gaussian fade curve penalizing games that absorb excessive actions ($a$) or tokens ($t$):
       $$C = 0.25 \cdot 2^{-(a/115)^2} + 0.75 \cdot 2^{-(t/62000)^2}$$
     - Successfully starved stalled wandering games while channeling compute into games with active multi-level breakthroughs.
  6. **Perception Enhancements:**
     - **10x Image Upscaling:** Upscaled board rendering by 10× (up from 4×), calibrated with image-patch token accounting.
     - **Frame Difference Images:** Rendered differential masks showing exact pixel changes between actions and fatal transitions.
     - **Fatal Game-Over Frames:** Exposed the exact death frame alongside the reset state so the model could diagnose why it failed.
     - **Animation Exposure:** Intermediate animation frames exposed as Python variables, triggering investigation prompts only when transient pixels (pixels changing multiple times between keyframes) were detected.
  7. **Action Mechanics:**
     - Exposed `UNDO` action (specifically crucial for reversible block-pushing and maze levels like `su15`).
     - Persistent Python helper functions: Saved model-defined functions across tool invocations to eliminate repetitive boilerplate generation.
     - Non-semantic action guards: Batch no-op guard (halts queued actions on zero interior change), Stale-state guard (blocks subsequent action calls in the same turn if the first was ineffective), and Terminal-state guard.

### 2.6 Early Precursors: StochasticGoose & RGB-Agent
* **StochasticGoose (Tufa Labs / Dries Smit & Jack Cole):**
  - Winner of the ARC-AGI-3 Agent Preview Competition [Official].
  - Source: `github.com/DriesSmit/ARC3-solution` [Primary Code].
  - Pure Reinforcement Learning / CNN action learner (no LLM).
  - Used a shared-backbone CNN to perform binary classification: predicting whether `ACTION1`–`ACTION6` would cause any frame change.
  - Employed hierarchical convolutional sampling (action type first, then spatial coordinate) to maintain strict 2D translational equivariance.
  - Hash-based deduplication in the replay buffer for sample efficiency under a ~200k action budget.
* **RGB-Agent / Duke Harness (Alexis Fox):**
  - First system to demonstrate near-human action efficiency on initial ARC-3 preview puzzles [Community].
  - Source: `blog.alexisfox.dev/arcagi3` [Community].
  - Introduced the "log-as-context" and "plan-queue" paradigm: maintaining lightweight game logs, inspecting history via bash/python tools, and queuing short action bursts based on explicit hypothesis generation.
  - Direct architectural predecessor to Tufa Labs' Duck harness.

### 2.7 Heavyweight Orchestrators: Symbolica AI (ARCgentica)
* **Author:** Symbolica AI  
* **Score:** 36.08% on ARC-AGI-3 public preview [Community / Single-source; marked UNVERIFIED for offline container track]  
* **Sources:**
  - `symbolica.ai/blog/arc-agi-3` [Community]
  - `github.com/symbolica-ai/agentica` [Primary Code]
* **Architecture:** Orchestrator-subagent hierarchy built on the typed Agentica SDK.
  - Orchestrator delegates tasks to specialized subagents that write Python programs, evaluate grid transformations, and recursively call sub-solvers.
  - Relied on frontier cloud models (Claude 3.5 Opus / Opus 4.6) with massive API token budgets.
  - **Offline Container Feasibility:** **Infeasible** in offline Kaggle container due to recursive multi-agent API dependency, extreme token overhead, and compute costs exceeding single-GPU boundaries.

---

## 3. Synthesis: What Worked vs. What Hurt

### 3.1 What Worked (Empirically Proven Techniques)

| Technique | Primary Proponent | Empirical Mechanism & Impact |
| :--- | :--- | :--- |
| **Python REPL Tool Interface** | Tufa Labs, Franzen | Allows the agent to write arbitrary code, inspect matrices, test invariants, and run simulation loops. Vastly superior to fixed action spaces. |
| **Speculative Drafting (MTP + INT4)** | Franzen (Pennyroyal) | Qwen3.8-Flash-Next W4A16 + Albucino INT4 drafter yielded 609–836+ tokens/sec decode throughput, enabling 110 competition games to finish within 9 hours. |
| **Hysteresis Blockwise Context Trimming** | Franzen | Trimming down to 57k tokens whenever context exceeds 128k achieved 93.43% prefix KV-cache reuse, virtually eliminating prefill latency stalls. |
| **Bounded Concurrent Admission (10 Streams)** | Franzen | Prevents KV-cache thrashing by limiting simultaneous active games to exact GPU cache capacity, freezing inactive games. |
| **Priority-Governed Compute Scheduling** | Franzen | Formula $P = (A + B) \cdot C$ starved wandering zero-progress games and redirected remaining runtime to games with active level breakthroughs. |
| **10x Perception Scaling & Difference Images** | Franzen | Upscaling grids to 10× and providing explicit delta frames between actions provided fine-grained visual evidence of mechanics. |
| **Exposing Intermediate Animations** | Franzen, Bröggen | Exposing intermediate animation frames in Python variables enabled solving dynamic puzzles containing transient pixel movements. |
| **Exposing UNDO Action** | Franzen | Backtracking allowed models to recover from fatal grid traps without restarting entire levels. |
| **Persistent Python Helper Namespace** | Franzen | Retaining defined functions across tool calls eliminated redundant code generation, saving token budget. |
| **Action Guards (Batch No-Op, Stale-State)** | Franzen | Halting execution when an action produces zero interior change prevented runaway action-budget waste. |
| **Click Heuristics & Dead-Signature Masking** | Reki | CPU-based numpy contour analysis prioritized rare button-like shapes and stopped wasted clicks on inert objects. |

### 3.2 What Hurt (Empirical Regressions & Anti-Patterns)

| Failed Technique | Evidence Source | Why It Failed |
| :--- | :--- | :--- |
| **Hand-Crafted Domain Tools & Macro Pathfinders** | Tufa Labs (MLST), SAM-AI V8 comparative analysis | Hardcoded pathfinders (e.g. BFS shortest path, shape extractors) constrained agent policy, introduced edge-case bugs, and degraded scores from 29.21 to 25.76. |
| **Structured Text World-Model Templates** | Franzen | Strict markdown/JSON templates suffered from parsing failures, repeatedly re-injected stale hallucinated facts, and consumed context. Disabling it improved scores. |
| **Periodic Semantic Summarization** | Franzen | Summaries blurred precise coordinate and procedural details, adding latency without improving level completions. |
| **Candidate Action Arbiters & Unguided Branching** | Murad (Forge) | Generating multiple action candidates and scoring them with an arbiter model added latency and decision noise. Disabling the arbiter produced Forge's 3rd place score. |
| **Greedy Resets & Low Abort Thresholds** | SAM-AI V8 Analysis | Forcing `EXPOSE_RESET` and aggressive abort thresholds (`NOOP_ABORT_THRESHOLD="3"`) burned action budgets prematurely and prevented stochastic recovery. |
| **System Prompt Contamination & Rule Policing** | SAM-AI V8 Analysis | Injecting verbose role-play directives ("Never waste steps", "Trigger orthogonal moves") restricted natural spatial reasoning and accelerated token trim boundaries. |
| **Composite Animation Overlays** | Franzen | Blending animated frames into single composite images confused visual reasoning compared to exposing raw sequential frames in Python variables. |

---

## 4. Compute Resources, Runtime & Hardware Scaling

### 4.1 Hardware Profiles in Practice
* **Official Kaggle ARC-AGI-3 Environment:**
  - Machine Shape: `NvidiaRtxPro6000` (48 GB GDDR6 VRAM, SM120 architecture, host memory offload supported).
  - Alternative Kaggle Shapes: 2x Nvidia T4 (16 GB each) or 1x Nvidia L4 (24 GB) — insufficient for Qwen3.8-Flash-Next at long contexts; strictly requires the RTX Pro 6000 shape.
  - Host RAM: Sufficient to hold the full unquantized BF16 PLE embedding table (~16 GB) while offloaded.

### 4.2 Quantization & Memory Budgeting
* **Weights:** Intel AutoRound W4A16 quantization compresses the base model into ~18 GB VRAM.
* **Draft Model:** Albucino INT4 MTP draft checkpoint consumes ~2.5 GB VRAM.
* **KV-Cache:** FP8 E4M3 quantization allocates the remaining ~25 GB VRAM to KV cache, supporting 139,264 context tokens across 10 concurrent streams.
* **Mamba Cache:** Fixed at 60 recurrent state slots to maintain prefix reuse for hybrid linear attention layers.

### 4.3 Training vs. Inference Spend
* **Zero Parametric Fine-Tuning:** All top-ranking ARC-AGI-3 solutions (The Duck, Reki, Forge, Franzen) utilized **frozen, pre-trained base models** with zero competition fine-tuning.
* **Development Compute:** Daniel Franzen reported utilizing Kaggle's free GPU quota plus ~150 rented GPU-hours on cloud RTX Pro 6000 instances solely for inference harness testing.

---

## 5. Feasibility within Kaggle's Offline BYOD Container

Kaggle submissions are governed by strict execution boundaries:
1. **Network Isolation:** `enable_internet = False`. No external API calls, GitHub fetches, or Hugging Face downloads are permitted during scoring.
2. **Execution Time Limit:** 9 hours (32,400 seconds) wall-clock time across all hidden evaluation games.
3. **Container Environment:** Kaggle BYOD Docker image (`gcr.io/kaggle-private-byod/python@sha256:...`).

### 5.1 Offline Packaging Feasibility: VERIFIED
Franzen demonstrated 100% offline feasibility via pre-bundled wheelhouses and dataset mounts:
* **Precompiled Wheels:** `dfranzen/pennyroyal-v253` supplies SGLang, FlashInfer, and Triton dependencies without internet access.
* **Harness Source:** `dfranzen/taaf-kaggle-source-bundle-copy` mounts the entire TAAF execution framework.
* **Offline Model Loading:** Models and draft checkpoints are mounted directly as Kaggle datasets (`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` and `dfranzen/albucino-qwen3-8-flash-next-drafter`).
* **Startup Overhead:** Offline venv extraction, wheel installation, and SGLang server initialization complete in approximately 3 to 4 minutes.

### 5.2 9-Hour Throughput & Action Budget Feasibility: VERIFIED
* **Live Competition Scale:** 110 hidden arcade games evaluated sequentially in a single session.
* **Time Budget per Game:** Average of $\frac{32,400 \text{ s}}{110} \approx 294.5 \text{ seconds}$ (~4.9 minutes) per game.
* **Pass Strategy:** Live submissions must enforce `n_passes = 1` (`TRUE_SUBMISSION = True`). Attempting multi-pass sampling (e.g. `n_passes = 4`) on 110 games results in 9-hour timeout failure.
* **Throughput Match:** At 609+ tokens/sec aggregate generation and 10,500+ tokens/sec prefill, 10 concurrent admission slots process the full 110 games within 7.5 to 8.2 hours, comfortably below the 9-hour cutoff.

---

## 6. Comparative System Matrix

| Dimension | Tufa Labs "The Duck" | Reki | Md Boktiar Murad "Forge" | Daniel Franzen (M2 Winner) | Symbolica ARCgentica |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Milestone / Date** | Milestone 1 (2026-06-30) | Milestone 1 (2026-06-30) | Milestone 1 (2026-06-30) | Milestone 2 (2026-09-30) | Early 2026 Preview |
| **Score / LB** | 1st Place M1 (1.60 public) | 2nd Place M1 (0.92 public) | 3rd Place M1 (0.86 public) | **1st Place M2 (27.89 LB)** | 36.08% Preview (UNVERIFIED) |
| **Model** | Qwen 3.6 27B FP8 | Gemma-4-31B | Gemma-4-31B | Qwen3.8-Flash-Next W4A16 | Claude 3.5 Opus |
| **Action Interface** | Python REPL | JSON Action (1–4 plan) | JSON Action | Python REPL | Python Script Synthesis |
| **Speculative Engine** | None (standard vLLM) | None | None | SGLang + INT4 MTP Drafter | API-based |
| **Context Strategy** | Infinite play via eviction | Sliding window | Sliding window | Hysteresis block trim (128k $\to$ 57k) | Multi-agent context forks |
| **Prefix Cache Hit Rate**| Low-Medium | Low | Low | **93.43% weighted reuse** | N/A (API) |
| **Scheduling Policy** | FIFO Sequential | FIFO Sequential | FIFO Sequential | **Dynamic Priority ($P=(A+B)C$)**| Static Recursive Tree |
| **Perception** | 4x RGB + ASCII | Labeled RGB images | Labeled RGB images | 10x RGB + Diff + Animations | Rendered Grids |
| **Action Guards** | Basic environment | Legal-action filter | Safe-move fallback | Batch No-op + Stale-state | Execution unit tests |
| **Offline Container** | VERIFIED Feasible | VERIFIED Feasible | VERIFIED Feasible | **VERIFIED Optimal Feasibility** | INFEASIBLE (Cloud API) |

---

## 7. Citations & Attribution Registry

| Reference Name | URL / Path | Access Date | Source Type | Verification Status | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **ARC Prize M1 Blog** | `https://arcprize.org/blog/arc-prize-2026-milestone-1` | 2026-10-09 | OFFICIAL | VERIFIED | Official winners announcement and architecture breakdown. |
| **ARC Prize M2 Status**| `https://arcprize.org/competitions/2026` | 2026-10-09 | OFFICIAL | VERIFIED | Milestone 2 prize schedule ($25k, $10k, $2.5k) ending Sept 30, 2026. |
| **Franzen Milestone 2 Code** | `https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution` | 2026-10-09 | PRIMARY CODE | VERIFIED | 1st place Milestone 2 notebook (Score: 27.89). |
| **Franzen Solution Repo** | `https://github.com/da-fr/arc-agi-3-solution` | 2026-10-09 | PRIMARY CODE | VERIFIED | Upstream repository containing full WRITEUP.md and SGLang patches. |
| **Franzen Technical Writeup** | `https://raw.githubusercontent.com/da-fr/arc-agi-3-solution/main/WRITEUP.md` | 2026-10-09 | PRIMARY CODE | VERIFIED | Primary source for priority scheduling, hysteresis trimming, and perception findings. |
| **Pennyroyal SGLang Fork** | `https://github.com/jpezzulli/sglang-rtxpro6000` | 2026-10-09 | PRIMARY CODE | VERIFIED | Commit `d00d88e`, v2.5.3 offline serving wheelhouse. |
| **Tufa Duck Harness Repo** | `https://github.com/Tufalabs/duck-harness` | 2026-10-09 | PRIMARY CODE | VERIFIED | Commit `7652836`, base solver architecture. |
| **Tufa Duck Technical Post** | `https://tufalabs.ai/research/duck-harness/` | 2026-10-09 | COMMUNITY | VERIFIED | Architectural diagram, cost comparisons vs GPT-5.4, public game scores. |
| **Tufa Kaggle Writeup** | `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133` | 2026-10-09 | COMMUNITY | VERIFIED | Kaggle discussion detailing the Python REPL interface. |
| **Tufa MLST Video** | `https://www.youtube.com/watch?v=Vg6FBKTlfOw` | 2026-10-09 | COMMUNITY | VERIFIED | Technical interview explaining why hand-crafted tools hurt. |
| **Reki 2nd Solution** | `https://www.kaggle.com/code/ruichardliu/milestone1-2nd-solution` | 2026-10-09 | PRIMARY CODE | VERIFIED | Notebook implementing click heuristics and Gemma-4-31B policy. |
| **Forge 3rd Solution** | `https://www.kaggle.com/code/mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone` | 2026-10-09 | PRIMARY CODE | VERIFIED | Notebook demonstrating generator/arbiter degradation when active. |
| **Official GPT-OSS-120B**| `https://www.kaggle.com/code/gregkamradt/arc-agi-3-gpt-oss-120b` | 2026-10-09 | PRIMARY CODE | VERIFIED | Official competition starter template. |
| **StochasticGoose** | `https://github.com/DriesSmit/ARC3-solution` | 2026-10-09 | PRIMARY CODE | VERIFIED | CNN action-learning preview winner. |
| **Symbolica ARCgentica** | `https://www.symbolica.ai/blog/arc-agi-3` | 2026-10-09 | COMMUNITY | UNVERIFIED | 36.08% claim single-source; cloud API only, infeasible offline. |
| **RGB-Agent / Duke Harness** | `https://blog.alexisfox.dev/arcagi3` | 2026-10-09 | COMMUNITY | VERIFIED | Foundational log-as-context and plan-queue concept. |

---

## 8. Concrete Recommendations for SAM-AI Solver Pipeline

1. **Keep the Franzen Milestone 2 SGLang + Priority Scheduling Base Intact:**
   - Base engine (Qwen3.8-Flash-Next W4A16 + Albucino Drafter on RTX Pro 6000) achieves 609–836 tokens/sec and 93.4%+ prefix reuse.
   - Do not alter SGLang launch flags or Mamba cache allocations.
2. **Eliminate All Hand-Crafted Macro Tools (Per Rule 8 & V8 Ablations):**
   - Strip out custom macro pathfinders (`sam_ai_v8_bfs_shortest_path`), verbose rule-policing system prompt prefixes, and hardcoded reset injections.
   - Rely strictly on native Python REPL inspection, 10× upscaled images, differential frames, and model-written helpers.
3. **Respect Submission Gateway Constraints:**
   - Offline validation runs: $N_{\text{passes}} = 4$ across local demo games.
   - True Kaggle submission runs: Strictly enforce $N_{\text{passes}} = 1$ and `concurrency = 120` (gated by 10 active admission streams) to ensure all 110 games complete well within the 9-hour limit.

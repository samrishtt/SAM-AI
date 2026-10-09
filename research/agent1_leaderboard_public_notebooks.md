# ARC-AGI-3 (ARC Prize 2026): Leaderboard & Public Notebooks Analysis

**Author:** Agent 1 (Leaderboard & Public Notebooks Analyst)  
**Date:** October 9, 2026  
**Scope:** Kaggle ARC Prize 2026 Interactive Track (`arc-prize-2026-arc-agi-3`)  

---

## 1. Executive Summary

The **ARC Prize 2026 - ARC-AGI-3** competition on Kaggle marks a paradigm shift from static grid transformation puzzles to dynamic, turn-based interactive game environments requiring real-time discovery of game mechanics, planning, and goal acquisition. 

Key findings as of October 9, 2026:
1. **Top Leaderboard Score:** The overall highest public leaderboard score is **55.89**, held by **Tufa Labs**. However, Tufa Labs did **not** open-source their solver for the Milestone 2 checkpoint (September 30, 2026).
2. **Top Open-Source / Public Notebook:** **Daniel Franzen** (`dfranzen`) won Milestone 2 with an official verified prize score of **27.89%** (frequently cited as **27.9%**) and a public notebook best of **31.47**, built as an extensive optimization fork of Tufa Labs' original Duck harness.
3. **Hardware & Serving Breakthroughs:** Top performance requires specialized local execution harnesses on Kaggle's dedicated **Nvidia RTX Pro 6000** instances. Key architectural enablers include custom SGLang runtimes, long-context Mamba prefix-cache retention (up to 128 Ki tokens), quantized weights (e.g., Qwen3.8-Flash-Next with Intel AutoRound W4A16), and dynamic inter-game compute allocation.
4. **Tufa Labs Open-Source Status:** As of early October 2026, **no newer official harness version was open-sourced by Tufa Labs**. Their public repository remains at the June Milestone 1 baseline (upstream commit `7652836`). Daniel Franzen's fork represents the state-of-the-art open-source codebase for the competition.
5. **Generalization, Overfitting, and Leakage:** Direct test set data leakage is prevented via Kaggle's air-gapped Phase-B evaluation environment (internet disabled on hidden private games). However, high leaderboard scores reflect substantial "harness engineering" (token budgets, caching, REPL tools) rather than standalone foundation model capabilities, with frontier models scoring <2% without execution scaffolding.

---

## 2. Highest-Scoring Public Notebooks

| Metric / Attribute | Daniel Franzen (Milestone 2 Winner) | Tufa Labs (Milestone 1 Winner) |
| :--- | :--- | :--- |
| **Notebook URL** | `kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution` | `kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` |
| **Repository** | `github.com/da-fr/arc-agi-3-solution` | `github.com/TufaLabs/duck-harness` |
| **Author / Team** | Daniel Franzen (`dfranzen` / `da-fr`, "The ARChitects") | Tufa Labs (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Dries Smit, Michal Tešnar, Stefano Viel) |
| **Reported Score** | **27.89%** (official milestone score); **31.47** (Kaggle public notebook V1 best) | **1.21%** (Milestone 1 winning score); (Team holds **55.89** on current public LB using unreleased internal solver) |
| **Date Published** | September 30, 2026 (release confirmed early October 2026) | June 30, 2026 |
| **License** | **Apache 2.0** | **Apache 2.0** |
| **Target Hardware** | 1x **Nvidia RTX Pro 6000** (`g4-standard-48` competition machine) | **Nvidia RTX Pro 6000** / Kaggle GPU accelerator |
| **Runtime Configuration** | Up to 9 hours limit; 10 parallel game streams; chunked prefill 8,192 tokens | Up to 9 hours limit; Python 3.12 competition wheelhouse |
| **Foundation Model Used** | **Qwen3.8-Flash-Next** (Intel W4A16 AutoRound, BF16 PLE offloaded to host RAM; Albucino quantized MTP draft for speculative decoding) | **Qwen 3.6 27B FP8** (`Qwen/Qwen3.6-27B-FP8`) |

---

## 3. In-Depth Solution Profiles

### 3.1. Daniel Franzen's Milestone 2 Solution (`dfranzen`)
- **Core Architecture:** Built upon Tufa Labs' Duck harness (forked from upstream commit `7652836056c59e044f093e3c13ed7438c814169e`).
- **Serving Stack:** Runs John Pezzulli's SGLang fork (*Pennyroyal* v2.5.3, commit `d00d88e`) with custom kernel patches:
  - Low-M BF16 GEMM support for small batch-size efficiency.
  - Speculative-state memory budget correction avoiding over-reservation.
  - Marlin scale-dtype fix resolving BF16/FP16 quantized weight loading.
  - Custom Mamba prefix-cache patch retaining only final prefill checkpoints and refreshing LRU positions.
- **Context & Cache Engineering:**
  - 128 Ki harness context window with 12 Ki output reservation.
  - Replaced fixed character-to-token heuristics with calibrated token counters and image-patch dimension accounting.
  - Hysteresis history trimming: Evicts a large block (down to ~57 Ki retained) rather than trimming turn-by-turn, achieving a **93.43% prefix cache hit rate**.
- **Action & Reasoning Augmentation:**
  - Expanded game visual scale from 4× to 10×.
  - Added difference images to highlight dynamic changes.
  - Enabled intermediate animation frame observation.
  - Added execution guards to abort wasted loops if board state remains unchanged.
  - Enabled explicit `UNDO` action usage.
- **Dynamic Compute Allocation:** Bounded admission to 10 concurrent streams; prioritized games demonstrating active progression over stagnating levels.

### 3.2. Tufa Labs Duck Harness (`jeroencottaar`)
- **Core Architecture:** The foundational LLM agent scaffold for ARC-AGI-3. Rather than direct API action calls, the model operates in an interactive Python REPL session through TAAF (Tufa ARC-AGI Framework).
- **Environment Interface:** Encodes grid observations as Python variables (`game`, `board`, `colors`). The model inspects state variables, tests speculative logic via helper functions, and triggers state transitions by executing Python code snippets.
- **Context Management:** "Infinite play via eviction": Maintains a static system prompt and evicts older interaction turns as the context approaches model limits.
- **Significance:** Won Milestone 1 on June 30, 2026 (1.21% score), proving that code-executing harnesses on open-weight models outperform direct next-action classification.

### 3.3. Other Community Baselines
- **ARC Prize Starter Kit (`arcprize/ARC-AGI-3-Kaggle-Starter`):** Official local development framework for packaging and evaluating agents against public game sets before pushing to Kaggle Phase-B submissions.
- **"Tufa Labs duck harness but Qwen 35b" / Community Variants:** Community iterations substituting alternative model weights (e.g., Qwen 2.5 32B/72B, custom LoRA checkpoints).
- **Heuristic Starters:** "Graph-Based Exploration" and "Memory-Guided Agent" community notebooks exploring graph state traversal and external memory buffers.

---

## 4. Tufa Labs Harness Investigation: Early October Status

### 4.1. Milestone 1 Documentation & Discussion #717133
- **Discussion Thread:** Kaggle Discussion ID `717133` (`kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133`).
- **Contents:** Authored by Tufa Labs (Jeroen Cottaar et al.), outlining the motivation for shifting from reinforcement learning (their earlier *StochasticGoose* project) to an LLM-driven REPL harness.
- **Key Takeaways:** Explains why interactive REPL code generation dramatically reduces hallucination compared to single-token action decoding, enabling models to verify hypotheses before committing moves.

### 4.2. Early October Open-Source Verification
- **Was a newer Tufa Labs harness open-sourced in early October?** **NO.**
- **Investigation Details:**
  - Upstream repository `github.com/TufaLabs/duck-harness` has not released a Milestone 2 update.
  - For the September 30 Milestone 2 checkpoint, Tufa Labs opted **not** to open-source their enhanced codebase.
  - Because Milestone prizes mandate immediate open-sourcing, the Milestone 2 prize was awarded to **Daniel Franzen** ($25,000 for 1st place).
  - Tufa Labs maintains the top public leaderboard score (**55.89**) using an unreleased internal solver. Under competition rules, they have until the final deadline (November 2, 2026) to publish their code if they wish to remain eligible for Grand and final placement prizes.

---

## 5. Verification of Scores, Divergence, and Overfitting/Leakage

### 5.1. Top Score Verification
- **Verified Leaderboard Leader:** **Tufa Labs** at **55.89** (Evaluation metric: action efficiency against median human benchmark across hidden test environments).
- **Verified Public Code Leader:** **Daniel Franzen** at **27.89%** (Milestone 2) / **31.47** (Notebook V1 public best).
- *Single-source claim note:* Reports mentioning OpenAI GPT-6 Astra (62.7% standard harness, 99.9% adapter) and Prime Intellect Prime Agent (95.5%) originate from external benchmark results outside the isolated Kaggle code submission track [UNVERIFIED].

### 5.2. Public vs. Private Leaderboard Divergence
- **Test Set Partitioning:** The competition evaluates agents on a hidden test suite of 110 games. 
  - **Public Split:** ~50% of games determine the ongoing public leaderboard.
  - **Private Split:** ~50% of games are reserved for final standings announced on December 4, 2026.
- **Expected Divergence:** Because the private split is completely unrevealed, final shakeup cannot yet be measured. However, participants report high run-to-run score variance on the public split caused by:
  - Agent exploration stochasticity (sampling temperature, random restart paths).
  - Context eviction edge cases causing abrupt loss of critical game state memories.
  - Compute budget starvation on challenging multi-level games.

### 5.3. Overfitting and Data Leakage Analysis
1. **Data Leakage Risk: LOW / CONTROLLED**
   - Submissions are evaluated in Kaggle's Phase-B environment with internet disabled (`offline`).
   - The test set games are held private and cannot be inspected or leaked via external API telemetry.
2. **Domain-Specific Overfitting: HIGH ATTENTION FROM ORGANIZERS**
   - The ARC Prize Foundation actively monitors for synthetic task templates or brittle hardcoded rules. Solutions relying on narrow heuristics or synthetic "lookalike" pretraining are disqualified from the official leaderboard and relegated to community tracks.
   - Earlier pre-launch attempts using pure reinforcement learning (e.g. *StochasticGoose*, 12.58% preview score) suffered catastrophic collapse when transferred to the full, varied benchmark.
3. **The "Harness Effect" Divergence:**
   - There is an immense divergence between raw model intelligence and harness efficiency. Unassisted frontier LLMs achieve <2% on ARC-AGI-3.
   - The jump from 1.21% (Milestone 1) to 27.89% (Franzen) and 55.89% (Tufa Labs) was driven by inference serving speed, context window scaling, prefix cache reuse, and perception enhancements (difference images, frame animations) rather than fundamental breakthroughs in foundational model weights alone.

---

## 6. Cited References

| Source / Entity | URL | Access Date | Source Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Kaggle Competition** | `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3` | Oct 9, 2026 | OFFICIAL | Competition hub, rules, and leaderboard |
| **ARC Prize Foundation** | `https://arcprize.org/` | Oct 9, 2026 | OFFICIAL | Official benchmark details, milestone announcements |
| **ARC-AGI-3 Starter Repo** | `https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter` | Oct 9, 2026 | OFFICIAL | Official participant development kit |
| **Daniel Franzen Notebook** | `https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution` | Oct 9, 2026 | PRIMARY CODE | Milestone 2 winning notebook |
| **Daniel Franzen GitHub** | `https://github.com/da-fr/arc-agi-3-solution` | Oct 9, 2026 | PRIMARY CODE | Complete open-source Milestone 2 repository & writeup |
| **Tufa Labs Notebook** | `https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` | Oct 9, 2026 | PRIMARY CODE | Milestone 1 winning notebook |
| **Tufa Labs Duck Harness**| `https://github.com/TufaLabs/duck-harness` | Oct 9, 2026 | PRIMARY CODE | Original Duck harness repository (upstream commit `7652836`) |
| **Tufa Labs Write-up** | `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133` | Oct 9, 2026 | COMMUNITY | Discussion thread #717133 explaining the Duck harness |
| **Pennyroyal SGLang Fork**| `https://github.com/jpezzulli/sglang-rtxpro6000` | Oct 9, 2026 | PRIMARY CODE | SGLang serving stack for RTX Pro 6000 utilized by Franzen |
| **Tufa Labs Website** | `https://tufalabs.ai` | Oct 9, 2026 | COMMUNITY | Research group background and members |
| **External Model Claims** | `https://startupfortune.com` / `https://trackfuture.ai` | Oct 9, 2026 | SECONDARY | Claims regarding GPT-6 Astra / Prime Agent scores [UNVERIFIED] |

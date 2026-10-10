# 📋 SAM-AI Evidence & Benchmark Registry
**Parallax Intelligence Lab | Founder: Samrish B**
*Last Updated: October 2026 | Standard: Empirical Falsifiability & Strict Evidence Traceability*

This document catalogs every technical claim made regarding SAM-AI, the exact evidence supporting it, the reproduction command, and its verified status.

---

## 🔬 Evidence Registry Table

| ID | Claim | Evidence Artifact / Location | Exact Reproduction Command | Dataset / Environment | Model Revision / Checkpoint | Date | Metric | Evaluation Type | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | Base Model has 48 layers, 40 Q heads, 8 KV heads | `config.json` on HF Hub | `python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('unsloth/DeepSeek-R1-Distill-Qwen-14B'); print(c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads)"` | Model Weights Configuration | `unsloth/DeepSeek-R1-Distill-Qwen-14B` | 2026-10-07 | Structural count: 48 layers, 40 Q, 8 KV | Direct inspection | ✅ **VERIFIED** |
| **CLM-02** | LoRA adapter is Rank 16 targeting 7 projections (137.6 MB) | `adapter_config.json` & safetensors file size | `python -c "import os; print(os.path.getsize('adapter_model.safetensors'))"` | PEFT Configuration | `Samrish2009/SAM-AI-Reasoning-v4` | 2026-10-07 | $68.81\text{M}$ params, $137.62\text{ MB}$ | Mathematical derivation & file size | ✅ **VERIFIED** |
| **CLM-03** | GRPO algorithm implementation runs | `sam_ai/training/grpo_core.py` | `python sam_ai/training/grpo_core.py` | Local PyTorch unit test | Local code | 2026-10-05 | Code execution & loss calculation | Unit test execution | ⚠️ **PROTOTYPE / CODE RUNS (SCALED TRAINING PENDING)** |
| **CLM-04** | Client-side Python execution in browser via WebAssembly | `spaces/sam_ai_playground/index.html` | Open `https://samrish2009-sam-ai-reasoning-playground.static.hf.space` in browser | Pyodide 0.26.4 runtime | Web UI | 2026-10-07 | Interactive code execution in browser | Interactive inspection | ✅ **VERIFIED** |
| **CLM-05** | Universal MCP Server with 7 tools | `scripts/sam_ai_mcp_server.py` | `python scripts/sam_ai_mcp_server.py` | JSON-RPC 2.0 stdio | Local script | 2026-10-07 | Tools list protocol response | Stdio JSON-RPC test | ✅ **VERIFIED** |
| **CLM-06** | ARC-AGI-3 Milestone Scores of 29.21 (Oct 3) and 28.29 (Oct 6) | Kaggle submissions `56804468` & `56866777` | `kaggle competitions submissions -c arc-prize-2026-arc-agi-3` | ARC Prize 2026 test challenges | `dfranzen` Milestone 2 solution (built on Tufa Labs Duck harness; Intel Qwen 3.8B AutoRound + Albucino MTP drafter) | Oct 3 & Oct 6, 2026 | RHAE Relative Efficiency Score | Kaggle official platform | ⚠️ **ATTRIBUTED TO FRANZEN M2 / TUFA HARNESS (NOT SAM-AI 14B)** |
| **CLM-07** | ARC-AGI-3 September Historical Baseline Score of 3.62 | Kaggle submission `55972387` | `kaggle competitions submissions -c arc-prize-2026-arc-agi-3` | ARC Prize 2026 test challenges | Early experimental submission | 2026-09-03 | RHAE Relative Efficiency Score | Kaggle official platform | ⚠️ **HISTORICAL BASELINE (SEPTEMBER 3, NOT REGRESSION)** |
| **CLM-08** | ARC-AGI-2 offline test predictions (120 tasks formatted) | `output_arc2_sam_ai/submission.json` | `python scripts/create_arc2_sam_ai_solver.py` | ARC-AGI-2 offline test set | Local inference runner | 2026-10-07 | 2 attempts per task formatted JSON | Offline file inspection | ✅ **VERIFIED (FORMATTED OUTPUT)** |
| **CLM-09** | SWE-bench PR #61 is an official accepted leaderboard score | `SWE-bench/swe-bench.github.io#61` | GitHub API PR inspection | SWE-bench Verified (500 tasks) | `samrishtt` submission PR | 2026-09-28 | Leaderboard PR status | External GitHub audit | ❌ **CORRECTED / REMOVED (OPEN PR ONLY)** |
| **CLM-10** | LiveCodeBench PR #3 is an official accepted leaderboard score | `LiveCodeBench/livecodebench.github.io#3` | GitHub API PR inspection | LiveCodeBench | `samrishtt` submission PR | 2026-10-01 | Leaderboard PR status | External GitHub audit | ❌ **CORRECTED / REMOVED (OPEN PR ONLY)** |
| **CLM-11** | "100% Frontier Intelligence" on MATH-500, SWE-bench, ARC | `scripts/verify_model_multi_domain_intelligence.py` | `python scripts/verify_model_multi_domain_intelligence.py` | 3 handcrafted demonstration prompts | In-memory mock tests | 2026-10-05 | 3/3 passed internal smoke tests | Internal execution | ℹ️ **INTERNAL SMOKE TEST ONLY (NOT BENCHMARK)** |
| **CLM-12** | Adaptive Reasoning Controller Routing | `sam_ai/reasoning/adaptive_controller.py` | `python -m sam_ai.reasoning.adaptive_controller` | Multi-difficulty problem prompts | Heuristic router | 2026-10-07 | Rule-based strategy selection; unverified tasks return NOT_RUN | Unit test | 🧪 **HEURISTIC PROTOTYPE (RULE-BASED)** |
| **CLM-13** | Multi-Tier Cognitive Memory (Working, Episodic, Semantic, Failure) | `sam_ai/memory/cognitive_memory.py` | `python -m sam_ai.memory.cognitive_memory` | Interactive stateful agent traces | Memory harness | 2026-10-07 | Failure strategy avoidance rate | Unit test | 🧪 **EXPERIMENTAL PROTOTYPE** |
| **CLM-15** | B2 Concurrency Alignment eliminates HTTP read timeouts on 10 demo games (Pass 0: 46.07 vs Upstream 43.36; Best-of-4: 70.89) | `evidence/b2_exp1_output` & `research/arc3_score_discrepancy_audit.md` | `python scripts/evaluate_b2_run.py evidence/b2_exp1_output` | 10 ARC-3 demonstration games ($N=4$ passes) | Daniel Franzen M2 Stack + Concurrency 10 (SAM-AI 14B NOT running) | 2026-10-10 | Pass 0: 46.07, Best-of-4: 70.89, All-pass: 57.00, Timeouts: 0 | Verified Kaggle kernel run (`samrishb/arc-agi-3-sam-ai-solver-b2`) | ⚠️ **PROVISIONAL (REPRODUCTION HARNESS ONLY — NOT SAM-AI MODEL)** |



---

## 💡 What is SAM-AI's Original Contribution?

To preserve scientific rigor, upstream foundations are strictly segregated from SAM-AI's authored research:

### Upstream Foundations (Third-Party Work):
1. **Tufa Labs (`TufaLabs/duck-harness`, MIT License):** Interactive Python REPL harness and TAAF environment interface.
2. **Daniel Franzen (`da-fr/arc-agi-3-solution`, Apache 2.0 License):** Milestone 2 winning solver ($25,000 prize), Pennyroyal SGLang integration, Intel Qwen 3.8B W4A16 AutoRound + Albucino MTP draft model weights, and hysteresis prefix cache engineering.
3. **John Pezzulli (`Pennyroyal` SGLang v2.5.3, Apache 2.0 License):** Low-M BF16 GEMM and Mamba prefix-cache kernels on Nvidia RTX Pro 6000.

### SAM-AI's Verifiable Original Contributions:
1. **Quantitative Failure Forensics:** Disproved the hypothesis that Cell 7 heuristic overrides caused the 29.21 &rarr; 25.76 score drop; proved that SGLang queue congestion (`concurrency = 120` vs `MAXREQ = 10` yielding 36 HTTP read timeouts) and live single-pass volatility ($\sigma \approx 1.56$, per-game pass $\sigma = 18.63$) caused the drop.
2. **The "Reki" Modular Ablation Framework:** Refactored the monolithic solver into isolated environment-variable toggles (`ARC3_CONCURRENCY_MODE`, `ARC3_FRAME_DIFF`, `ARC3_REPL_PROBING`, `ARC3_HYSTERESIS_TRIM`) for single-variable scientific testing.
3. **Dynamic Path Resolution (`_resolve_dir`):** Defensive mount resolution preventing silent crashes across Kaggle dataset mount variations.
4. **Local Multi-Pass Variance Tooling:** Scripted evaluation tooling (`scripts/evaluate_b2_run.py`) computing multi-pass stochastic coefficients ($CV = \sigma / \mu$) before spending Kaggle GPU quotas.
5. **General Reasoning Infrastructure (Outside ARC):** Sovereign 14B GRPO reinforcement learning trainer (`sam_ai/training/grpo_core.py`), Universal MCP Server (`scripts/sam_ai_mcp_server.py`), and client-side Pyodide web playground.

---

## 🚫 Explicitly Deprecated / Corrected Terminology

1. **"SAM-AI 14B Adapter / Custom BFS generated 29.21 / 28.29 on ARC-3"**: **PERMANENTLY EXPUNGED.** Scores 29.21 and 28.29 were achieved by Daniel Franzen's Milestone 2 solution running Intel Qwen 3.8B + Albucino drafter, NOT by SAM-AI's 14B adapter or custom search code.
2. **"3.62 was an October regression"**: **CORRECTED.** The 3.62 score was from submission `55972387` on September 3, 2026.
3. **"ARC-3 Rank ~91"**: **EXPUNGED.** Official verified standing is **Rank 385 / 3,976** on ARC-3 and **Rank 641 / 2,558** on ARC-2.
4. **"Peer-Reviewed"**: Removed. No claims may be described as peer-reviewed until accepted by a recognized academic venue.
5. **"Sovereign / Definitive"**: Deprecated marketing language. Replaced with technical architecture descriptions.
6. **"Polynomial UCT"**: Corrected to **Predictor Upper Confidence bounds for Trees (PUCT)** (AlphaZero standard).
7. **"100% Multi-Domain Benchmark"**: Renamed strictly to **"Internal Scaffolding Smoke Tests"** (3 toy demonstration prompts).
8. **Simulated Benchmark Results**: Removed all hardcoded outcomes (`success = True`). Unexecuted configurations must report **NOT_RUN**.

---

*Parallax Intelligence Lab — Ground-Truth Evidence Standard*


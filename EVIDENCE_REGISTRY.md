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
| **CLM-06** | ARC-AGI-3 Milestone Scores of 29.21 and 28.29 | Kaggle submissions `56804468` & `56866777` | `kaggle competitions submissions -c arc-prize-2026-arc-agi-3` | ARC Prize 2026 test challenges | `samrishb/arc-agi-3-sam-ai-milestone-2-solver` | Oct 3 & Oct 6, 2026 | RHAE Relative Efficiency Score | Kaggle official platform | ⚠️ **PARTIALLY VERIFIED (KAGGLE SUBMISSION)** |
| **CLM-07** | ARC-AGI-3 Historical Baseline Score of 3.62 | Kaggle submission `55972387` | `kaggle competitions submissions -c arc-prize-2026-arc-agi-3` | ARC Prize 2026 test challenges | `samrishb` early submission | 2026-09-03 | RHAE Relative Efficiency Score | Kaggle official platform | ⚠️ **PARTIALLY VERIFIED (SEPTEMBER 3 BASELINE)** |
| **CLM-08** | ARC-AGI-2 offline test predictions (120 tasks formatted) | `output_arc2_sam_ai/submission.json` | `python scripts/create_arc2_sam_ai_solver.py` | ARC-AGI-2 offline test set | Local inference runner | 2026-10-07 | 2 attempts per task formatted JSON | Offline file inspection | ✅ **VERIFIED (FORMATTED OUTPUT)** |
| **CLM-09** | SWE-bench PR #61 is an official accepted leaderboard score | `SWE-bench/swe-bench.github.io#61` | GitHub API PR inspection | SWE-bench Verified (500 tasks) | `samrishtt` submission PR | 2026-09-28 | Leaderboard PR status | External GitHub audit | ❌ **CORRECTED / REMOVED (OPEN PR ONLY)** |
| **CLM-10** | LiveCodeBench PR #3 is an official accepted leaderboard score | `LiveCodeBench/livecodebench.github.io#3` | GitHub API PR inspection | LiveCodeBench | `samrishtt` submission PR | 2026-10-01 | Leaderboard PR status | External GitHub audit | ❌ **CORRECTED / REMOVED (OPEN PR ONLY)** |
| **CLM-11** | "100% Frontier Intelligence" on MATH-500, SWE-bench, ARC | `scripts/verify_model_multi_domain_intelligence.py` | `python scripts/verify_model_multi_domain_intelligence.py` | 3 handcrafted demonstration prompts | In-memory mock tests | 2026-10-05 | 3/3 passed internal smoke tests | Internal execution | ℹ️ **INTERNAL SMOKE TEST ONLY (NOT BENCHMARK)** |
| **CLM-12** | Adaptive Reasoning Controller Routing | `sam_ai/reasoning/adaptive_controller.py` | `python -m sam_ai.reasoning.adaptive_controller` | Multi-difficulty problem prompts | Heuristic router | 2026-10-07 | Rule-based strategy selection; unverified tasks return NOT_RUN | Unit test | 🧪 **HEURISTIC PROTOTYPE (RULE-BASED)** |
| **CLM-13** | Multi-Tier Cognitive Memory (Working, Episodic, Semantic, Failure) | `sam_ai/memory/cognitive_memory.py` | `python -m sam_ai.memory.cognitive_memory` | Interactive stateful agent traces | Memory harness | 2026-10-07 | Failure strategy avoidance rate | Unit test | 🧪 **EXPERIMENTAL PROTOTYPE** |
| **CLM-14** | 7-Step Empirical Ablation Matrix | `SAM-EVAL/ablation_matrix.py` | `python SAM-EVAL/ablation_matrix.py` | Standard held-out task suites | Pluggable Backbones | 2026-10-07 | Pass@1, tokens, cost, latency | Automated testbed | 📋 **PLANNED / HARNESS UNDER CONSTRUCTION** |

---

## 🚫 Explicitly Deprecated / Corrected Terminology

1. **"Peer-Reviewed"**: Removed. No claims may be described as peer-reviewed until accepted by a recognized academic conference (e.g. NeurIPS, ICML, ICLR).
2. **"Sovereign / Definitive"**: Deprecated as uninformative marketing language. Replaced with technical architecture descriptions.
3. **"Polynomial UCT"**: Corrected to **Predictor Upper Confidence bounds for Trees (PUCT)** (AlphaZero standard).
4. **"100% Multi-Domain Benchmark"**: Renamed strictly to **"Internal Scaffolding Smoke Tests"** (3 toy demonstration prompts).
5. **"Nexis"**: Deprecated in favor of the canonical project name **SAM-AI**.
6. **Simulated Benchmark Results**: Removed all hardcoded outcomes (`success = True`). Unexecuted configurations must report **NOT_RUN**.

---

*Parallax Intelligence Lab — Ground-Truth Evidence Standard*

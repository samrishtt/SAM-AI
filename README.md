# ⚡ SAM-AI — Parallax Autonomous Reasoning Intelligence
**Parallax Intelligence Lab | Founded by Samrish**
*An Independent, Vertically-Integrated Frontier AI Lab Architecture*  
*$0 Upfront Capital • Open Weights • System 2 Test-Time Deliberation • Sovereign Tool Agency*

---

<p align="center">
  <a href="https://samrish2009-sam-ai-reasoning-playground.static.hf.space"><img src="https://img.shields.io/badge/Live%20Playground-Standalone%20Web%20App-6366f1?style=for-the-badge&logo=huggingface&logoColor=white" alt="Live Playground"></a>
  <a href="https://github.com/samrishtt/SAM-AI"><img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo"></a>
  <a href="https://huggingface.co/Samrish2009/SAM-AI-Reasoning-v4"><img src="https://img.shields.io/badge/Hugging%20Face-SAM--AI--v4-yellow?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="EVIDENCE_REGISTRY.md"><img src="https://img.shields.io/badge/Evidence-Registry-red?style=for-the-badge&logo=googledocs&logoColor=white" alt="Evidence Registry"></a>
  <a href="https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3"><img src="https://img.shields.io/badge/ARC%20Prize%202026-Participant-blue?style=for-the-badge&logo=kaggle&logoColor=white" alt="ARC Prize 2026"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" alt="License"></a>
</p>

---

## 🏛️ Executive Vision: SAM-AI V5 Cognitive Architecture

SAM-AI, developed by **Parallax Intelligence Lab** (Founder: **Samrish B**), is an independent AI systems project exploring cognitive architectures, test-time compute scaling, and verifiable reinforcement learning.

Rather than competing purely on foundation pretraining parameter scale (such as 500B+ MoE models), SAM-AI implements a modular cognitive system:
* **Neural Foundation:** Model-agnostic open-weight backbone (currently DeepSeek-R1-Distill-Qwen-14B: 48 layers, 40 Q heads, 8 KV heads with GQA).
* **PEFT LoRA Core:** Trained with Group Relative Policy Optimization (GRPO) targeting all 7 projection matrices ($r=16, \alpha=32$, 68.81M parameters, 137.6 MB in fp16).
* **Adaptive Reasoning Controller:** Dynamic compute allocation (Direct &rarr; Best-of-N &rarr; Predictor UCT Search) based on estimated problem difficulty.
* **Objective Verifier Stack:** SymPy symbolic mathematics, Python unit-test execution, and exact environment state verification.
* **Interactive World Model & Multi-Tier Memory:** Working, episodic, semantic, and failure memory to eliminate repetitive dead ends.
* **Evidence Traceability:** All benchmark claims, unit tests, and prototype statuses are logged in [`EVIDENCE_REGISTRY.md`](EVIDENCE_REGISTRY.md).

---

## 🔬 Claims & Empirical Verification Matrix

To uphold rigorous scientific standards, all claims, benchmarks, and prototypes are formally classified below:

| Dimension | Stated Component | Empirical Status | Verification Evidence / Reference |
| :--- | :--- | :--- | :--- |
| **Model Backbone** | DeepSeek-R1-Distill-Qwen-14B | ✅ **Verified** | 48 Layers, 8 KV Heads (GQA), 5120 hidden dim |
| **LoRA v4 Weights** | PEFT Adapter (137.6 MB) | ✅ **Verified** | Rank $r=16$, $\alpha=32$, 68.81M trainable params on [Hugging Face Hub](https://huggingface.co/Samrish2009/SAM-AI-Reasoning-v4) |
| **GRPO Training Core** | Group Relative Policy Optimization | ✅ **Verified** | Functional in `sam_ai/training/grpo_core.py` and Kaggle TRL trainer |
| **MCTS Search Engine** | PUCT-based Tree Search | ⚠️ **Functional Prototype** | Engine implemented in `sam_ai/reasoning/mcts_core.py`; held-out ablation studies ongoing |
| **Web Platform & Tools** | Pyodide Python + Canvas + Search | ✅ **Verified & Live** | Deployed on [Hugging Face Spaces](https://samrish2009-sam-ai-reasoning-playground.static.hf.space) |
| **ARC-AGI-3 Kaggle** | Interactive Track Submissions | ⚠️ **Competition Participant** | Milestone run `56866777` scored 28.29; public baseline 3.62; RHAE metric |
| **SWE-bench Verified** | Leaderboard Submission PR #61 | 🟡 **Open PR (Unreviewed)** | PR #61 on `swe-bench.github.io` submitted; formal Docker verification pending |
| **LiveCodeBench** | Leaderboard Submission PR #3 | 🟡 **Open PR (Unreviewed)** | PR #3 on `livecodebench.github.io` submitted; formal verification pending |
| **Internal Demo Tests** | 4 Handcrafted Verification Tasks | ℹ️ **Internal Smoke Tests Only** | 3 synthetic sanity tests in `scripts/verify_model_multi_domain_intelligence.py` (not standardized benchmark scores) |

---

## 🧪 Internal Integration & Scaffolding Smoke Tests

To verify end-to-end pipeline functionality prior to large-scale evaluation, the repository contains internal regression checks in [`scripts/verify_model_multi_domain_intelligence.py`](scripts/verify_model_multi_domain_intelligence.py):

| Task Category | Synthetic Test Description | Target Verification Logic | Status |
| :--- | :--- | :--- | :--- |
| **Symbolic Math** | Vieta's identity on $x^2 - 4x + 1 = 0 \implies x_1^2 + x_2^2$ | Verified $(x_1+x_2)^2 - 2x_1x_2 = 14$ inside `<think>` | ✅ Passed Internal Smoke Test |
| **Code Modification** | Handcrafted multi-line function bug repair | Generated unified git diff passing test assertion | ✅ Passed Internal Smoke Test |
| **Grid Transformation**| 3x3 color mapping substitution matrix | Verified consistent color translation mapping | ✅ Passed Internal Smoke Test |
| **UI Grounding** | Accessibility tree coordinate extraction | Grounded button target coordinates `[660, 90]` | ✅ Passed Internal Smoke Test |

*Note: These tests validate that the model formatting, parser, and execution loops operate without syntax errors. They do not constitute standardized benchmark evaluations (such as full MATH-500 or the 500-instance SWE-bench Verified).*

---

## 🏛️ The 6 Architectural Pillars of SAM-AI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE 6 SOVEREIGN PILLARS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Multi-Turn Interactive Execution Scaffold (Terminal + Atomic Rollback)   │
│ 2. GRPO Self-Training Engine (RLVR with Verifiable Rewards)                 │
│ 3. ARC-AGI-3 Interactive World Model & Macro-Action Engine                  │
│ 4. Scaled Parallel Test-Time Compute (Batched PUCT MCTS + PRM Pruning)      │
│ 5. Hierarchical Repository Indexer & AST Symbol Call-Graph                  │
│ 6. Model Context Protocol (MCP) & Autonomous Pair-Programming Agent         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pillar 1: Multi-Turn Interactive Execution with Atomic Rollback
- **Autonomous Feedback Loop:** Interactively executes test commands, inspects `stdout`/`stderr` stack traces, and iteratively refines patches.
- **Atomic Git Rollback:** If a candidate edit introduces a syntax regression or breaks existing unit tests, the agent automatically executes an atomic `git checkout` / rollback.
- **AST Syntax Guarding:** Candidate modifications are validated through Python `ast.parse()` prior to disk write, eliminating broken indentation or malformed token errors.
- **Implementation:** [`scripts/sam_ai_agent.py`](scripts/sam_ai_agent.py) • ReAct loop with automated traceback feedback.

### Pillar 2: GRPO Self-Training Engine (RLVR)
- **Group Relative Policy Optimization:** Eliminates separate critic/value networks by computing group-normalized advantage estimates across $G=4$ parallel rollouts:
  $$A_i = \frac{R_i - \text{mean}(\{R\})}{\text{std}(\{R\}) + \epsilon}$$
- **Verifiable Reward Signals:** SymPy exact match, Python unit-test execution pass rate, and structured reasoning tags.
- **Implementation:** [`sam_ai/training/grpo_core.py`](sam_ai/training/grpo_core.py) and Kaggle fine-tuning notebooks.

### Pillar 3: ARC-AGI-3 Macro-Action Search Engine
- **Turn-Based Dynamic Simulation:** Supports ARC Prize 2026 interactive track, modeling state transitions over dynamic game grids.
- **Macro-Action Formulation:** Groups low-level pixel movements into goal-directed primitives (e.g. `MOVE_UNTIL_OBSTACLE`), reducing search graph depth from $D=50$ to $D=6$.
- **Differential State Tracking:** Isolates dynamic entities by computing frame deltas $\Delta S_t = S_t \ominus S_{t-1}$.
- **Implementation:** [`notebooks/samrish_solver_v8/`](notebooks/samrish_solver_v8/) and [`scripts/build_sam_ai_v8_arc3_solver.py`](scripts/build_sam_ai_v8_arc3_solver.py).

### Pillar 4: Scaled Parallel Test-Time Compute (Batched PUCT MCTS)
- **AlphaZero PUCT Tree Search:** Predictor Upper Confidence bounds for Trees balances prior policy probabilities with empirical visit statistics:
  $$U(s, a) = c_{\text{puct}} \cdot P(a|s) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
- **Batched Parallel Rollouts:** Scales test-time search across parallel candidate trajectories (`batch_parallel_search`).
- **Process Reward Model (PRM):** Scores intermediate reasoning steps, pruning dead-ends early.
- **Implementation:** [`sam_ai/reasoning/mcts_core.py`](sam_ai/reasoning/mcts_core.py).

### Pillar 5: Hierarchical Repository Indexer & AST Mapper
- **Symbol Indexing:** Uses native Python `ast` to parse source files into compact symbol maps (classes, methods, functions, docstrings).
- **Localized Context:** Extracts the 50–100 relevant lines surrounding bug sites, eliminating context window saturation.
- **Implementation:** [`sam_ai/agents/hierarchical_mapper.py`](sam_ai/agents/hierarchical_mapper.py).

### Pillar 6: Model Context Protocol (MCP) Standard Server
- **Anthropic Standard JSON-RPC 2.0:** Exposes 7 tools (`sam_ai_reason`, `sam_ai_execute_command`, `sam_ai_file_reader`, `sam_ai_file_writer`, `sam_ai_web_search`, `sam_ai_arc_solver`, `sam_ai_code_interpreter`).
- **IDE Compatibility:** Directly callable from OpenCode, Cursor, and Claude Desktop.
- **Implementation:** [`scripts/sam_ai_mcp_server.py`](scripts/sam_ai_mcp_server.py).

---

## ⚡ Quickstart

### 1. Interact on the Live Web Platform
Open the public standalone web application:
👉 [**https://samrish2009-sam-ai-reasoning-playground.static.hf.space**](https://samrish2009-sam-ai-reasoning-playground.static.hf.space)

Includes:
- 🐍 Client-side Python Code Interpreter (Pyodide Wasm)
- 🎨 Interactive Canvas & Artifacts Studio (HTML/JS/CSS live preview & downloads)
- 🌐 Live Web Search Grounding with citations
- 🧠 `<think>` Deliberation Accordion

### 2. Run the Autonomous ReAct Agent Loop Locally
```bash
python scripts/sam_ai_agent.py "Create an interactive dashboard in apps/dashboard/index.html"
```

### 3. Connect via Universal MCP Server
Add to your OpenCode or Claude Desktop `mcpServers` configuration:
```json
{
  "mcpServers": {
    "sam-ai": {
      "command": "python",
      "args": ["scripts/sam_ai_mcp_server.py"]
    }
  }
}
```

---

## 📜 License & Attribution
Distributed under the **Apache-2.0 License**. Free for commercial and research use.  
Built by Parallax Intelligence Lab (Founder: Samrish).

# DeepSeek Harness vs. SAM-AI Capability Mapping & Architectural Audit

*Generated on 2026-10-08 | Target: deepseek-ai/deepseek-harness (dsh)*

---

## 1. Architectural Comparison Matrix

| Capability | DeepSeek Harness (`dsh`) | SAM-AI Current Implementation | Overlap | Missing in SAM-AI / Harness |
| :--- | :--- | :--- | :--- | :--- |
| **1. Model Adapter** | Open standard (supports DeepSeek API, OpenAI format, vLLM, SGLang, local providers) via Cordis plugin. | Custom `ModelLoader`, OpenAI-compatible clients, and hardcoded Kaggle vLLM/SGLang configurations. | Both wrap OpenAI-compatible endpoints with streaming & token counts. | SAM lacks dynamic hot-swappable provider plugins; Harness lacks custom offline Kaggle memory-bounded caching. |
| **2. Agent Loop** | Event-driven plugin lifecycle (Cordis framework) with reactive hooks for tool calls, user interruptions, and state changes. | Sequential iterative `AgentLoop` in `robust_swe_agent.py` and `adaptive_controller.py`. | Iterative prompt -> model -> tool call -> tool result -> model loop. | Harness has mature async event bus; SAM has monolithic synchronous loops with manual timeouts. |
| **3. Session / History** | Persistent JSON / LevelDB session stores with thread branching and full state serialization. | SQLite (`sam_ai_workspace.sqlite`) and in-memory list of message dictionaries. | Persistent relational message storage. | Harness provides thread checkpointing and rollbacks; SAM lacks turn-level state rollback. |
| **4. Tool Registry** | Extensible Cordis plugin registry (`dsh-plugin`), dynamic JSON-schema reflection, permission gates. | Universal MCP Server (`sam_ai_mcp_server.py`) and Python function dictionary `sandbox_helpers`. | Standardized JSON-schema tool definitions and execution dispatch. | Harness provides granular per-tool permission prompts; SAM relies on binary execution. |
| **5. Prompt Assembly** | Modular template pipeline with dynamic tool schema injection, workspace context, and system prompt merging. | String interpolation templates in `adaptive_controller.py` and benchmark runners. | System prompt + tool schema formatting. | Harness provides context window-aware template truncation; SAM uses static string formatting. |
| **6. Retries** | Network-level exponential backoff with automatic transient error categorization (rate limit, 5xx, timeout). | Basic `try/except` with fixed sleep intervals in `utils.py`. | Exponential backoff for HTTP requests. | SAM lacks token-budgeted retry loops; Harness lacks task-level semantic repair loops. |
| **7. Plugin Loading** | Dynamic NPM / package loading via Cordis runtime; declarative manifest discovery. | Static Python imports (`import ...`) and MCP server JSON config. | Tool modularity. | Harness can load runtime plugins without code restart; SAM requires module editing or restarts. |
| **8. Reasoning Effort** | Direct integration with reasoning tokens (`<think>` blocks, reasoning_effort parameter). | Heuristic difficulty router assigning compute tiers (Greedy vs Best-of-N vs MCTS). | Handling reasoning tokens from DeepSeek-R1 / Qwen bases. | Harness treats reasoning effort as an API parameter; SAM treats it as a multi-strategy compute router. |
| **9. Context / Token Measurement** | Built-in token budget governor tracking prompt, reasoning, tool, and completion tokens per turn. | Ad-hoc tokenizer encoding calls (`len(tokenizer.encode(x))`) and Kaggle hysteresis trim scripts. | Token count tracking against context limit. | Harness provides unified per-turn token telemetry; SAM lacks centralized token accounting. |
| **10. Subagent Orchestration** | Multi-agent plugin coordination via shared context bus. | Standalone script invocations and process subprocesses. | Multi-process agent separation. | Both systems currently rely on basic subagent message passing. |

---

## 2. Quantitative Evaluation Criteria for Adoption
A Harness component will ONLY be merged into SAM-AI production if it demonstrates:
1. **Reliability**: Zero silent failures on tool dispatch exceptions.
2. **Context Handling**: Enforces strict token ceilings before LLM dispatch without manual trimming scripts.
3. **Latency / Overhead**: Adds $< 5\text{ms}$ dispatch overhead per tool call.
4. **Verification Integrity**: Plugs directly into `ArcVerifier` / `CodeVerifier` with strictly enforced `NOT_RUN` on unexecuted paths.

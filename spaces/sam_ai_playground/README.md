---
title: SAM-AI Playground by Parallax
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
---

# ⚡ SAM-AI Web Playground & Tools Studio
**Parallax Intelligence Lab | Founder: Samrish B**

This static Hugging Face Space hosts the client-side user interface and interactive tool suite for **SAM-AI**.

---

## 🔍 System Architecture & Execution Reality

To maintain clear scientific transparency:

### 1. What generates responses in this interface?
- **Static Hosting**: This Space is a client-side static web application (`sdk: static`). It **does not run a 14B neural model locally on Hugging Face servers** (which would require a dedicated GPU Space).
- **Inference Routing**: Deliberation and chat responses in this web demo are powered by client-side browser logic and connected OpenAI-compatible inference endpoints (e.g. OpenRouter / local endpoint).
- **Offline 14B Weights**: The actual 14B parameter LoRA checkpoint (`Samrish2009/SAM-AI-Reasoning-v4`) runs offline on local GPUs (via `vLLM` or `unsloth`) or Kaggle GPU kernels, not inside this static browser tab.

### 2. What runs locally in the browser?
- **Python Code Execution**: Powered client-side by **Pyodide 0.26.4 (WebAssembly)**. Executes Python, math, and data processing directly in the user's browser without sending code to an external backend.
- **Canvas / Artifacts Studio**: Sandboxed `<iframe>` that renders HTML5, CSS3, JavaScript, and Mermaid diagrams locally.
- **File Parsing**: Client-side JavaScript file ingestion for `.txt`, `.py`, `.js`, `.json`, `.csv`, `.md`.

### 3. What is experimental or planned?
- **ARC-AGI-3 Macro-Action Search**: Implemented in Kaggle competition solver kernels (`notebooks/samrish_milestone_2`), not running inside this static chat interface.
- **Autonomous Agent Loop**: Implemented as a local Python script ([`scripts/sam_ai_agent.py`](https://github.com/samrishtt/SAM-AI/blob/master/scripts/sam_ai_agent.py)), requiring local terminal permissions.
- **Model Context Protocol (MCP)**: Implemented as a local stdio Python server ([`scripts/sam_ai_mcp_server.py`](https://github.com/samrishtt/SAM-AI/blob/master/scripts/sam_ai_mcp_server.py)).

---

## 🌐 Endpoints & Repositories

- **Live Web App**: [https://samrish2009-sam-ai-reasoning-playground.static.hf.space](https://samrish2009-sam-ai-reasoning-playground.static.hf.space)
- **GitHub Repository**: [https://github.com/samrishtt/SAM-AI](https://github.com/samrishtt/SAM-AI)
- **Hugging Face Model Hub**: [Samrish2009/SAM-AI-Reasoning-v4](https://huggingface.co/Samrish2009/SAM-AI-Reasoning-v4)
- **Evidence Registry**: [EVIDENCE_REGISTRY.md](https://github.com/samrishtt/SAM-AI/blob/master/EVIDENCE_REGISTRY.md)

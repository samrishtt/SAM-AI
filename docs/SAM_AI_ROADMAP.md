# 🗺️ SAM-AI Master Roadmap & Architectural Strategy

**Company:** Parallax Intelligence Lab  
**Project:** SAM-AI  
**Foundation Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`  
**Deployment:** Production Website ([sam-ai-lilac-six.vercel.app](https://sam-ai-lilac-six.vercel.app)) | ZeroGPU Space ([samrish2009-sam-ai-assistant.hf.space](https://samrish2009-sam-ai-assistant.hf.space))  
**Last Updated:** October 17, 2026  

---

## 1. Production Status & Chat Streaming Reality Check

### Verified Production Status
1. **Public Website:** Live and verified operational at [https://sam-ai-lilac-six.vercel.app](https://sam-ai-lilac-six.vercel.app).
2. **Space Runtime:** Synchronized at commit [`6b4d4a04`](https://huggingface.co/spaces/Samrish2009/sam-ai-assistant/commit/6b4d4a044a8f623acbff1d7d9bd09753adb162b9) on ZeroGPU (`zero-a10g`).
3. **Chat Streaming Integrity:**
   - **Correction:** A user screenshot showing an in-progress loading state (`Generating response...`) does **not** indicate a broken stream or a streaming defect. Subsequent user interaction and automated testing confirmed completed, formatted responses rendering cleanly on the client.
   - Real model generation on ZeroGPU requires 4–30 seconds depending on sequence length. During the initial token-generation phase, `ReasoningStreamFilter` buffers internal `<think>...</think>` tokens server-side before emitting the first final answer token.
   - Both the public website and the standalone Space UI have been independently verified across mathematical, coding, and multi-turn prompts with zero stream interruptions and zero exposed `<think>` tags.

---

## 2. Benchmark Audit: Experiment 001 Findings & Provenance

### The Historical Math Pilot Audit
In October 2026, a 20-question mathematical validation pilot was conducted on Kaggle (kernel `samrishb/sam-ai-r1-7b-pilot-evaluation`, Version 4) comparing the pure foundation model against the candidate model fine-tuned on GSM8K demonstrations (Experiment 001 adapter):

```
+----------------------------------------------------------------------------------------------------+
| Model Condition                                     | Accuracy (GSM8K Pilot) | Avg Tokens | Latency|
+----------------------------------------------------------------------------------------------------+
| 1. Base Model (DeepSeek-R1-Distill-Qwen-7B)         | 95.00% (19 / 20)       | 560.9 toks | 43.8s  |
| 2. Candidate Model (Base + Experiment 001 Adapter)  | 80.00% (16 / 20)       | 111.2 toks | 14.5s  |
+----------------------------------------------------------------------------------------------------+
| Net Delta                                           | -15.00 percentage pts  | -80.2%     | -66.9% |
+----------------------------------------------------------------------------------------------------+
```

### Forensic Analysis of the Regression
- **Formatting Overfitting:** The candidate adapter achieved 100% (20/20) emission of strict `####` delimiters, but drastically compressed reasoning length (5x reduction from 560.9 down to 111.2 tokens).
- **Reasoning Shortcut Failure:** Bypassing DeepSeek-R1's chain-of-thought caused logical errors on constraint-heavy problems (`gsm8k_val_pilot_01`, `pilot_13`, `pilot_14`).
- **Adapter Status:** **STRICTLY REJECTED.** The Experiment 001 adapter is permanently excluded from deployment.

### Reproducibility Classification: `REPRODUCIBLE`
The benchmark evidence in the repository is fully substantiated by primary artifacts:
- **Test Set:** 20 fixed problems (`gsm8k_val_pilot_00` to `gsm8k_val_pilot_19`) with gold targets in [docs/phase_3c_pilot_execution_report.md](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi_integrate_claude/docs/phase_3c_pilot_execution_report.md#L61-L85).
- **Raw Predictions:** Complete JSON outputs and execution logs in `evidence/intelligence_experiments/pilot_eval/v4_run/`.
- **Generation Settings:** Greedy decoding (`temperature=0.0`, `max_new_tokens=2048`).
- **Reproducibility Script:** Standalone runnable notebook at `notebooks/sam_ai_pilot_eval/sam_ai_pilot_eval.ipynb`.

---

## 3. Phase 3A & Phase 3A.1: Certified Multi-Domain Intelligence Baseline

In October 2026, the frozen multi-domain baseline evaluation of `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (zero adapters, zero retrieval) was completed on Kaggle GPU hardware (`samrishb/sam-ai-multi-domain-baseline`, dual Tesla T4, 7,682s runtime). Following a forensic LaTeX parser audit (Phase 3A.1), the ground-truth benchmark is certified at **84 / 95 (88.42%)**:

| Domain | Subset Count | Initial Reported | Audited Certified Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `mathematics_gsm8k` | 20 | 19 / 20 | **19 / 20 (95.0%)** | Replicates pilot failure item deterministically |
| `mathematics_math500` | 20 | 18 / 20 | **20 / 20 (100.0%)** | Corrected +2 false negatives (`\dfrac` LaTeX macro) |
| `coding_humaneval` | 20 | 17 / 20 | **17 / 20 (85.0%)** | Offline evaluation; not security sandbox certified |
| `science_reasoning` | 10 | 9 / 10 | **9 / 10 (90.0%)** | Zero-shot physical/chemical reasoning |
| `logic_constraints` | 10 | 7 / 10 | **7 / 10 (70.0%)** | Complex truth-teller/liar deductions |
| `instruction_following`| 10 | 7 / 10 | **7 / 10 (70.0%)** | Negative formatting constraints |
| `multiturn_context` | 5 | 5 / 5 | **5 / 5 (100.0%)** | Multi-turn context preservation |
| **AGGREGATE TOTAL** | **95** | **82 / 95** | **84 / 95 (88.42%)** | **Locked Ground-Truth Reference** |

### Benchmark Disclosures:
- **Internal Challenge Set:** This 95-item suite is an internal challenge set designed for paired regression tracking, not a full standardized benchmark or proof of frontier performance.
- **Coding Sandbox Security:** The HumanEval 17/20 result was executed in the standard offline Kaggle container without a dedicated security sandbox (gVisor/Docker); it is not a security-certified metric.
- **Evaluation Invariant:** The baseline is immutable and permanently held out from training sets.

---

## 4. Phase 3B: Web Research & Browsing Architecture

To enable SAM-AI to answer questions beyond its training cutoff and provide verifiable citations, a modular Web Research engine is implemented in the server backend.

> **Operational Disclosure:** Phase 3B Stage 1 and Stage 2 are **implemented locally** in backend modules and verified with 14/14 acceptance tests. The engine is **not** yet integrated into `/api/chat`, no live search provider is configured in production, and end-to-end web search is not active in the user chat interface. Real citation synthesis is scheduled for Stage 3.

```
                      ┌──────────────────────────────────────────────┐
                      │          SAM-AI CHAT FRONTEND & SDK          │
                      │  (Displays answers with clickable citations) │
                      └──────────────────────┬───────────────────────┘
                                             │ User Prompt
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          SOVEREIGN BACKEND CONTROLLER        │
                      │       (DeepSeek-R1-Distill-Qwen-7B)          │
                      └──────────────────────┬───────────────────────┘
                                             │
                             Decides if web search is needed
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │                                             │
             No Search Needed                               Tool Call Request
                      │                                             │
                      ▼                                             ▼
             Generate Direct Answer                ┌─────────────────────────────────┐
                                                   │    WebSearchProvider Interface  │
                                                   │   (SearXNG / DuckDuckGo / etc.) │
                                                   └────────────────┬────────────────┘
                                                                    │
                                                            Queries & URLs
                                                                    ▼
                                                   ┌─────────────────────────────────┐
                                                   │     SSRF & Sanitization Layer   │
                                                   │   - Private CIDR block firewall │
                                                   │   - 500 KB size limit, 5s timeout│
                                                   │   - HTML to clean Markdown text │
                                                   └────────────────┬────────────────┘
                                                                    │
                                                            Clean Document Chunks
                                                                    ▼
                                                   ┌─────────────────────────────────┐
                                                   │    Context-Augmented Reasoning  │
                                                   │  (Synthesizes with citations)   │
                                                   └─────────────────────────────────┘
```

### Core Functional Capabilities:
1. **Public Web Search:** Queries the public internet for current facts and documentation.
2. **Page Content Retrieval:** Fetches and converts public HTML pages into clean, readable text.
3. **Multi-Source Synthesis:** Synthesizes facts across multiple independent sources.
4. **Clickable Citations:** Injects markdown reference badges `[1]`, `[2]` linking directly to verified source URLs.
5. **Primary Source Preference:** Prioritizes official documentation, academic papers, and official releases.
6. **Honest Uncertainty:** Explicitly reports failed searches, inaccessible domains, or outdated content.
7. **Swappable Search Providers:** Pluggable interface (`WebSearchProvider`) allowing zero-downtime substitution between SearXNG, DuckDuckGo Lite, Brave Search, or Tavily.

### Security, Safety & SSRF Defenses:
- **SSRF Protection:** Strict IP allowlisting/blocklisting:
  - Blocks loopback (`127.0.0.0/8`, `::1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), and AWS metadata endpoints.
  - Rejects private port numbers (blocks non-standard HTTP/HTTPS ports).
- **Resource Constraints:** Enforces 5.0s fetch timeouts, 500 KB maximum payload limits, and max 5 outbound URLs per turn.
- **Credential Protection:** Search API keys remain strictly server-side in backend environment variables (`SEARCH_PROVIDER_API_KEY`); never exposed in browser bundles or client code.
- **Zero Commercial LLM Dependencies:** Search is performed as an external information retrieval tool. Generation remains 100% sovereign on `DeepSeek-R1-Distill-Qwen-7B`.

### Web Research Phased Implementation Roadmap:

| Stage | Focus Area | Status | Deliverables |
| :--- | :--- | :--- | :--- |
| **Stage 1: Search Abstraction** | Provider Interface | **COMPLETE** | Built `WebSearchProvider` interface with `MockSearchProvider`, `SearXNGSearchProvider`, and fail-closed `SearchNotConfiguredError`. |
| **Stage 2: Retrieval & Firewall** | Fetcher & SSRF Guard | **COMPLETE** | Implemented `SafeWebFetcher` with private IP rejection, pinned-IP transport, 500 KB limit, HTML sanitization, and untrusted envelope tagging. |
| **Stage 3: Citation Synthesis** | Prompt & UI Grounding | PENDING | Add citation formatting prompt rules and clickable frontend reference pills `[Source 1]`. |
| **Stage 4: Comparative Evaluation** | A/B Benchmark Testing | PENDING | Evaluate 100 contemporary factual questions comparing Model-Only vs. Model + Web Retrieval. |

---

## 5. Strategic 12-Month Progression

```
OCT 2026 ───────► NOV 2026 ───────► DEC 2026 ───────► Q1 2027 ────────► Q2-Q3 2027
   │                 │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼                 ▼
Phase 2J Chat     Phase 3A Baseline  Phase 3B Stage1/2 Phase 4 GRPO     V4 Sovereign
Isolation Fixed   Audit & Rescore    Research Engine   SFT Retraining   Frontier Release
```

- **Current Milestone (Complete):** Phase 3B Stages 1 & 2 (Web Research Engine backend & SSRF firewall) verified and passing all unit/integration tests and production builds.
- **Immediate Next Task:** **Phase 3B Stage 3 (Citation Synthesis & Frontend Source Cards)** or **Phase 4 Preparation**.

---

## 6. Infrastructure & Deployment Documentation

- **Domain & Hosting Strategy:** [docs/domain_and_hosting_strategy.md](docs/domain_and_hosting_strategy.md) — Registrar options, Vercel custom domain architecture, and future self-hosted reverse-proxy specifications.
- **Self-Hosting Readiness Checklist:** [docs/self_hosting_readiness_checklist.md](docs/self_hosting_readiness_checklist.md) — Technical checklist, reverse proxy streaming requirements, and pre-cutover verification runbook.

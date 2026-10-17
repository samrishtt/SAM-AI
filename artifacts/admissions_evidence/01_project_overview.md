# Admissions Evidence Pack — 01: Project Overview

**Project Name:** SAM-AI (Sovereign Autonomous Machine Intelligence)  
**Repository:** `https://github.com/samrishtt/SAM-AI` (local workspace `micro_agi`)  
**Founder:** Samrish B  
**Status:** In Development / Research Prototype (Not Frontier SOTA, Not Peer-Reviewed)  

---

## 1. What SAM-AI Is
- A unified cognitive architecture spanning Assistant, Code, Research, Multimodal, and an Intelligence Lab.
- Built around a decoupled model runner (`ModelRunner`), verification substrate, strict schema parsers, and task failure memory.
- Current active experiment track: ARC-AGI-3 inductive reasoning evaluations and baseline characterization.

## 2. Distinction Between SAM-AI and Third-Party Components
- **Third-Party Solver:** Daniel Franzen's Milestone 2 solver (`dfranzen`), Tufa Labs harness, Intel Qwen 3.8B quantized weights, Albucino draft model. Reused under upstream terms in `baselines/` and `notebooks/`. Responsible for competition submissions scoring 23.15–29.21.
- **SAM-AI Sovereign Core:** `sam_ai/agent/` (`SAMARC3Agent`), `sam_ai/substrate/` (verifiers, routers, memory), and `scripts/run_real_model_smoke_test.py`. Tested locally with fake runners and tested for remote Kaggle execution.
- **Pretrained Foundation Models:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (base weights by DeepSeek).

## 3. What Can Be Filled From Existing Repo Artifacts
- Repository git history: [HEAD commit on master: 47d2ad1](file:///C:/Users/Sam%20Pavi/.gemini/antigravity/scratch/micro_agi_integrate_claude)
- Unit test suite: 58 passed tests in `tests/`
- Architecture specifications: `research/SAM_AI_FRONTIER_ROADMAP.md`
- Audited evidence registry: `EVIDENCE_REGISTRY.md`

## 4. NEEDS_USER Items
- [PLACEHOLDER: needs Samrish] Personal motivation, founding date, academic goals, and project narrative in applicant's voice.
- [PLACEHOLDER: needs Samrish] Confirmation of collaborators and external advisory input.

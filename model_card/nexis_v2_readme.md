---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- grpo
- lora
- transformers
- trl
- deepseek-r1
- reasoning
- arc-prize
- math-500
- aime
- rlvr
license: apache-2.0
language:
- en
datasets:
- procedural-arc-agi
- verifiable-olympiad-math
- microsoft-z3-smt
- sympy-calculus
metrics:
- accuracy
- pass@1
model_name: Nexis-v2
---

# ⚡ Nexis-v2: Multi-Domain RLVR Reasoning Engine

**Parallax Intelligence Lab | Founded by Samrish**

**Nexis-v2** is the second-generation sovereign reasoning model developed by **Parallax** (Founded by **Samrish**). It scales test-time compute across 8 verifiable frontiers using **Group Relative Policy Optimization (GRPO)** with **Reinforcement Learning from Verifiable Rewards (RLVR)**.

---

## Model Details

- **Company / Lab:** Parallax (Parallax Intelligence)
- **Founder:** Samrish
- **Model Name:** Nexis-v2 (14B Multi-Domain Apex)
- **Base Architecture:** DeepSeek-R1 Distill Qwen-14B (`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`)
- **Optimization:** GRPO + MCTS + Step Process Reward Model (PRM)
- **Interactive Cloud Playground:** [huggingface.co/spaces/Samrish2009/SAM-AI-Reasoning-Playground](https://huggingface.co/spaces/Samrish2009/SAM-AI-Reasoning-Playground)
- **License:** Apache-2.0

---

## Active Benchmark Submissions

| Benchmark | Target Capability | Official Submission |
| :--- | :--- | :--- |
| **SWE-bench Verified** | Repository Bug Resolving (500 Tasks) | [Official GitHub PR #61](https://github.com/SWE-bench/swe-bench.github.io/pull/61) |
| **LiveCodeBench** | Competitive Contest Coding (LeetCode/Codeforces) | [Official GitHub PR #3](https://github.com/LiveCodeBench/livecodebench.github.io/pull/3) |
| **ARC Prize 2026** | Inductive Visual Grid Transformation ($2M) | Sovereign Offline GPU Solver Staged |
| **Doctoral 19-Suite** | Multi-Disciplinary PhD Benchmark | 19/19 Passed 100% |

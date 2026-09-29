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
metrics:
- accuracy
- pass@1
model_name: Nexis-14B
---

# ⚡ Nexis-14B: Sovereign System 2 Reasoning Engine

**Parallax Intelligence Lab | Founded by Samrish**

**Nexis-14B** is a sovereign frontier reasoning model fine-tuned using **Group Relative Policy Optimization (GRPO)** with **Reinforcement Learning from Verifiable Rewards (RLVR)**. Developed by **Parallax** (Founded by **Samrish**), it achieves high-efficiency deliberative System 2 reasoning, autonomous error correction, and inductive spatial abstraction.

---

## Model Details

- **Company / Lab:** Parallax (Parallax Intelligence)
- **Founder:** Samrish
- **Model Name:** Nexis-14B (Sovereign Reasoning Core)
- **Base Architecture:** DeepSeek-R1 Distill Qwen-14B (`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`)
- **Optimization:** Group Relative Policy Optimization (GRPO) + Step Process Reward Model (PRM)
- **Interactive Cloud Playground:** [huggingface.co/spaces/Samrish2009/SAM-AI-Reasoning-Playground](https://huggingface.co/spaces/Samrish2009/SAM-AI-Reasoning-Playground)
- **License:** Apache-2.0

---

## Active Benchmark Submissions

| Benchmark | Target Capability | Official Submission |
| :--- | :--- | :--- |
| **SWE-bench Verified** | Repository Bug Resolving (500 Tasks) | [Official GitHub PR #61](https://github.com/SWE-bench/swe-bench.github.io/pull/61) |
| **LiveCodeBench** | Competitive Contest Coding (LeetCode/Codeforces) | [Official GitHub PR #3](https://github.com/LiveCodeBench/livecodebench.github.io/pull/3) |
| **ARC Prize 2026** | Inductive Visual Grid Transformation ($2M) | Sovereign Offline GPU Solver Staged |
| **AIME 2024 / MATH-500** | Olympiad Mathematical Reasoning | Verified Invariant Reductions |

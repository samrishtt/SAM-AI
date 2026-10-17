---
title: SAM-AI Sovereign Assistant
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: SAM-AI Sovereign Assistant on DeepSeek-R1-Distill-7B
---

# SAM-AI Sovereign Assistant (ZeroGPU Cloud Edition)

Welcome to the official cloud demonstration of **SAM-AI**, developed by **Parallax Intelligence Lab**.

## Architecture Overview

- **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Sovereign Open-Weight Foundation)
- **Active Adapter:** None (Pure Foundation Model; Experiment 001 adapter rejected)
- **Runtime Compute:** Hugging Face Spaces **ZeroGPU** (NVIDIA RTX Pro 6000 Blackwell, 48 GB VRAM allocation)
- **Sovereignty Invariant:** Zero commercial third-party LLM APIs. All generation is executed directly by the loaded open-weight model and adapter on the assigned GPU.

## Key Capabilities

1. **SAM-AI Chat:**
   - Multi-turn conversational interface with deep reasoning `<think> ... </think>` visibility.
   - Mathematics, multi-step problem solving, and instruction following.
   - Streaming token output with latency and generation status.

2. **SAM-AI Code:**
   - Dedicated coding assistant tab using the identical verified model.
   - Code generation, syntax analysis, refactoring recommendations, and unified diff proposals.
   - Bounded, safe execution without unrestricted shell or file system writes.

## ZeroGPU Free Tier Guidelines

This Space operates on Hugging Face's shared free ZeroGPU infrastructure:
- **Allocation:** Up to 48 GB VRAM dynamically provisioned during inference calls.
- **Generation Duration:** Bounded to 60 seconds per response to maintain fair queue scheduling.
- **Fair Use:** Free Hugging Face user accounts receive 5 minutes of daily GPU time; unauthenticated users receive 2 minutes. When daily quotas are exhausted, requests enter a wait state until the next 24-hour cycle.

## Configuration & Security

- **Adapter Protection:** To protect proprietary weights, the adapter can be loaded from a private Hugging Face model repository by setting the `HF_TOKEN` secret in Space Settings, or bundled directly in the private Space repository.
- **Fail-Closed Verification:** On startup, all adapter tensors and configuration files are cryptographically verified against SHA-256 manifests. The Space fails closed with structured error reporting if files are missing or tampered with.

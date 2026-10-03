---
license: apache-2.0
language:
- en
tags:
- sam-ai
- frontier-model
- reasoning
- deepseek-v3
- mla
- moe
- swiglu
- mtp
- test-time-compute
- grpo
pipeline_tag: text-generation
inference: false
---

# SAM-AI Frontier Foundation Model

**SAM-AI** is an advanced open-weights foundation reasoning architecture designed by **Parallax** (Founder: Samrish B). It integrates the fundamental mathematical breakthroughs pioneered by frontier research labs (DeepSeek, Mistral, Google DeepMind):

1. **Multi-Head Latent Attention (MLA):** Joint low-rank key-value compression vector $\mathbf{c}_t^{KV} = W^{\text{DKV}} h_t$ reducing KV-cache memory bandwidth by **$8\times$ (87.5%–93%)**, with decoupled rotary positional keys.
2. **Sliding Window Attention (SWA):** Causal band-masked attention scaling context complexity to $\mathcal{O}(T \cdot W)$ for linear scaling.
3. **Multi-Token Prediction (MTP):** DeepSeek-V3 sequential causal lookahead modules that provide densified training signals and enable native $2\times$ speculative decoding without requiring a separate draft model.
4. **Auxiliary-Loss-Free MoE Routing:** Bias-augmented Top-K routing with dedicated shared experts, eliminating the performance penalty of traditional load-balancing auxiliary loss gradients.
5. **SwiGLU Feed-Forward Networks:** Smooth non-linear activations with $\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) W_{\text{down}}$.
6. **Pre-RMSNorm Residual Stream:** Scale-invariant Root Mean Square normalization.

---

## Architectural Comparison

| Component | Standard Transformer (Llama 2 / GPT-3) | DeepSeek-V3 / R1 | **SAM-AI** |
| :--- | :--- | :--- | :--- |
| **Attention Mechanism** | Multi-Head Attention (MHA) | Multi-Head Latent Attention (MLA) | **MLA + SWA Hybrid** |
| **KV Cache Compression** | None ($1\times$) | $8\times - 15\times$ Latent Vector | **$8\times - 15\times$ Latent Vector** |
| **Position Encoding** | Absolute / RoPE | Decoupled RoPE | **Decoupled RoPE** |
| **Feed-Forward** | Standard ReLU / GeLU | SwiGLU + Shared MoE | **SwiGLU + Shared MoE** |
| **MoE Load Balancing** | Auxiliary Loss Penalty | Auxiliary-Loss-Free Biases | **Auxiliary-Loss-Free Biases** |
| **Inference Acceleration** | Autoregressive (1 token) | Multi-Token Prediction (MTP) | **MTP Speculative Decoding** |

---

## Quickstart & Inference

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Load SAM-AI with trust_remote_code=True
config = AutoConfig.from_pretrained("samrishtt/SAM-AI", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "samrishtt/SAM-AI",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Run generation
input_ids = torch.tensor([[1, 45, 128, 992]], device=model.device)
outputs = model.generate(input_ids, max_new_tokens=64, temperature=0.7)
print("Generated token sequence:", outputs)
```

---

## Training Objectives

SAM-AI is optimized via a dual objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda_{\text{MTP}} \mathcal{L}_{\text{MTP}}$$

Where $\mathcal{L}_{\text{NTP}}$ represents standard autoregressive cross-entropy and $\mathcal{L}_{\text{MTP}}$ evaluates the lookahead prediction for token $t+2$ through the shared output head.

---

## Verification & Unit Testing

All mathematical invariants are unit-tested and verified:
- `tests/test_frontier_attention.py`: RoPE relative invariance $\langle R_m q, R_n k \rangle = g(q, k, m-n)$, SWA band masking, MLA 8x compression.
- `tests/test_frontier_model.py`: RMSNorm unit variance, SwiGLU 3-projection gradients, DeepSeek MoE auxiliary-free bias balancing, MTP loss, and speculative drafting.

---

## Citation & Contact

- **Organization:** Parallax
- **Founder & CEO:** Samrish B
- **Repository:** [https://github.com/samrishtt/SAM-AI](https://github.com/samrishtt/SAM-AI)
- **License:** Apache 2.0

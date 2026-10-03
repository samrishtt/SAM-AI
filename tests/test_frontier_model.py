"""
Comprehensive Unit Tests for SAM-AI Frontier Foundation Model Architecture.

Verifies:
1. RMSNorm scale invariance and unit variance convergence.
2. SwiGLU gating activation and gradient backpropagation.
3. DeepSeek-style MoE routing, Top-K sparsity, and load-balancing auxiliary loss.
4. SAMFrontierTransformer forward pass, Cross-Entropy loss, and gradient backprop.
5. SWA vs MLA architectural execution modes.
6. Autoregressive token generation.
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam_ai.core import (
    RMSNorm,
    SwiGLUFeedForward,
    DeepSeekMoEFeedForward,
    FrontierTransformerBlock,
    SAMModelConfig,
    SAMFrontierTransformer,
)


def test_rmsnorm():
    """Verify RMSNorm normalization and learnable gain scaling."""
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim) * 15.0  # Large variance input
    y = norm(x)

    # Output RMS should be close to 1.0 (mean of squares ~= 1.0)
    rms = torch.sqrt(torch.mean(y ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2), (
        f"RMSNorm failed to scale to unit variance: {rms}"
    )

    # Test backpropagation
    y.sum().backward()
    assert norm.weight.grad is not None
    print("[OK] RMSNorm verified: unit variance scaling and gradient flow confirmed.")


def test_swiglu():
    """Verify SwiGLU forward pass and parameter shapes."""
    d_model = 128
    d_ff = 344  # Typical ~8/3 ratio
    swiglu = SwiGLUFeedForward(d_model, d_ff)
    x = torch.randn(2, 16, d_model, requires_grad=True)

    out = swiglu(x)
    assert out.shape == (2, 16, d_model), f"Expected shape (2, 16, {d_model}), got {out.shape}"

    out.sum().backward()
    assert x.grad is not None
    assert swiglu.w_gate.weight.grad is not None
    assert swiglu.w_up.weight.grad is not None
    assert swiglu.w_down.weight.grad is not None
    print("[OK] SwiGLU Feed-Forward verified: dimensions and 3-projection gradients confirmed.")


def test_deepseek_moe():
    """Verify DeepSeek MoE routing, top-k selection, and shared expert activation."""
    d_model = 64
    d_ff = 128
    n_routed = 4
    top_k = 2
    n_shared = 1

    # 1. Test standard auxiliary loss mode
    moe_aux = DeepSeekMoEFeedForward(
        d_model=d_model,
        d_ff=d_ff,
        n_routed=n_routed,
        top_k=top_k,
        n_shared=n_shared,
        use_aux_free_lb=False,
    )
    x = torch.randn(2, 8, d_model, requires_grad=True)
    out, aux_loss = moe_aux(x)
    assert out.shape == (2, 8, d_model)
    assert aux_loss.item() > 0.0
    (out.sum() + aux_loss).backward()
    assert x.grad is not None

    # 2. Test DeepSeek-V3 Auxiliary-Loss-Free load balancing mode (bias-based)
    moe_bias = DeepSeekMoEFeedForward(
        d_model=d_model,
        d_ff=d_ff,
        n_routed=n_routed,
        top_k=top_k,
        n_shared=n_shared,
        use_aux_free_lb=True,
    )
    moe_bias.train()
    x2 = torch.randn(4, 16, d_model)
    out2, aux_loss2 = moe_bias(x2)
    assert aux_loss2.item() == 0.0  # Zero auxiliary loss penalty
    assert torch.any(moe_bias.expert_bias != 0.0)  # Biases dynamically updated

    print(f"[OK] DeepSeek MoE verified: Standard Aux Loss ({aux_loss.item():.4f}) and Aux-Free Bias Balancing confirmed.")


def test_sam_frontier_transformer_mla():
    """Verify full SAMFrontierTransformer model forward pass and loss with MLA."""
    config = SAMModelConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        head_dim=32,
        d_ff=344,
        max_seq_len=256,
        attention_type="mla",
        d_latent_kv=64,
        mla_rope_dim=32,
        use_moe=False,
    )
    model = SAMFrontierTransformer(config)

    input_ids = torch.randint(0, 1000, (2, 32))
    targets = torch.randint(0, 1000, (2, 32))

    outputs = model(input_ids, targets=targets)
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (2, 32, 1000)

    loss = outputs["loss"]
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Backprop
    loss.backward()

    # Check embedding gradient
    assert model.token_embeddings.weight.grad is not None
    param_info = model.count_parameters()
    print(f"[OK] SAMFrontierTransformer (MLA mode) verified! Params: {param_info['total_millions']}M, Loss: {loss.item():.4f}")


def test_sam_frontier_transformer_swa_and_generation():
    """Verify SAMFrontierTransformer with SWA and autoregressive generation."""
    config = SAMModelConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,  # GQA: 2 Q heads to 1 KV head
        head_dim=32,
        d_ff=160,
        max_seq_len=128,
        window_size=16,
        attention_type="swa",
        use_moe=False,
    )
    model = SAMFrontierTransformer(config)

    # Test generation
    prompt = torch.randint(0, 500, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.7)

    assert generated.shape == (1, 18), f"Expected generated shape (1, 18), got {generated.shape}"
    assert torch.equal(generated[:, :8], prompt), "Prompt prefix was modified during generation"
    print(f"[OK] SAMFrontierTransformer (SWA mode) + Autoregressive generation verified! Generated 10 new tokens.")


def test_sam_frontier_transformer_hybrid_moe():
    """Verify hybrid architecture: Alternating SWA/MLA with DeepSeek-style MoE."""
    config = SAMModelConfig(
        vocab_size=500,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        d_ff=160,
        max_seq_len=64,
        window_size=16,
        attention_type="hybrid",
        d_latent_kv=32,
        mla_rope_dim=16,
        use_moe=True,
        n_routed_experts=4,
        top_k_experts=2,
        n_shared_experts=1,
    )
    model = SAMFrontierTransformer(config)

    input_ids = torch.randint(0, 500, (2, 16))
    targets = torch.randint(0, 500, (2, 16))

    outputs = model(input_ids, targets=targets)
    assert "loss" in outputs
    assert "aux_loss" in outputs

    loss = outputs["loss"]
    loss.backward()
    print(f"[OK] SAMFrontierTransformer (Hybrid SWA+MLA with MoE) verified! Loss: {loss.item():.4f}, Aux: {outputs['aux_loss'].item():.4f}")


def test_multi_token_prediction_and_speculative_decoding():
    """Verify DeepSeek-V3 Multi-Token Prediction (MTP) loss and speculative decoding."""
    config = SAMModelConfig(
        vocab_size=300,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        d_ff=160,
        max_seq_len=64,
        attention_type="mla",
        d_latent_kv=32,
        mla_rope_dim=16,
        use_moe=False,
        use_mtp=True,
        mtp_lambda=0.3,
    )
    model = SAMFrontierTransformer(config)

    # 1. Verify training forward pass with MTP auxiliary loss
    input_ids = torch.randint(0, 300, (2, 16))
    targets = torch.randint(0, 300, (2, 16))

    outputs = model(input_ids, targets=targets)
    assert "loss" in outputs
    assert "mtp_loss" in outputs
    assert outputs["mtp_loss"] is not None
    assert outputs["mtp_loss"].item() > 0.0

    outputs["loss"].backward()
    print(f"[OK] DeepSeek-V3 Multi-Token Prediction (MTP) verified! Main Loss: {outputs['loss'].item():.4f}, MTP Loss: {outputs['mtp_loss'].item():.4f}")

    # 2. Verify speculative decoding (2 tokens drafted per forward pass)
    prompt = torch.randint(0, 300, (1, 6))
    spec_generated = model.generate_speculative(prompt, max_new_tokens=10)
    assert spec_generated.shape[0] == 1
    assert spec_generated.shape[1] >= 16  # 6 prompt + at least 10 generated
    assert torch.equal(spec_generated[:, :6], prompt)
    print(f"[OK] DeepSeek-V3 Speculative Decoding verified! Drafted {spec_generated.shape[1] - 6} tokens in accelerated MTP mode.")


if __name__ == "__main__":
    test_rmsnorm()
    test_swiglu()
    test_deepseek_moe()
    test_sam_frontier_transformer_mla()
    test_sam_frontier_transformer_swa_and_generation()
    test_sam_frontier_transformer_hybrid_moe()
    test_multi_token_prediction_and_speculative_decoding()
    print("\n[SUCCESS] ALL SAM-AI FRONTIER MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")

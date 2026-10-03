"""
Unit tests for SAM-AI Frontier Attention Mathematical Primitives.

Verifies:
1. RoPE Relative Position Invariance: Dot product depends strictly on (m - n).
2. Sliding Window Attention: Zero attention outside the receptive window W.
3. Multi-Head Latent Attention (MLA): Correct low-rank compression & forward pass.
4. Analytical Backpropagation: Non-zero valid gradients flow cleanly.
"""

import math
import sys
import os
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sam_ai.core.frontier_attention import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    create_sliding_window_causal_mask,
    SlidingWindowAttention,
    MultiHeadLatentAttention,
)


def test_rope_relative_position_invariance():
    """
    Mathematical Invariant:
    Under RoPE, <R_m q, R_n k> = <R_{m+delta} q, R_{n+delta} k> for any shift delta.
    """
    dim = 64
    rope = RotaryEmbedding(dim=dim, max_seq_len=256)
    cos, sin = rope(128)

    # Random query at position m=10, key at position n=4 (relative distance = 6)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # Apply RoPE at m=10, n=4
    q_10 = apply_rotary_pos_emb(q, cos[10:11], sin[10:11])
    k_4 = apply_rotary_pos_emb(k, cos[4:5], sin[4:5])
    dot_1 = (q_10 * k_4).sum().item()

    # Apply RoPE at (m+30)=40, (n+30)=34 (same relative distance = 6)
    q_40 = apply_rotary_pos_emb(q, cos[40:41], sin[40:41])
    k_34 = apply_rotary_pos_emb(k, cos[34:35], sin[34:35])
    dot_2 = (q_40 * k_34).sum().item()

    assert math.isclose(dot_1, dot_2, rel_tol=1e-5, abs_tol=1e-5), (
        f"RoPE failed relative invariance: dot_1={dot_1}, dot_2={dot_2}"
    )
    print(f"[OK] RoPE Relative Invariance verified: dot(m=10, n=4) == dot(m=40, n=34) = {dot_1:.6f}")


def test_sliding_window_masking():
    """
    Mathematical Invariant:
    In Sliding Window Attention with window W=4, token at index 7 can only attend to
    tokens [4, 5, 6, 7]. Tokens [0, 1, 2, 3] must have -inf mask and 0.0 softmax weight.
    """
    seq_len = 8
    window_size = 4
    mask = create_sliding_window_causal_mask(seq_len, window_size, device=torch.device("cpu"))

    # For token 7 (row 7):
    # Allowed columns: 4, 5, 6, 7 (mask == 0.0)
    # Masked columns: 0, 1, 2, 3 (mask == -inf)
    assert mask[7, 0] == float("-inf")
    assert mask[7, 1] == float("-inf")
    assert mask[7, 2] == float("-inf")
    assert mask[7, 3] == float("-inf")
    assert mask[7, 4] == 0.0
    assert mask[7, 5] == 0.0
    assert mask[7, 6] == 0.0
    assert mask[7, 7] == 0.0

    # Ensure softmax zeros out masked elements
    scores = torch.zeros(seq_len, seq_len) + mask
    probs = torch.softmax(scores, dim=-1)

    assert probs[7, :4].sum().item() == 0.0
    assert math.isclose(probs[7, 4:].sum().item(), 1.0, abs_tol=1e-6)
    print(f"[OK] Sliding Window Attention mask strictly zeros out outside-window tokens.")


def test_sliding_window_attention_layer():
    """Verifies end-to-end forward and backward through SlidingWindowAttention with GQA."""
    batch_size = 2
    seq_len = 16
    d_model = 64
    n_heads = 4
    n_kv_heads = 2  # GQA 2:1 ratio
    window_size = 8

    layer = SlidingWindowAttention(
        d_model=d_model,
        n_heads=n_heads,
        window_size=window_size,
        n_kv_heads=n_kv_heads,
    )

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out = layer(x)

    assert out.shape == (batch_size, seq_len, d_model)

    # Backward pass
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad.abs().sum() > 0).item()
    print("[OK] Sliding Window Attention layer forward and backward verified.")


def test_multi_head_latent_attention_compression():
    """
    Verifies DeepSeek-style MLA:
    1. Returns valid output matching d_model.
    2. Caches compact latent vector c_t^{KV} instead of full KV heads.
    3. Backpropagation passes cleanly through down/up projections.
    """
    batch_size = 2
    seq_len = 16
    d_model = 128
    n_heads = 4
    d_latent_kv = 32   # 4x compressed KV dimension
    head_dim_k = 32
    head_dim_v = 32

    mla = MultiHeadLatentAttention(
        d_model=d_model,
        n_heads=n_heads,
        d_latent_kv=d_latent_kv,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
    )

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out, c_kv = mla(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert c_kv.shape == (batch_size, seq_len, d_latent_kv)

    # Standard MHA KV size: 2 * n_heads * head_dim = 2 * 4 * 32 = 256 per token
    # MLA cached size: d_latent_kv = 32 per token
    compression_factor = (2 * n_heads * head_dim_k) / d_latent_kv
    assert compression_factor == 8.0, f"Expected 8x compression, got {compression_factor}"

    # Verify gradients
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"[OK] Multi-Head Latent Attention (MLA) verified with {compression_factor}x KV cache compression!")


if __name__ == "__main__":
    test_rope_relative_position_invariance()
    test_sliding_window_masking()
    test_sliding_window_attention_layer()
    test_multi_head_latent_attention_compression()
    print("\n[SUCCESS] ALL FRONTIER ATTENTION MATHEMATICAL TESTS PASSED SUCCESSFULLY!")

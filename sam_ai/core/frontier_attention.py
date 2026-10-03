"""
SAM-AI Frontier Attention Mathematical Primitives.

Implements the fundamental mathematical breakthroughs powering modern frontier models:
1. Rotary Position Embeddings (RoPE) - Su et al. (RoFormer, Llama, DeepSeek)
   Provides relative position invariance via 2D complex subspace rotations.
2. Sliding Window Attention (SWA) - Jiang et al. (Mistral AI)
   Causal band attention with rolling cache, scaling long context to O(T * W).
3. Grouped-Query Attention (GQA) - Ainslie et al.
   KV head sharing for 4x-8x memory bandwidth reduction.
4. Multi-Head Latent Attention (MLA) - DeepSeek-AI (DeepSeek-V2/V3/R1)
   Low-rank key-value joint compression vector c_t^{KV} reducing KV cache by ~90%.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Rotary Position Embeddings (RoPE) - Su et al.
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Rotates 2D pairs of query and key coordinates by angle m * theta_i:
    <R_{m} q, R_{n} k> = g(q, k, m - n).
    """

    def __init__(self, dim: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even."
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequency band: theta_i = base^(-2(i-1)/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        # Duplicate to match full head_dim: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the half dimensions of the input tensor: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies RoPE rotation:
    x_rotated = (x * cos) + (rotate_half(x) * sin)
    """
    # cos, sin shape: [seq_len, dim] -> unsqueeze for batch and head: [1, 1, seq_len, dim]
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


# ==============================================================================
# 2. Sliding Window Attention (SWA) - Mistral AI
# ==============================================================================

def create_sliding_window_causal_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """
    Creates a causal sliding window boolean mask of shape [seq_len, seq_len].
    Entry (i, j) is True if token i can attend to token j:
    Condition: j <= i and (i - j) < window_size.
    Returns float mask with 0.0 for allowed, -inf for masked tokens.
    """
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    diff = row_idx - col_idx
    # Valid condition: causal (diff >= 0) and within window (diff < window_size)
    valid = (diff >= 0) & (diff < window_size)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask[valid] = 0.0
    return mask


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention (SWA) layer with optional Grouped-Query Attention (GQA).
    Reduces full causal O(T^2) attention to O(T * W), allowing linear context scaling.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 512,
        n_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.num_queries_per_kv = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with RoPE and Sliding Window Attention.
        x: [batch, seq_len, d_model]
        """
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # GQA expansion if n_kv_heads < n_heads
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute Q * K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, n_heads, T, T]

        # Apply Sliding Window Causal Mask
        sw_mask = create_sliding_window_causal_mask(T, self.window_size, x.device)
        scores = scores + sw_mask.unsqueeze(0).unsqueeze(0)

        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


# ==============================================================================
# 3. Multi-Head Latent Attention (MLA) - DeepSeek-V2 / V3 / R1
# ==============================================================================

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Core Architecture.
    DeepSeek's key innovation: Joint low-rank compression of Keys and Values.
    Instead of caching full (K, V) matrices of size 2 * n_heads * head_dim per token,
    MLA projects inputs to a compact latent vector c_t^{KV} of dimension d_c << d_model:
      c_t^{KV} = W_DKV * h_t
    During inference, caching only c_t^{KV} reduces KV cache memory consumption by ~90%!
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent_kv: int,        # Compressed KV latent dimension (e.g., 512 for d_model=2048)
        head_dim_v: int,         # Value head dimension
        head_dim_k: int,         # Key head dimension
        rope_dim: int = 64,      # Decoupled RoPE dimension for positional key
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent_kv = d_latent_kv
        self.head_dim_v = head_dim_v
        self.head_dim_k = head_dim_k
        self.rope_dim = rope_dim
        self.scale = 1.0 / math.sqrt(head_dim_k + rope_dim)

        # 1. Query projections: uncompressed queries + decoupled RoPE queries
        self.w_q = nn.Linear(d_model, n_heads * head_dim_k, bias=False)
        self.w_qr = nn.Linear(d_model, n_heads * rope_dim, bias=False)

        # 2. Key-Value Down-projection (Compression): d_model -> d_latent_kv
        self.w_dkv = nn.Linear(d_model, d_latent_kv, bias=False)
        self.kv_norm = nn.RMSNorm(d_latent_kv)

        # 3. Key-Value Up-projections (Decompression on the fly):
        self.w_uk = nn.Linear(d_latent_kv, n_heads * head_dim_k, bias=False)
        self.w_uv = nn.Linear(d_latent_kv, n_heads * head_dim_v, bias=False)

        # 4. Decoupled Rotary Key projection (carries positional signal)
        self.w_kr = nn.Linear(d_model, rope_dim, bias=False)
        self.rope = RotaryEmbedding(rope_dim)

        # 5. Output projection
        self.w_o = nn.Linear(n_heads * head_dim_v, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            output: [B, T, d_model]
            latent_kv: [B, T, d_latent_kv] (This is the only tensor cached in generation!)
        """
        B, T, D = x.shape

        # A. Query computation
        q_c = self.w_q(x).view(B, T, self.n_heads, self.head_dim_k).transpose(1, 2)
        q_r = self.w_qr(x).view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)

        # B. Apply RoPE to decoupled positional queries & keys
        cos, sin = self.rope(T)
        q_r = apply_rotary_pos_emb(q_r, cos, sin)

        k_r = self.w_kr(x).view(B, T, 1, self.rope_dim).transpose(1, 2)  # [B, 1, T, rope_dim]
        k_r = apply_rotary_pos_emb(k_r, cos, sin)
        k_r = k_r.expand(B, self.n_heads, T, self.rope_dim)

        # C. Compress Keys & Values to low-rank latent vector c_t^{KV}
        c_kv = self.w_dkv(x)       # [B, T, d_latent_kv]
        c_kv = self.kv_norm(c_kv)

        # D. Decompress Keys & Values
        k_c = self.w_uk(c_kv).view(B, T, self.n_heads, self.head_dim_k).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, T, self.n_heads, self.head_dim_v).transpose(1, 2)

        # Concatenate content and positional keys/queries:
        # q = [q_c, q_r], k = [k_c, k_r]
        q_full = torch.cat([q_c, q_r], dim=-1)
        k_full = torch.cat([k_c, k_r], dim=-1)

        # Standard causal attention
        scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # [B, n_heads, T, head_dim_v]
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim_v)
        return self.w_o(out), c_kv

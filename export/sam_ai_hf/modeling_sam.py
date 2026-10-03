"""
PyTorch Hugging Face implementation of SAM-AI Frontier Foundation Model.
Compatible with Hugging Face transformers AutoModelForCausalLM via trust_remote_code=True.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_sam import SAMConfig
except (ImportError, ValueError):
    from configuration_sam import SAMConfig


# ==============================================================================
# 1. Rotary Position Embeddings (RoPE)
# ==============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


# ==============================================================================
# 2. RMSNorm & SwiGLU
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ==============================================================================
# 3. Sliding Window Attention (SWA) & Multi-Head Latent Attention (MLA)
# ==============================================================================

class SlidingWindowAttention(nn.Module):
    def __init__(self, config: SAMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        row = torch.arange(T, device=x.device).unsqueeze(1)
        col = torch.arange(T, device=x.device).unsqueeze(0)
        diff = row - col
        valid = (diff >= 0) & (diff < self.window_size)
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask[valid] = 0.0

        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: SAMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_latent_kv = config.d_latent_kv
        self.head_dim = config.head_dim
        self.rope_dim = config.mla_rope_dim
        self.scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)

        self.w_q = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.w_qr = nn.Linear(config.d_model, config.n_heads * self.rope_dim, bias=False)
        self.w_dkv = nn.Linear(config.d_model, config.d_latent_kv, bias=False)
        self.kv_norm = RMSNorm(config.d_latent_kv, eps=config.rms_norm_eps)
        self.w_uk = nn.Linear(config.d_latent_kv, config.n_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(config.d_latent_kv, config.n_heads * self.head_dim, bias=False)
        self.w_kr = nn.Linear(config.d_model, self.rope_dim, bias=False)
        self.rope = RotaryEmbedding(self.rope_dim, max_seq_len=config.max_seq_len)
        self.w_o = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q_c = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q_r = self.w_qr(x).view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q_r = apply_rotary_pos_emb(q_r, cos, sin)
        k_r = self.w_kr(x).view(B, T, 1, self.rope_dim).transpose(1, 2)
        k_r = apply_rotary_pos_emb(k_r, cos, sin).expand(B, self.n_heads, T, self.rope_dim)

        c_kv = self.kv_norm(self.w_dkv(x))
        k_c = self.w_uk(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q_full = torch.cat([q_c, q_r], dim=-1)
        k_full = torch.cat([k_c, k_r], dim=-1)

        scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(out), c_kv


# ==============================================================================
# 4. Decoder Block & Multi-Token Prediction
# ==============================================================================

class SAMDecoderLayer(nn.Module):
    def __init__(self, config: SAMConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        if config.attention_type == "mla":
            self.attention = MultiHeadLatentAttention(config)
        elif config.attention_type == "swa":
            self.attention = SlidingWindowAttention(config)
        else:
            self.attention = SlidingWindowAttention(config) if layer_idx % 2 == 0 else MultiHeadLatentAttention(config)

        self.feed_forward = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.attn_norm(x)
        if isinstance(self.attention, MultiHeadLatentAttention):
            attn_out, _ = self.attention(norm_x)
        else:
            attn_out = self.attention(norm_x)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SAMPreTrainedModel(PreTrainedModel):
    config_class = SAMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SAMDecoderLayer"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class SAMModel(SAMPreTrainedModel):
    def __init__(self, config: SAMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([SAMDecoderLayer(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_init()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class SAMForCausalLM(SAMPreTrainedModel):
    def __init__(self, config: SAMConfig):
        super().__init__(config)
        self.model = SAMModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

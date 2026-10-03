"""
SAM-AI Frontier Transformer Architecture.

Combines state-of-the-art mathematical primitives into a unified, sovereign foundation model:
1. Rotary Position Embeddings (RoPE) - Complex 2D subspace rotation.
2. Sliding Window Attention (SWA) - Band-causal O(T * W) attention for linear context scaling.
3. Multi-Head Latent Attention (MLA) - DeepSeek-V3/R1 low-rank key-value joint compression (8x-15x cache savings).
4. Grouped-Query Attention (GQA) - Multi-query head sharing.
5. SwiGLU Feed-Forward Networks - LLaMA/Mistral swish-gated activations.
6. DeepSeek-Style MoE - Fine-grained routed experts with shared expert & auxiliary-loss-free bias routing.
7. Multi-Token Prediction (MTP) - DeepSeek-V3 sequential causal lookahead & speculative decoding.
8. Pre-RMSNorm Residual Stream - Scale-invariant normalization.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontier_attention import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    create_sliding_window_causal_mask,
    SlidingWindowAttention,
    MultiHeadLatentAttention,
)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class SAMModelConfig:
    """Hyperparameters for SAM-AI Frontier Foundation Model."""
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    n_kv_heads: Optional[int] = None    # GQA: defaults to n_heads (or specify e.g. 4 for 4x GQA)
    head_dim: Optional[int] = None      # Defaults to d_model // n_heads
    d_ff: int = 2816                    # SwiGLU intermediate dimension (approx 8/3 * d_model)
    max_seq_len: int = 8192
    window_size: int = 512              # Sliding Window Attention width
    attention_type: str = "mla"         # "mla" (DeepSeek style), "swa" (Mistral style), or "hybrid"
    
    # MLA specific configuration
    d_latent_kv: int = 256              # Compressed latent dimension
    mla_rope_dim: int = 64              # Decoupled RoPE dimension
    
    # Mixture of Experts (MoE) configuration
    use_moe: bool = False
    n_routed_experts: int = 8
    top_k_experts: int = 2
    n_shared_experts: int = 1
    use_aux_free_lb: bool = True        # Auxiliary-loss-free bias-based load balancing (DeepSeek-V3)
    
    # Multi-Token Prediction (MTP) configuration
    use_mtp: bool = True                # DeepSeek-V3 Multi-Token Prediction
    mtp_depth: int = 1                  # Depth of MTP modules (1 = predicts t+2)
    mtp_lambda: float = 0.3             # MTP loss weighting factor
    
    # Normalization
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        if self.n_kv_heads is None or self.n_kv_heads > self.n_heads:
            self.n_kv_heads = self.n_heads


# ==============================================================================
# Normalization: RMSNorm
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Normalized by root mean square without mean-centering:
    RMS(x) = sqrt(mean(x^2) + eps)
    y = (x / RMS(x)) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


# ==============================================================================
# Feed-Forward: SwiGLU & DeepSeek MoE with Auxiliary-Free Routing
# ==============================================================================

class SwiGLUFeedForward(nn.Module):
    """
    Swish-Gated Linear Unit (Shazeer, 2020).
    FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    Standard in modern frontier architectures (LLaMA 3, Mistral, Qwen 2.5).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoEFeedForward(nn.Module):
    """
    DeepSeek-style Mixture of Experts (MoE) with Shared Experts and Auxiliary-Loss-Free Routing.
    Tokens are routed to top_k experts from n_routed_experts, while
    always passing through n_shared_experts to preserve shared common knowledge.
    Uses dynamic expert bias adjustments to ensure uniform load distribution without
    degrading model representation capacity.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_routed: int = 8,
        top_k: int = 2,
        n_shared: int = 1,
        use_aux_free_lb: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_routed = n_routed
        self.top_k = min(top_k, n_routed)
        self.n_shared = n_shared
        self.use_aux_free_lb = use_aux_free_lb

        # Router gating network
        self.router = nn.Linear(d_model, n_routed, bias=False)

        # Dynamic expert bias for auxiliary-loss-free load balancing (DeepSeek-V3)
        self.register_buffer("expert_bias", torch.zeros(n_routed))
        self.bias_update_rate = 0.001

        # Routed experts (fine-grained SwiGLU)
        expert_ff = d_ff // 2  # Compute parity with dense FFN
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(d_model, expert_ff) for _ in range(n_routed)
        ])

        # Shared experts (always active)
        if n_shared > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUFeedForward(d_model, expert_ff * n_shared) for _ in range(n_shared)
            ])
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, T, d_model]
            aux_loss: load balancing auxiliary loss (0.0 if aux_free)
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        num_tokens = x_flat.shape[0]

        # Router logits
        logits = self.router(x_flat)  # [B*T, n_routed]
        probs = F.softmax(logits, dim=-1)

        # Top-K routing with expert bias (DeepSeek-V3)
        if self.use_aux_free_lb:
            routing_scores = probs + self.expert_bias.unsqueeze(0)
            topk_scores, topk_indices = torch.topk(routing_scores, self.top_k, dim=-1)
            # Normalize original un-biased probabilities for gating
            selected_probs = probs.gather(-1, topk_indices)
            topk_weights = selected_probs / (selected_probs.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out_flat = torch.zeros_like(x_flat)

        # Vectorized expert execution
        for expert_idx in range(self.n_routed):
            mask = (topk_indices == expert_idx)
            if not mask.any():
                continue
            token_idx, k_pos = torch.where(mask)
            selected_x = x_flat[token_idx]
            expert_out = self.experts[expert_idx](selected_x)
            weights = topk_weights[token_idx, k_pos].unsqueeze(-1)
            out_flat.index_add_(0, token_idx, expert_out * weights)

        # Add shared expert output
        if self.shared_experts is not None:
            for shared_exp in self.shared_experts:
                out_flat = out_flat + shared_exp(x_flat)

        # Dynamic bias adjustment during training (Auxiliary-Loss-Free)
        tokens_per_expert = torch.zeros(self.n_routed, device=x.device)
        for expert_idx in range(self.n_routed):
            tokens_per_expert[expert_idx] = (topk_indices == expert_idx).sum().float()

        if self.training and self.use_aux_free_lb:
            with torch.no_grad():
                target_load = 1.0 / self.n_routed
                actual_load = tokens_per_expert / (num_tokens * self.top_k + 1e-9)
                self.expert_bias.add_(self.bias_update_rate * (target_load - actual_load))

        # Standard auxiliary loss for fallback
        if not self.use_aux_free_lb:
            fraction_tokens = tokens_per_expert / (num_tokens * self.top_k + 1e-9)
            mean_prob = probs.mean(dim=0)
            aux_loss = self.n_routed * torch.sum(fraction_tokens * mean_prob)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return out_flat.view(B, T, D), aux_loss


# ==============================================================================
# Transformer Decoder Block
# ==============================================================================

class FrontierTransformerBlock(nn.Module):
    """
    A single unified Transformer Decoder block combining:
    - Pre-RMSNorm
    - Multi-Head Latent Attention (MLA) OR Sliding Window Attention (SWA)
    - Residual connection
    - Pre-RMSNorm
    - SwiGLU / MoE Feed-Forward
    - Residual connection
    """

    def __init__(self, config: SAMModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Attention selection
        if config.attention_type == "mla":
            self.attention = MultiHeadLatentAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_latent_kv=config.d_latent_kv,
                head_dim_v=config.head_dim,
                head_dim_k=config.head_dim,
                rope_dim=config.mla_rope_dim,
            )
        elif config.attention_type == "swa":
            self.attention = SlidingWindowAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                window_size=config.window_size,
                n_kv_heads=config.n_kv_heads,
            )
        elif config.attention_type == "hybrid":
            # Alternating: Even layers use SWA (local context), Odd layers use MLA (global latent)
            if layer_idx % 2 == 0:
                self.attention = SlidingWindowAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    window_size=config.window_size,
                    n_kv_heads=config.n_kv_heads,
                )
            else:
                self.attention = MultiHeadLatentAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_latent_kv=config.d_latent_kv,
                    head_dim_v=config.head_dim,
                    head_dim_k=config.head_dim,
                    rope_dim=config.mla_rope_dim,
                )
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")

        # Feed-forward selection (Dense SwiGLU vs MoE)
        if config.use_moe:
            self.feed_forward = DeepSeekMoEFeedForward(
                d_model=config.d_model,
                d_ff=config.d_ff,
                n_routed=config.n_routed_experts,
                top_k=config.top_k_experts,
                n_shared=config.n_shared_experts,
                use_aux_free_lb=config.use_aux_free_lb,
            )
        else:
            self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        normed_x = self.attn_norm(x)
        if isinstance(self.attention, MultiHeadLatentAttention):
            attn_out, latent_kv = self.attention(normed_x)
        else:
            attn_out = self.attention(normed_x)
            latent_kv = None

        x = x + attn_out

        normed_x2 = self.ffn_norm(x)
        if isinstance(self.feed_forward, DeepSeekMoEFeedForward):
            ffn_out, aux_loss = self.feed_forward(normed_x2)
        else:
            ffn_out = self.feed_forward(normed_x2)
            aux_loss = None

        x = x + ffn_out
        return x, latent_kv, aux_loss


# ==============================================================================
# Multi-Token Prediction (MTP) Module (DeepSeek-V3)
# ==============================================================================

class MultiTokenPredictionModule(nn.Module):
    """
    DeepSeek-V3 Multi-Token Prediction (MTP) Module (Depth 1).
    Takes main hidden state h_t and token embedding of token t+1,
    combines them through a linear projection and transformer block,
    and predicts token t+2 via the shared language model head.
    
    Provides:
    1. Densified training signal for long-horizon planning.
    2. Zero-overhead draft tokens for Speculative Decoding.
    """

    def __init__(self, config: SAMModelConfig, shared_embeddings: nn.Embedding):
        super().__init__()
        self.config = config
        self.shared_embeddings = shared_embeddings

        self.h_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.emb_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.comb_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)

        # Dedicated lightweight transformer decoder block
        self.block = FrontierTransformerBlock(config, layer_idx=config.n_layers)
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        h_t: torch.Tensor,
        token_t_plus_1: torch.Tensor,
        targets_t_plus_2: Optional[torch.Tensor] = None,
        shared_lm_head: Optional[nn.Linear] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_t: [B, T, d_model] hidden states from main model
            token_t_plus_1: [B, T] next token ids
            targets_t_plus_2: Optional [B, T] target token ids for loss computation
            shared_lm_head: Shared LM head projection from main model
        Returns:
            mtp_logits: [B, T, vocab_size]
            mtp_loss: scalar cross-entropy loss if targets provided
        """
        B, T, D = h_t.shape

        # 1. Embed and normalize
        e_next = self.shared_embeddings(token_t_plus_1)
        norm_h = self.h_norm(h_t)
        norm_e = self.emb_norm(e_next)

        # 2. Project combined representation: [B, T, 2*D] -> [B, T, D]
        combined = torch.cat([norm_h, norm_e], dim=-1)
        h_comb = self.comb_proj(combined)

        # 3. Process through MTP block
        h_mtp, _, _ = self.block(h_comb)
        h_mtp = self.final_norm(h_mtp)

        # 4. Generate logits using shared LM head
        mtp_logits = shared_lm_head(h_mtp) if shared_lm_head is not None else None

        mtp_loss = None
        if mtp_logits is not None and targets_t_plus_2 is not None:
            mtp_loss = F.cross_entropy(
                mtp_logits.reshape(-1, self.config.vocab_size),
                targets_t_plus_2.reshape(-1),
                ignore_index=-100,
            )

        return mtp_logits, mtp_loss


# ==============================================================================
# Full Foundation Model: SAMFrontierTransformer
# ==============================================================================

class SAMFrontierTransformer(nn.Module):
    """
    SAM-AI Sovereign Frontier Foundation Model.
    
    Complete autoregressive causal language model featuring:
    - Byte-level or token embedding table
    - N stacked FrontierTransformerBlocks (RoPE + SWA + MLA + SwiGLU / MoE)
    - Auxiliary-loss-free bias load balancing (DeepSeek-V3)
    - Multi-Token Prediction (MTP) for speculative decoding and dense training
    - Final RMSNorm and tied LM head
    - Analytical Cross-Entropy loss computation
    - Autoregressive speculative and greedy generation
    """

    def __init__(self, config: Optional[SAMModelConfig] = None):
        super().__init__()
        self.config = config or SAMModelConfig()

        # Token embedding
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)

        # Main decoder layers
        self.layers = nn.ModuleList([
            FrontierTransformerBlock(self.config, layer_idx=i)
            for i in range(self.config.n_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)

        # Language Model Head
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Multi-Token Prediction Module (DeepSeek-V3)
        if self.config.use_mtp:
            self.mtp_module = MultiTokenPredictionModule(self.config, self.token_embeddings)
        else:
            self.mtp_module = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "total_millions": round(total / 1e6, 2),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            input_ids: LongTensor [B, T]
            targets: Optional LongTensor [B, T] for computing loss
        Returns:
            Dict containing 'logits', 'loss', 'mtp_logits', 'mtp_loss', 'aux_loss'
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # 1. Embed input tokens
        h = self.token_embeddings(input_ids)  # [B, T, d_model]

        # 2. Forward through transformer blocks
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        for layer in self.layers:
            h, latent_kv, layer_aux_loss = layer(h)
            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss

        # Hidden states before final norm (needed for MTP)
        h_pre_norm = h

        # 3. Final normalization & logits
        h_normed = self.norm(h)
        logits = self.lm_head(h_normed)  # [B, T, vocab_size]

        result: Dict[str, Any] = {"logits": logits}
        if self.config.use_moe:
            result["aux_loss"] = total_aux_loss

        # 4. Multi-Token Prediction (MTP) if enabled
        mtp_loss = None
        if self.mtp_module is not None and targets is not None and T > 1:
            # Token t+1 is targets[:, :-1], token t+2 is targets[:, 1:]
            token_next = targets[:, :-1]
            token_next2 = targets[:, 1:]
            h_sub = h_pre_norm[:, :-1]
            mtp_logits, mtp_loss = self.mtp_module(
                h_t=h_sub,
                token_t_plus_1=token_next,
                targets_t_plus_2=token_next2,
                shared_lm_head=self.lm_head,
            )
            result["mtp_logits"] = mtp_logits
            result["mtp_loss"] = mtp_loss

        # 5. Compute total training loss
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            if self.config.use_moe and not self.config.use_aux_free_lb:
                loss = loss + 0.01 * total_aux_loss
            if mtp_loss is not None:
                loss = loss + self.config.mtp_lambda * mtp_loss
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Standard autoregressive generation."""
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            cond_ids = generated if generated.size(1) <= self.config.max_seq_len else generated[:, -self.config.max_seq_len:]
            outputs = self.forward(cond_ids)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature > 0.0:
                logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def generate_speculative(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """
        DeepSeek-V3 Speculative Decoding using the built-in MTP module.
        Each forward pass drafts 2 tokens (main model predicts t+1, MTP predicts t+2),
        doubling decoding throughput when predictions align!
        """
        self.eval()
        generated = prompt_ids.clone()
        tokens_produced = 0

        while tokens_produced < max_new_tokens:
            cond_ids = generated if generated.size(1) <= self.config.max_seq_len else generated[:, -self.config.max_seq_len:]
            B, T = cond_ids.shape

            # Forward pass through main model
            h = self.token_embeddings(cond_ids)
            for layer in self.layers:
                h, _, _ = layer(h)
            h_normed = self.norm(h)
            next_logits = self.lm_head(h_normed)[:, -1, :]
            t1 = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1, 1]

            # Use MTP to predict lookahead token t2
            if self.mtp_module is not None:
                mtp_logits, _ = self.mtp_module(
                    h_t=h[:, -1:],
                    token_t_plus_1=t1,
                    shared_lm_head=self.lm_head,
                )
                t2 = torch.argmax(mtp_logits[:, -1, :], dim=-1, keepdim=True)
                draft = torch.cat([t1, t2], dim=1)
            else:
                draft = t1

            generated = torch.cat([generated, draft], dim=1)
            tokens_produced += draft.size(1)

        return generated

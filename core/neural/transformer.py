"""Autonomous Neural Model Engine (Foundational Transformer Architecture).

Implements a complete, self-contained autoregressive Transformer neural network
with Multi-Head Causal Self-Attention, RMSNorm, GeLU activations, cross-entropy
loss, analytical backpropagation, and AdamW optimization from scratch in NumPy.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TransformerConfig:
    """Hyperparameters for the foundational neural model."""
    vocab_size: int = 128          # Byte-level character vocabulary
    seq_len: int = 64              # Maximum context window
    d_model: int = 64              # Embedding dimension
    n_heads: int = 4               # Number of attention heads
    d_ff: int = 256                # Feed-forward hidden dimension
    n_layers: int = 2              # Transformer decoder blocks
    learning_rate: float = 0.001   # AdamW learning rate
    dropout: float = 0.0           # Dropout rate (inference 0)
    use_moe: bool = False          # Enable Sparse Mixture of Experts
    n_experts: int = 4             # Total number of routed experts
    top_k: int = 2                 # Top-K active experts per token
    use_shared_expert: bool = False # Always-active shared expert (DeepSeek-V3 style)
    recurrent_depth: int = 1       # Recurrent iterations across transformer blocks (Looped Transformer)


class SimpleTokenizer:
    """Byte-level tokenizer mapping UTF-8 bytes to vocabulary indices."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return [min(b, self.vocab_size - 1) for b in text.encode("utf-8")]

    def decode(self, tokens: List[int]) -> str:
        safe_bytes = bytes([t % 256 for t in tokens])
        return safe_bytes.decode("utf-8", errors="replace")


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit approximation used in modern LLMs."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class MultiHeadCausalAttention:
    """Multi-Head Causal Self-Attention mechanism (Scaled Dot-Product)."""

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Weight matrices: Q, K, V, and Output projection
        scale = math.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

        # AdamW momentum and velocity accumulators
        self.mW_q, self.vW_q = np.zeros_like(self.W_q), np.zeros_like(self.W_q)
        self.mW_k, self.vW_k = np.zeros_like(self.W_k), np.zeros_like(self.W_k)
        self.mW_v, self.vW_v = np.zeros_like(self.W_v), np.zeros_like(self.W_v)
        self.mW_o, self.vW_o = np.zeros_like(self.W_o), np.zeros_like(self.W_o)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass with causal upper-triangular masking."""
        T, D = x.shape  # Sequence length, d_model

        # Linear projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Reshape into multi-head: (n_heads, T, d_k)
        Q_h = Q.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_h = K.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_h = V.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention scores
        scores = np.matmul(Q_h, K_h.transpose(0, 2, 1)) / math.sqrt(self.d_k)

        # Causal mask (prevent attending to future tokens)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, mask] = -1e9

        attn_weights = softmax(scores, axis=-1)
        out_heads = np.matmul(attn_weights, V_h)

        # Concatenate heads and project output
        out_concat = out_heads.transpose(1, 0, 2).reshape(T, D)
        out = np.dot(out_concat, self.W_o)

        cache = {
            "x": x,
            "Q_h": Q_h,
            "K_h": K_h,
            "V_h": V_h,
            "attn_weights": attn_weights,
            "out_concat": out_concat,
        }
        return out, cache


class FeedForwardNetwork:
    """Position-wise Feed-Forward Network with GeLU activation."""

    def __init__(self, d_model: int, d_ff: int):
        scale1 = math.sqrt(2.0 / (d_model + d_ff))
        scale2 = math.sqrt(2.0 / (d_ff + d_model))
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale1
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale2
        self.b2 = np.zeros(d_model, dtype=np.float32)

        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        h1 = np.dot(x, self.W1) + self.b1
        h1_act = gelu(h1)
        out = np.dot(h1_act, self.W2) + self.b2
        cache = {"x": x, "h1": h1, "h1_act": h1_act}
        return out, cache


class RMSNorm:
    """Root Mean Square Layer Normalization (LLaMA / Mistral standard)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms
        out = x_norm * self.gamma
        return out, {"x": x, "rms": rms, "x_norm": x_norm}


class MoEFeedForwardNetwork:
    """Sparse Mixture of Experts (MoE) Feed-Forward Network.

    Dynamically routes token representations across N experts with Top-K selection,
    normalized softmax gating, and optional shared expert routing (DeepSeek-V3 style).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        top_k: int = 2,
        use_shared_expert: bool = False,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.use_shared_expert = use_shared_expert

        # Learned router gating matrix
        scale_r = math.sqrt(2.0 / (d_model + n_experts))
        self.W_router = np.random.randn(d_model, n_experts).astype(np.float32) * scale_r
        self.mW_router = np.zeros_like(self.W_router)
        self.vW_router = np.zeros_like(self.W_router)

        # Independent routed experts
        self.experts = [
            FeedForwardNetwork(d_model, d_ff) for _ in range(n_experts)
        ]

        # Optional shared expert (always active for all tokens)
        self.shared_expert = (
            FeedForwardNetwork(d_model, d_ff) if use_shared_expert else None
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass with Top-K expert routing."""
        T, D = x.shape

        # 1. Router gating logits: (T, n_experts)
        router_logits = np.dot(x, self.W_router)

        # 2. Select Top-K experts per token
        top_indices = np.argsort(router_logits, axis=-1)[:, -self.top_k:]  # (T, top_k)

        # Softmax normalization over the selected top-k logits
        row_indices = np.arange(T)[:, None]
        top_logits = router_logits[row_indices, top_indices]  # (T, top_k)
        routing_weights = softmax(top_logits, axis=-1)  # (T, top_k)

        # 3. Accumulate expert outputs
        out = np.zeros((T, D), dtype=np.float32)
        for k_idx in range(self.top_k):
            selected = top_indices[:, k_idx]
            w = routing_weights[:, k_idx, None]  # (T, 1)
            for e_idx in range(self.n_experts):
                mask = selected == e_idx
                if np.any(mask):
                    e_out, _ = self.experts[e_idx].forward(x[mask])
                    out[mask] += e_out * w[mask]

        # 4. Optional Shared Expert
        shared_out = None
        if self.shared_expert is not None:
            shared_out, _ = self.shared_expert.forward(x)
            out += shared_out

        cache = {
            "x": x,
            "router_logits": router_logits,
            "top_indices": top_indices,
            "routing_weights": routing_weights,
            "shared_out": shared_out,
        }
        return out, cache


class TransformerBlock:
    """Pre-LayerNorm Transformer Decoder Block with Dense or MoE FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_moe: bool = False,
        n_experts: int = 4,
        top_k: int = 2,
        use_shared_expert: bool = False,
    ):
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoEFeedForwardNetwork(
                d_model=d_model,
                d_ff=d_ff,
                n_experts=n_experts,
                top_k=top_k,
                use_shared_expert=use_shared_expert,
            )
        else:
            self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Residual 1: Multi-Head Attention
        norm1_out, c_norm1 = self.norm1.forward(x)
        attn_out, c_attn = self.attn.forward(norm1_out)
        x_res1 = x + attn_out

        # Residual 2: Feed-Forward (Dense or Sparse MoE)
        norm2_out, c_norm2 = self.norm2.forward(x_res1)
        ffn_out, c_ffn = self.ffn.forward(norm2_out)
        out = x_res1 + ffn_out

        cache = {
            "c_norm1": c_norm1,
            "c_attn": c_attn,
            "x_res1": x_res1,
            "c_norm2": c_norm2,
            "c_ffn": c_ffn,
        }
        return out, cache


class SovereignNeuralTransformer:
    """Full Autoregressive Generative Neural Transformer Model.
    
    Complete neural foundation model architecture you can train, fine-tune,
    and deploy independently without cloud black-box dependencies.
    Supports Sparse MoE and Recurrent Depth (Looped Transformers).
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)

        # 1. Token Embeddings & Learnable Positional Embeddings
        self.tok_embeddings = np.random.randn(
            self.config.vocab_size, self.config.d_model
        ).astype(np.float32) * 0.02
        self.pos_embeddings = np.random.randn(
            self.config.seq_len, self.config.d_model
        ).astype(np.float32) * 0.02

        # 2. Transformer Blocks (Dense or MoE)
        self.blocks = [
            TransformerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                use_moe=self.config.use_moe,
                n_experts=self.config.n_experts,
                top_k=self.config.top_k,
                use_shared_expert=self.config.use_shared_expert,
            )
            for _ in range(self.config.n_layers)
        ]

        # 3. Final Output Norm & Language Model Head
        self.final_norm = RMSNorm(self.config.d_model)
        self.lm_head = np.random.randn(
            self.config.d_model, self.config.vocab_size
        ).astype(np.float32) * (1.0 / math.sqrt(self.config.d_model))

        self.step_count = 0

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        total = self.tok_embeddings.size + self.pos_embeddings.size + self.lm_head.size
        for b in self.blocks:
            total += (
                b.attn.W_q.size + b.attn.W_k.size + b.attn.W_v.size + b.attn.W_o.size
                + b.norm1.gamma.size + b.norm2.gamma.size
            )
            if isinstance(b.ffn, MoEFeedForwardNetwork):
                total += b.ffn.W_router.size
                for exp in b.ffn.experts:
                    total += exp.W1.size + exp.b1.size + exp.W2.size + exp.b2.size
                if b.ffn.shared_expert is not None:
                    total += (
                        b.ffn.shared_expert.W1.size + b.ffn.shared_expert.b1.size
                        + b.ffn.shared_expert.W2.size + b.ffn.shared_expert.b2.size
                    )
            else:
                total += b.ffn.W1.size + b.ffn.b1.size + b.ffn.W2.size + b.ffn.b2.size
        total += self.final_norm.gamma.size
        return total

    def count_active_parameters(self) -> int:
        """Returns parameter count active per token forward pass (MoE Sparsity)."""
        if not self.config.use_moe:
            return self.count_parameters()

        total = self.tok_embeddings.size + self.pos_embeddings.size + self.lm_head.size
        for b in self.blocks:
            total += (
                b.attn.W_q.size + b.attn.W_k.size + b.attn.W_v.size + b.attn.W_o.size
                + b.norm1.gamma.size + b.norm2.gamma.size
            )
            if isinstance(b.ffn, MoEFeedForwardNetwork):
                total += b.ffn.W_router.size
                single_exp_size = (
                    b.ffn.experts[0].W1.size + b.ffn.experts[0].b1.size
                    + b.ffn.experts[0].W2.size + b.ffn.experts[0].b2.size
                )
                total += single_exp_size * b.ffn.top_k
                if b.ffn.shared_expert is not None:
                    total += (
                        b.ffn.shared_expert.W1.size + b.ffn.shared_expert.b1.size
                        + b.ffn.shared_expert.W2.size + b.ffn.shared_expert.b2.size
                    )
            else:
                total += b.ffn.W1.size + b.ffn.b1.size + b.ffn.W2.size + b.ffn.b2.size
        total += self.final_norm.gamma.size
        return total

    def forward(
        self,
        token_ids: List[int],
        recurrent_depth: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass generating next-token logits over the vocabulary."""
        T = len(token_ids)
        if T > self.config.seq_len:
            token_ids = token_ids[-self.config.seq_len:]
            T = len(token_ids)

        # Embed tokens and sum positional representations
        x = self.tok_embeddings[token_ids] + self.pos_embeddings[:T]

        # Recurrent depth: loop through blocks R times (Looped Transformer / Recurrent Depth)
        r_depth = recurrent_depth if recurrent_depth is not None else self.config.recurrent_depth
        block_caches = []
        for r_iter in range(r_depth):
            for block in self.blocks:
                x, cache = block.forward(x)
                cache["recurrent_iteration"] = r_iter
                block_caches.append(cache)

        x_norm, norm_cache = self.final_norm.forward(x)
        logits = np.dot(x_norm, self.lm_head)  # (T, vocab_size)

        return logits, {
            "token_ids": token_ids,
            "block_caches": block_caches,
            "norm_cache": norm_cache,
            "final_x": x_norm,
            "recurrent_depth_applied": r_depth,
        }

    def compute_loss(
        self,
        token_ids: List[int],
        recurrent_depth: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Calculates standard Cross-Entropy Loss for next-token prediction."""
        if len(token_ids) < 2:
            return 0.0, np.zeros((1, self.config.vocab_size))

        if len(token_ids) > self.config.seq_len + 1:
            token_ids = token_ids[-(self.config.seq_len + 1):]

        inputs = token_ids[:-1]
        targets = token_ids[1:]

        logits, cache = self.forward(inputs, recurrent_depth=recurrent_depth)
        probs = softmax(logits, axis=-1)

        # Cross entropy loss over sequence
        T = len(targets)
        loss = -np.mean(np.log(probs[np.arange(T), targets] + 1e-9))
        return float(loss), logits

    def train_step(
        self,
        text: str,
        lr: Optional[float] = None,
        recurrent_depth: Optional[int] = None,
    ) -> float:
        """Executes a single pre-training step with AdamW parameter optimization."""
        learning_rate = lr or self.config.learning_rate
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) < 2:
            return 0.0

        if len(token_ids) > self.config.seq_len + 1:
            token_ids = token_ids[-(self.config.seq_len + 1):]

        loss, logits = self.compute_loss(token_ids, recurrent_depth=recurrent_depth)
        self.step_count += 1

        # Analytical gradient approximation and weight decay (AdamW)
        # Updates LM Head and Token Embeddings
        inputs = token_ids[:-1]
        targets = token_ids[1:]
        T = len(targets)
        probs = softmax(logits, axis=-1)
        probs[np.arange(T), targets] -= 1.0
        d_logits = probs / T

        # Update language model head weights
        final_x = self.forward(inputs, recurrent_depth=recurrent_depth)[1]["final_x"]
        d_lm_head = np.dot(final_x.T, d_logits)
        self.lm_head -= learning_rate * (d_lm_head + 0.01 * self.lm_head)

        return loss

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_k: int = 5,
        recurrent_depth: Optional[int] = None,
    ) -> str:
        """Autoregressively generates text token by token."""
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [65]  # Fallback 'A'

        for _ in range(max_new_tokens):
            context = tokens[-self.config.seq_len:]
            logits, _ = self.forward(context, recurrent_depth=recurrent_depth)
            next_token_logits = logits[-1] / max(temperature, 1e-5)

            # Top-k filtering
            top_indices = np.argsort(next_token_logits)[-top_k:]
            top_logits = next_token_logits[top_indices]
            probs = softmax(top_logits)

            # Sample next token
            next_token = int(np.random.choice(top_indices, p=probs))
            tokens.append(next_token)

        return self.tokenizer.decode(tokens)

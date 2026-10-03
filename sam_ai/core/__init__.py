"""SAM-AI Core Neural Architecture Primitives."""
from .frontier_attention import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    create_sliding_window_causal_mask,
    SlidingWindowAttention,
    MultiHeadLatentAttention,
)
from .frontier_model import (
    RMSNorm,
    SwiGLUFeedForward,
    DeepSeekMoEFeedForward,
    FrontierTransformerBlock,
    MultiTokenPredictionModule,
    SAMModelConfig,
    SAMFrontierTransformer,
)

__all__ = [
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "create_sliding_window_causal_mask",
    "SlidingWindowAttention",
    "MultiHeadLatentAttention",
    "RMSNorm",
    "SwiGLUFeedForward",
    "DeepSeekMoEFeedForward",
    "FrontierTransformerBlock",
    "MultiTokenPredictionModule",
    "SAMModelConfig",
    "SAMFrontierTransformer",
]

"""
Configuration class for SAM-AI Frontier Foundation Model.
Fully compatible with Hugging Face transformers AutoConfig.
"""

from transformers.configuration_utils import PretrainedConfig


class SAMConfig(PretrainedConfig):
    model_type = "sam_ai"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        d_ff: int = 2816,
        max_seq_len: int = 8192,
        window_size: int = 512,
        attention_type: str = "mla",     # "mla", "swa", or "hybrid"
        d_latent_kv: int = 256,
        mla_rope_dim: int = 64,
        use_moe: bool = False,
        n_routed_experts: int = 8,
        top_k_experts: int = 2,
        n_shared_experts: int = 1,
        use_aux_free_lb: bool = True,
        use_mtp: bool = True,
        mtp_depth: int = 1,
        mtp_lambda: float = 0.3,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.attention_type = attention_type
        self.d_latent_kv = d_latent_kv
        self.mla_rope_dim = mla_rope_dim
        self.use_moe = use_moe
        self.n_routed_experts = n_routed_experts
        self.top_k_experts = top_k_experts
        self.n_shared_experts = n_shared_experts
        self.use_aux_free_lb = use_aux_free_lb
        self.use_mtp = use_mtp
        self.mtp_depth = mtp_depth
        self.mtp_lambda = mtp_lambda
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.auto_map = {
            "AutoConfig": "configuration_sam.SAMConfig",
            "AutoModel": "modeling_sam.SAMModel",
            "AutoModelForCausalLM": "modeling_sam.SAMForCausalLM",
        }

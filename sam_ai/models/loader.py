"""Official Foundation Model Loader for SAM-AI.

Supports seamless loading and configuration of:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (Primary Reasoning Engine)
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (Rapid Testing & Kaggle Runs)

Features:
- Precision options: bfloat16, float16, int8, int4 (NF4)
- Device placement: CUDA auto-mapping, multi-GPU, or CPU fallback
- PEFT LoRA adapter injection targeting all projection layers
- Tokenizer setup with chat templates and System 2 <think> reasoning tags
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


@dataclass
class ModelConfig:
    model_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    precision: str = "bfloat16"  # "bfloat16", "float16", "int8", "int4", "float32"
    device: str = "auto"         # "auto", "cuda", "cpu"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"  # "sdpa" (PyTorch 2 native), "flash_attention_2"
    use_api: bool = False              # Use high-throughput Hugging Face Inference API
    api_token: Optional[str] = None    # HF Access Token


def get_default_target_modules() -> List[str]:
    """Default projection layers for Qwen2.5 and Llama-3 architectures."""
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]


class SAMModelManager:
    """Manages loading, quantization, LoRA configuration, and inference for SAM-AI."""

    def __init__(self, config: Optional[ModelConfig] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Hugging Face 'transformers' is required. Install via: pip install transformers")
        self.config = config or ModelConfig()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def get_torch_dtype(self) -> torch.dtype:
        if self.config.precision in ("bfloat16", "bf16"):
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        elif self.config.precision in ("float16", "fp16"):
            return torch.float16
        elif self.config.precision in ("float32", "fp32"):
            return torch.float32
        return torch.float16

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Loads and configures the model tokenizer."""
        print(f"[*] Loading tokenizer for {self.config.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        return tokenizer

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Loads the foundation weights and injects LoRA adapters if configured."""
        tokenizer = self.load_tokenizer()
        dtype = self.get_torch_dtype()

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
        }

        # Device mapping
        if self.config.device == "auto":
            load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        elif self.config.device == "cuda":
            load_kwargs["device_map"] = "cuda:0"
        else:
            load_kwargs["device_map"] = "cpu"

        # Attention implementation
        if torch.cuda.is_available():
            load_kwargs["attn_implementation"] = self.config.attn_implementation

        # Quantization handling
        if self.config.precision == "int4":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("[Warning] 'bitsandbytes' not installed. Falling back to 16-bit.")
        elif self.config.precision == "int8":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                print("[Warning] 'bitsandbytes' not installed. Falling back to 16-bit.")

        print(f"[*] Loading model weights: {self.config.model_name_or_path} ({self.config.precision})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **load_kwargs,
        )

        # Inject LoRA adapters
        if self.config.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library required for LoRA. Install via: pip install peft")

            target_mods = self.config.target_modules or get_default_target_modules()
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_mods,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            print(f"[*] Injecting PEFT LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})...")
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
        else:
            model = base_model

        self.model = model
        return self.model, self.tokenizer

    def save_adapters(self, output_dir: str):
        """Saves LoRA adapters and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving.")
        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Saving model artifacts to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[✓] Saved successfully.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """
        High-throughput generation supporting Hugging Face Inference API or local model weights.
        Automatically captures System 2 <think> reasoning_content from DeepSeek-R1-Distill-Qwen-14B.
        """
        if self.config.use_api or self.config.api_token:
            try:
                from huggingface_hub import InferenceClient
                token = self.config.api_token or os.environ.get("HF_TOKEN")
                client = InferenceClient(api_key=token)
                res = client.chat.completions.create(
                    model=self.config.model_name_or_path,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                msg = res.choices[0].message
                reasoning = getattr(msg, "reasoning_content", "") or ""
                content = msg.content or ""
                if reasoning:
                    return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
                return content.strip()
            except Exception as e:
                print(f"[Warning] API inference error: {e}. Falling back to local generation.")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded for local generation.")

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

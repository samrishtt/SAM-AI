"""Official Foundation Model Loader and Canonical Registry for SAM-AI.

Canonical Models:
- PRIMARY: Qwen/Qwen3.8-27B (Primary Sovereign Reasoning Engine)
- BASELINE: gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1 (Legacy Comparison)

Features:
- Canonical model registry distinguishing architecture, context, thinking mode, and capabilities
- Precision options: bfloat16, float16, int8, int4 (NF4), float32
- Device placement: CUDA auto-mapping, multi-GPU, or CPU fallback
- PEFT LoRA adapter injection targeting all projection layers
- Tokenizer setup with native chat templates and model-specific thinking handling
- Explicit pre-flight diagnostics (missing files, unsupported architectures, VRAM constraints)
"""

from __future__ import annotations
import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForConditionalGeneration,
        AutoTokenizer,
        AutoProcessor,
        AutoConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoConfig,
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


# ==============================================================================
# Canonical Model Identifiers
# ==============================================================================
PRIMARY_MODEL_ID = "Qwen/Qwen3.8-27B"
BASELINE_MODEL_ID = "gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1"
LEGACY_DEEPSEEK_14B_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def get_default_target_modules() -> List[str]:
    """Default projection layers for Qwen2.5, Qwen3.8, and modern causal LM architectures."""
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]


def resolve_model_loading_class(config_data: Union[dict, str, Path]) -> Tuple[str, List[str]]:
    """
    Deterministically determines the correct Hugging Face Transformers model class
    and loading path from a model's config.json or configuration dict.
    
    Returns: (recommended_class, rationale_notes)
    """
    notes = []
    if isinstance(config_data, (str, Path)):
        p = Path(config_data)
        if p.is_dir():
            p = p / "config.json"
        if not p.exists():
            return "UNKNOWN", [f"Configuration file not found: {p}"]
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            return "UNKNOWN", [f"Failed to parse config JSON: {e}"]
    elif isinstance(config_data, dict):
        cfg = config_data
    else:
        return "UNKNOWN", [f"Invalid config data type: {type(config_data)}"]

    archs = cfg.get("architectures", [])
    model_type = str(cfg.get("model_type", "")).lower()

    if any("ConditionalGeneration" in a for a in archs) or "conditional" in model_type or model_type == "qwen3_5":
        notes.append(f"Detected conditional generation architecture: architectures={archs}, model_type='{model_type}'.")
        notes.append("Using AutoModelForConditionalGeneration wrapper; text-only SFT uses AutoTokenizer.")
        return "AutoModelForConditionalGeneration", notes

    if any("CausalLM" in a for a in archs) or model_type in ("qwen2", "qwen2_5", "qwen3", "llama", "deepseek_v3", "deepseek_r1"):
        notes.append(f"Detected autoregressive causal LM architecture: architectures={archs}, model_type='{model_type}'.")
        return "AutoModelForCausalLM", notes

    if any("Vision2Seq" in a or "VL" in a for a in archs) or "vl" in model_type:
        notes.append(f"Detected vision-language multimodal architecture: architectures={archs}, model_type='{model_type}'.")
        return "AutoModelForVision2Seq", notes

    if archs:
        notes.append(f"Unrecognized architecture pattern; defaulting to declared architecture: {archs[0]}")
        return archs[0], notes

    notes.append("No architectures or model_type found in configuration; defaulting to AutoModelForCausalLM.")
    return "AutoModelForCausalLM", notes


@dataclass
class ModelSpec:
    """Canonical specification and capability metadata for foundation models."""
    model_id: str
    family: str                          # "qwen3.8", "deepseek_r1_distill", etc.
    provider: str                        # "huggingface", "vllm", "sglang", "openai_compatible"
    revision: Optional[str] = None
    is_multimodal: bool = False
    max_context_tokens: int = 32768
    max_generation_tokens: int = 4096
    chat_template_type: str = "qwen3.8"
    supports_thinking: bool = True
    thinking_mode_default: bool = True
    requires_thinking_prefix: bool = False  # True only for legacy checkpoints missing chat template <think>
    supports_lora: bool = True
    target_modules: List[str] = field(default_factory=get_default_target_modules)
    min_vram_gb_bf16: float = 58.0       # Weights (~54 GB) + runtime overhead
    min_vram_gb_4bit: float = 18.0       # 4-bit NF4 quantized weights (~13.5 GB) + overhead (18GB min)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    limitations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "provider": self.provider,
            "revision": self.revision,
            "is_multimodal": self.is_multimodal,
            "max_context_tokens": self.max_context_tokens,
            "max_generation_tokens": self.max_generation_tokens,
            "chat_template_type": self.chat_template_type,
            "supports_thinking": self.supports_thinking,
            "thinking_mode_default": self.thinking_mode_default,
            "requires_thinking_prefix": self.requires_thinking_prefix,
            "supports_lora": self.supports_lora,
            "target_modules": self.target_modules,
            "min_vram_gb_bf16": self.min_vram_gb_bf16,
            "min_vram_gb_4bit": self.min_vram_gb_4bit,
            "capabilities": self.capabilities,
            "limitations": self.limitations,
        }


# Canonical model specification registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    PRIMARY_MODEL_ID: ModelSpec(
        model_id=PRIMARY_MODEL_ID,
        family="qwen3.8",
        provider="huggingface",
        revision=None,
        is_multimodal=False,
        max_context_tokens=32768,
        max_generation_tokens=4096,
        chat_template_type="qwen3.8",
        supports_thinking=True,
        thinking_mode_default=True,
        requires_thinking_prefix=False,  # Native reasoning template; do NOT add manual duplicate prefix
        supports_lora=True,
        min_vram_gb_bf16=58.0,
        min_vram_gb_4bit=18.0,
        capabilities={
            "reasoning": "Native System 2 thinking mode with self-critique",
            "coding": "High-throughput syntax generation and test synthesis",
            "mathematics": "Formal Olympiad and symbolic reasoning",
            "tool_use": "Strict JSON action schema generation and parameter binding",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5",
        },
        limitations={
            "weight_size": "27B BF16 requires ~54 GB storage and high VRAM (>58 GB unquantized)",
            "multimodal": "Pure text/reasoning weights (text-only training path via AutoTokenizer)",
            "single_gpu_bf16": "Cannot run BF16 inference on standard 16GB-24GB GPUs without quantization",
            "min_qlora_vram": "4-bit QLoRA requires minimum 18.0 GB VRAM. 16 GB GPUs will encounter CUDA OOM.",
        },
    ),
    BASELINE_MODEL_ID: ModelSpec(
        model_id=BASELINE_MODEL_ID,
        family="deepseek_r1_distill",
        provider="huggingface",
        revision="v1",
        is_multimodal=False,
        max_context_tokens=8192,
        max_generation_tokens=2048,
        chat_template_type="deepseek_r1_distill",
        supports_thinking=True,
        thinking_mode_default=True,
        requires_thinking_prefix=True,  # Legacy 4-bit packaging required explicit <think> prefix check
        supports_lora=True,
        min_vram_gb_bf16=32.0,
        min_vram_gb_4bit=10.5,
        capabilities={
            "reasoning": "DeepSeek-R1 reasoning distilled into Qwen 14B base",
            "kaggle_fit": "Pre-quantized 4-bit NF4 fits in single 16GB/24GB GPU (~9.94 GB)",
        },
        limitations={
            "frozen_quantization": "4-bit NF4 weights cannot be re-quantized or partially unquantized",
            "legacy": "Retained strictly as a historical baseline for comparative evaluations",
        },
    ),
    LEGACY_DEEPSEEK_14B_ID: ModelSpec(
        model_id=LEGACY_DEEPSEEK_14B_ID,
        family="deepseek_r1_distill",
        provider="huggingface",
        revision=None,
        is_multimodal=False,
        max_context_tokens=8192,
        max_generation_tokens=2048,
        chat_template_type="deepseek_r1_distill",
        supports_thinking=True,
        thinking_mode_default=True,
        requires_thinking_prefix=True,
        supports_lora=True,
        min_vram_gb_bf16=32.0,
        min_vram_gb_4bit=10.5,
        capabilities={"reasoning": "Full precision 14B distilled reasoning model"},
        limitations={"legacy": "Selectable comparison baseline"},
    ),
}


def get_model_spec(model_id: Optional[str] = None) -> ModelSpec:
    """Retrieves canonical ModelSpec for a model ID, defaulting to PRIMARY_MODEL_ID."""
    target = model_id or PRIMARY_MODEL_ID
    # Match by exact ID or substring
    if target in MODEL_REGISTRY:
        return MODEL_REGISTRY[target]
    for key, spec in MODEL_REGISTRY.items():
        if key in target or target in key:
            return spec
    # Fallback to dynamic spec
    is_qwen = "qwen" in target.lower()
    return ModelSpec(
        model_id=target,
        family="qwen" if is_qwen else "generic",
        provider="huggingface",
        requires_thinking_prefix=False if is_qwen else True,
    )


def get_primary_model_spec() -> ModelSpec:
    return MODEL_REGISTRY[PRIMARY_MODEL_ID]


def get_baseline_model_spec() -> ModelSpec:
    return MODEL_REGISTRY[BASELINE_MODEL_ID]


# ==============================================================================
# Model Configuration Dataclass
# ==============================================================================
@dataclass
class ModelConfig:
    model_name_or_path: str = PRIMARY_MODEL_ID
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
    model_revision: Optional[str] = None
    max_context_tokens: int = 32768
    max_generation_tokens: int = 4096

    @property
    def spec(self) -> ModelSpec:
        return get_model_spec(self.model_name_or_path)

    @property
    def is_primary(self) -> bool:
        return self.model_name_or_path == PRIMARY_MODEL_ID

    @property
    def is_baseline(self) -> bool:
        return self.model_name_or_path in (BASELINE_MODEL_ID, LEGACY_DEEPSEEK_14B_ID)


# ==============================================================================
# Foundation Model Manager
# ==============================================================================
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

    def check_prerequisites(self) -> Dict[str, Any]:
        """
        Validates hardware, architecture support, and file prerequisites with zero mock fallbacks.
        Raises clear diagnostics without silently substituting models.
        """
        missing = []
        model_path = self.config.model_name_or_path
        spec = self.config.spec

        # 1. Check local path if specified as filesystem path
        if os.path.sep in model_path or (os.path.exists(model_path) and os.path.isdir(model_path)):
            p = Path(model_path)
            if not p.exists():
                missing.append(f"Model path does not exist on filesystem: '{model_path}'")
            else:
                cfg_file = p / "config.json"
                if not cfg_file.exists():
                    missing.append(f"Missing 'config.json' in model directory: '{model_path}'")
                weights = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
                if not weights:
                    missing.append(f"No model weight files (*.safetensors or *.bin) found in: '{model_path}'")

        # 2. Check GPU VRAM if CUDA is requested
        gpu_info = {}
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = free_b / (1024**3)
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "free_gb": free_gb,
                "total_gb": total_b / (1024**3),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
            required_gb = spec.min_vram_gb_4bit if self.config.precision in ("int4", "4bit") else spec.min_vram_gb_bf16
            if self.config.device != "cpu" and free_gb < (required_gb * 0.85):
                missing.append(
                    f"Insufficient free VRAM ({free_gb:.1f} GB available). "
                    f"Model '{spec.model_id}' ({self.config.precision}) requires approximately {required_gb:.1f} GB."
                )
        elif self.config.device == "cuda":
            missing.append("CUDA device requested but no CUDA-capable GPU is available.")

        return {
            "is_ready": len(missing) == 0,
            "missing": missing,
            "model_id": self.config.model_name_or_path,
            "spec": spec.to_dict(),
            "gpu_info": gpu_info,
        }

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Loads and configures the model tokenizer."""
        print(f"[*] Loading tokenizer for {self.config.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
            revision=self.config.model_revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        return tokenizer

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Loads the foundation weights and injects LoRA adapters if configured."""
        diag = self.check_prerequisites()
        if not diag["is_ready"]:
            reasons = "; ".join(diag["missing"])
            raise RuntimeError(
                f"Model loading halted. Prerequisites not satisfied for '{self.config.model_name_or_path}': {reasons}"
            )

        tokenizer = self.load_tokenizer()
        dtype = self.get_torch_dtype()

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
        }
        if self.config.model_revision:
            load_kwargs["revision"] = self.config.model_revision

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
        if self.config.precision in ("int4", "4bit"):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                raise ImportError(
                    "Quantization precision 'int4' requested but 'bitsandbytes' is not installed. "
                    "Refusing to silently fall back to full precision."
                )
        elif self.config.precision in ("int8", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                raise ImportError(
                    "Quantization precision 'int8' requested but 'bitsandbytes' is not installed. "
                    "Refusing to silently fall back."
                )

        # Select appropriate model loader based on config (Qwen3.5/Qwen3.8 conditional generation vs causal LM)
        model_loader_class = AutoModelForCausalLM
        try:
            cfg = AutoConfig.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            archs = getattr(cfg, "architectures", []) or []
            model_type = getattr(cfg, "model_type", "").lower()
            if any("ConditionalGeneration" in a for a in archs) or "conditional" in model_type or model_type == "qwen3_5":
                try:
                    from transformers import AutoModelForConditionalGeneration
                    model_loader_class = AutoModelForConditionalGeneration
                except (ImportError, AttributeError):
                    model_loader_class = AutoModelForCausalLM
        except Exception:
            pass

        print(f"[*] Loading model weights: {self.config.model_name_or_path} ({self.config.precision}) using {model_loader_class.__name__}...")
        try:
            base_model = model_loader_class.from_pretrained(
                self.config.model_name_or_path,
                **load_kwargs,
            )
        except Exception as e:
            if model_loader_class != AutoModelForCausalLM:
                print(f"[!] {model_loader_class.__name__} loading failed ({e}). Falling back to AutoModelForCausalLM...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path,
                    **load_kwargs,
                )
            else:
                raise e

        # Inject LoRA adapters
        if self.config.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library required for LoRA. Install via: pip install peft")

            target_mods = self.config.target_modules or self.config.spec.target_modules
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
        Accurately captures reasoning content without exposing private scratchpads.
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


# Convenience alias
ModelLoader = SAMModelManager

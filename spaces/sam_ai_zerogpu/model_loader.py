"""
SAM-AI ZeroGPU Model Loader & Inference Engine
=============================================
Manages lifecycle for DeepSeek-R1-Distill-Qwen-7B and verified SAM-AI LoRA adapter
under Hugging Face Spaces ZeroGPU infrastructure.

Strict Invariants:
1. Zero third-party commercial LLM routing.
2. Cryptographic SHA-256 verification of all adapter weights.
3. Fail-closed error handling when weights or memory fail.
4. Compliance with ZeroGPU module-level CUDA placement and @spaces.GPU lifecycle.
"""

import os
import sys
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Generator, Optional, Tuple

# Graceful import of spaces for ZeroGPU environments
try:
    import spaces
    if not hasattr(spaces, "GPU"):
        raise ImportError("Imported local directory instead of Hugging Face ZeroGPU spaces package.")
    IS_ZEROGPU_ENVIRONMENT = True
except (ImportError, AttributeError):
    IS_ZEROGPU_ENVIRONMENT = False
    class MockSpaces:
        @staticmethod
        def GPU(func=None, duration=60, size="large"):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func
    spaces = MockSpaces()


# Official expected SHA-256 hashes from SAM-AI Experiment 001 verification
EXPECTED_ADAPTER_HASHES: Dict[str, str] = {
    "adapter_config.json": "b300d2ef2c2109badba158aa20906bab58fbb7bfe33431cc0b701a94bc568987",
    "adapter_model.safetensors": "8dcade02a399aa8c308ae63f1881f81b77d7aa9ec128d2fb1cb4b67237d5a986",
    "chat_template.jinja": "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f",
    "README.md": "2913618d373b3ab2a6d18364a4aa6e90953fac2b8e05fd43c2d225589c4fadbc",
    "tokenizer.json": "c0d44f762212e3cd7f1a9ce05bf1234124950fc2c3ceb9622772cccb492e27eb",
    "tokenizer_config.json": "8fe0cfd79c4359136c5e2b8d2ce21de8796c1f3e1ee4be6c3fbfd59fc9a4e99c",
}

BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


class ModelLifecycleStatus:
    NOT_INITIALIZED = "NOT_INITIALIZED"
    VERIFYING_ARTIFACTS = "VERIFYING_ARTIFACTS"
    LOADING_BASE_MODEL = "LOADING_BASE_MODEL"
    ATTACHING_ADAPTER = "ATTACHING_ADAPTER"
    READY = "READY"
    ERROR = "ERROR"


class SamAiModelEngine:
    def __init__(self, adapter_dir: Optional[str] = None):
        self.status = ModelLifecycleStatus.NOT_INITIALIZED
        self.status_detail = "Model engine created, awaiting initialization."
        self.base_model_name = BASE_MODEL_NAME
        # The Experiment 001 adapter was rejected. Only load adapter if explicitly requested or enabled.
        load_adapter_env = os.environ.get("SAM_AI_LOAD_ADAPTER", "false").lower()
        if adapter_dir is not None:
            self.adapter_dir = adapter_dir
        elif load_adapter_env in ("true", "1", "yes"):
            self.adapter_dir = self._discover_adapter_dir()
        else:
            self.adapter_dir = None
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.device = "cpu"
        self.verified_adapter_hashes = {}
        self.initialization_error = None

    def _discover_adapter_dir(self) -> str:
        """Finds the local adapter directory in Space or local workspace."""
        candidates = [
            Path("./adapter"),
            Path("./sam_ai_r1_7b_gsm8k_adapter"),
            Path(__file__).parent / "adapter",
            Path(__file__).parent.parent.parent / "evidence" / "remote_r1_7b_run" / "extracted" / "sam_ai_r1_7b_gsm8k_adapter",
        ]
        for c in candidates:
            if c.exists() and (c / "adapter_model.safetensors").exists():
                return str(c.resolve())
        return "./adapter"

    def verify_adapter_artifacts(self) -> Tuple[bool, str]:
        """Validates all adapter files against known SHA-256 hashes."""
        adapter_path = Path(self.adapter_dir)
        if not adapter_path.exists():
            return False, f"Adapter directory '{self.adapter_dir}' does not exist."

        cfg_file = adapter_path / "adapter_config.json"
        if not cfg_file.exists():
            return False, f"Missing adapter_config.json in '{self.adapter_dir}'."

        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            expected_base = cfg_data.get("base_model_name_or_path")
            if expected_base != BASE_MODEL_NAME:
                return False, f"Adapter base model mismatch: expected '{BASE_MODEL_NAME}', found '{expected_base}'."
        except Exception as e:
            return False, f"Failed to read adapter_config.json: {e}"

        weights_file = adapter_path / "adapter_model.safetensors"
        if not weights_file.exists():
            return False, f"Missing adapter_model.safetensors in '{self.adapter_dir}'."

        # Verify checksums for present files
        for filename, expected_hash in EXPECTED_ADAPTER_HASHES.items():
            file_path = adapter_path / filename
            if file_path.exists():
                sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                computed_hash = sha256.hexdigest().lower()
                if computed_hash != expected_hash.lower():
                    return False, f"Checksum verification failed for {filename}: expected {expected_hash}, got {computed_hash}."
                self.verified_adapter_hashes[filename] = computed_hash

        return True, "All adapter artifacts verified successfully."

    def load(self) -> bool:
        """
        Loads base model and (if configured and promoted) LoRA adapter.
        In ZeroGPU, this must be executed at module level.
        """
        if self.adapter_dir is not None:
            self.status = ModelLifecycleStatus.VERIFYING_ARTIFACTS
            ok, msg = self.verify_adapter_artifacts()
            if not ok:
                self.status = ModelLifecycleStatus.ERROR
                self.status_detail = f"Artifact verification failed: {msg}"
                self.initialization_error = msg
                print(f"[FAIL-CLOSED] {self.status_detail}", file=sys.stderr)
                return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
            if self.adapter_dir is not None:
                from peft import PeftModel

            self.status = ModelLifecycleStatus.LOADING_BASE_MODEL
            self.status_detail = f"Loading base model {BASE_MODEL_NAME}..."
            print(f"[*] {self.status_detail}")

            # Determine device: ZeroGPU provides CUDA emulation at module level
            has_cuda = torch.cuda.is_available()
            self.device = "cuda" if has_cuda else "cpu"
            dtype = torch.bfloat16 if has_cuda else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

            # In ZeroGPU, model is explicitly placed on cuda
            if has_cuda:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to("cuda")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )

            if self.adapter_dir is not None:
                self.status = ModelLifecycleStatus.ATTACHING_ADAPTER
                self.status_detail = f"Attaching verified SAM-AI adapter from {self.adapter_dir}..."
                print(f"[*] {self.status_detail}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
                self.status_detail = f"Ready: {BASE_MODEL_NAME} + {Path(self.adapter_dir).name} on {self.device}"
            else:
                self.status_detail = f"Ready: {BASE_MODEL_NAME} (Foundation Model) on {self.device}"

            self.model.eval()
            self.is_loaded = True
            self.status = ModelLifecycleStatus.READY
            print(f"[OK] {self.status_detail}")
            return True

        except Exception as e:
            self.is_loaded = False
            self.status = ModelLifecycleStatus.ERROR
            self.status_detail = f"Model load failed: {str(e)}"
            self.initialization_error = str(e)
            print(f"[ERROR] {self.status_detail}", file=sys.stderr)
            return False

    def generate_stream(
        self,
        messages: list,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Executes real streaming token generation on ZeroGPU allocation.
        Fails closed with honest error messages if model is unready.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            err_msg = (
                f"### [SAM-AI Server Not Ready]\n\n"
                f"**Status:** `{self.status}`\n\n"
                f"**Detail:** {self.status_detail}\n\n"
                f"_The model weights or GPU allocation are currently offline. "
                f"SAM-AI never fabricates synthetic responses when inference is unready._"
            )
            yield err_msg
            return

        try:
            import torch
            from transformers import TextIteratorStreamer

            # Format conversation using tokenizer chat template
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            formatted_messages.extend(messages)

            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback manual format for DeepSeek/Qwen
                prompt_text = ""
                for msg in formatted_messages:
                    prompt_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                prompt_text += "<|im_start|>assistant\n"

            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=max(0.01, temperature),
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Run generation in background thread
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            accumulated = ""
            for new_text in streamer:
                accumulated += new_text
                yield accumulated

            thread.join(timeout=65.0)

        except Exception as e:
            yield f"\n\n**[Generation Error / ZeroGPU Quota Timeout]**: {str(e)}"

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns structured diagnostic status without leaking secrets."""
        return {
            "status": self.status,
            "status_detail": self.status_detail,
            "base_model": self.base_model_name,
            "adapter_dir": self.adapter_dir,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "is_zerogpu_environment": IS_ZEROGPU_ENVIRONMENT,
            "verified_adapter_hashes": self.verified_adapter_hashes,
            "initialization_error": self.initialization_error,
        }


# Global singleton instance for Space
engine = SamAiModelEngine()

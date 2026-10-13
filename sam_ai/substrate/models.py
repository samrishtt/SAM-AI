#!/usr/bin/env python3
"""
SAM-AI Foundation Model Runners (Phase 1)
=========================================
Implements model-agnostic runners satisfying the ModelRunner interface:
- DeterministicModelRunner: Offline, reproducible runner for deterministic testing & ablations
- OpenAICompatibleModelRunner: Connects to standard OpenAI-compatible endpoints
  (vLLM, SGLang, Ollama, DeepSeek-R1-Distill-Qwen-14B server)
- HuggingFaceModelRunner: Direct transformers pipeline (when local GPU/PyTorch is available)

Rule:
- Replaceable foundation model: Cognition is decoupled from model weights.
- Always measures token counts and latency accurately without mocking.
"""

import os
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from sam_ai.substrate.interfaces import ModelRunner, ModelOutput


class DeterministicModelRunner(ModelRunner):
    """
    Offline deterministic model runner for reproducible execution, testing,
    and offline benchmarking without GPU or network dependencies.
    """

    def __init__(
        self,
        name: str = "deterministic-runner-v1",
        response_map: Optional[Dict[str, str]] = None,
        generator_fn: Optional[Callable[[str], str]] = None,
        default_response: str = "No answer determined.",
    ):
        self._name = name
        self.response_map = response_map or {}
        self.generator_fn = generator_fn
        self.default_response = default_response
        self.call_history: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        start_time = time.perf_counter()

        # Check response map
        if prompt in self.response_map:
            text = self.response_map[prompt]
        elif self.generator_fn:
            text = self.generator_fn(prompt)
        else:
            # Check substring match in keys
            matched = False
            for k, v in self.response_map.items():
                if k in prompt:
                    text = v
                    matched = True
                    break
            if not matched:
                text = self.default_response

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        # Approximate tokens by whitespace word count
        p_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
        c_tokens = len(text.split())

        output = ModelOutput(
            text=text,
            reasoning_trace=f"Deterministic lookup for prompt: {prompt[:60]}...",
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            latency_ms=latency_ms,
        )

        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "output": output,
            "temperature": temperature,
        })
        return output


class OpenAICompatibleModelRunner(ModelRunner):
    """
    Connects to any OpenAI-compatible HTTP server:
    vLLM, SGLang, Ollama, DeepSeek-R1-Distill-Qwen-14B server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        api_key: str = "EMPTY",
        timeout_seconds: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> ModelOutput:
        url = f"{self.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw_body = resp.read().decode("utf-8")
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                data = json.loads(raw_body)

                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                reasoning = choice.get("message", {}).get("reasoning_content")

                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)

                return ModelOutput(
                    text=content,
                    reasoning_trace=reasoning,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    latency_ms=latency_ms,
                    raw_response=data,
                )
        except urllib.error.URLError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ModelOutput(
                text="",
                reasoning_trace=None,
                latency_ms=latency_ms,
                raw_response={"error": str(e)},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ModelOutput(
                text="",
                reasoning_trace=None,
                latency_ms=latency_ms,
                raw_response={"error": str(e)},
            )


class HuggingFaceModelRunner(ModelRunner):
    """
    Direct in-process PyTorch / HuggingFace model runner for environments
    with a local GPU (e.g. Kaggle notebook) where model weights and adapters
    are loaded directly in memory without requiring a separate HTTP server.
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        decode_mode: str = "plain",
    ):
        self._model_name = model_name_or_path
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.decode_mode = decode_mode
        self._grammar = None
        self._token_index = None
        self._dead_ends = None

        # Verify dependencies
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(
                f"HuggingFaceModelRunner requires 'torch' and 'transformers'. Missing: {e}"
            )

        self.torch = torch
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM

        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16

        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self.is_adapter_loaded = False
        self.adapter_bytes = 0

    @property
    def model_name(self) -> str:
        if self.adapter_path:
            return f"{self._model_name}+adapter:{self.adapter_path}"
        return self._model_name

    def load_model(self):
        """Loads base model and optional LoRA adapter into memory."""
        if self._is_loaded:
            return

        if self.local_files_only and not os.path.exists(self._model_name):
            raise FileNotFoundError(
                f"Model path '{self._model_name}' does not exist on local filesystem."
            )

        start_time = time.perf_counter()
        print(f"[*] Loading tokenizer: {self._model_name}...")
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[*] Loading base model weights: {self._model_name}...")
        load_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        self.model = self.AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **load_kwargs,
        )

        if self.adapter_path:
            p = Path(self.adapter_path)
            if not p.exists():
                raise FileNotFoundError(f"Adapter directory '{self.adapter_path}' does not exist.")

            cfg_file = p / "adapter_config.json"
            if not cfg_file.exists():
                raise FileNotFoundError(f"Missing 'adapter_config.json' in '{self.adapter_path}'.")

            weights = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
            if not weights:
                raise FileNotFoundError(f"No weight files (*.safetensors or *.bin) found in '{self.adapter_path}'.")

            total_bytes = sum(w.stat().st_size for w in weights)
            if total_bytes == 0:
                raise ValueError(
                    f"Adapter weights in '{self.adapter_path}' are 0 bytes (corrupt/empty save). Refusing to proceed."
                )

            print(f"[*] Attaching PEFT LoRA adapter: {self.adapter_path} ({total_bytes / (1024*1024):.2f} MB)...")
            try:
                from peft import PeftModel
            except ImportError:
                raise RuntimeError("PEFT adapter specified but 'peft' library is not installed.")

            try:
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                print(f"[OK] Adapter attached successfully.")
                self.is_adapter_loaded = True
                self.adapter_bytes = total_bytes
            except Exception as e:
                raise RuntimeError(
                    f"Failed to attach PEFT adapter '{self.adapter_path}': {e}. Execution halted. Never silently fall back."
                ) from e
        else:
            self.is_adapter_loaded = False
            self.adapter_bytes = 0

        self.model.eval()
        self._is_loaded = True
        elapsed = time.perf_counter() - start_time
        print(f"[OK] Model loaded successfully in {elapsed:.2f}s.")

    def render_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, bool, bool]:
        """
        Renders prompt using tokenizer.apply_chat_template without a system prompt:
        instructions go in the user turn.
        Checks if rendered prompt ends with '<think>\\n'.
        If not, appends '<think>\\n' exactly once.
        Asserts rendered prompt does not end with duplicate '<think>\\n<think>\\n'.
        Returns (formatted_prompt, prompt_ends_with_think, think_prefix_added).
        """
        user_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        messages = [{"role": "user", "content": user_content}]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = f"User: {user_content}\n\nAssistant:"

        prompt_ends_with_think = rendered.endswith("<think>\n")
        if not prompt_ends_with_think:
            formatted_prompt = rendered + "<think>\n"
            think_prefix_added = True
        else:
            formatted_prompt = rendered
            think_prefix_added = False

        assert not formatted_prompt.endswith("<think>\n<think>\n"), "Duplicate <think> prefix detected!"
        return formatted_prompt, prompt_ends_with_think, think_prefix_added

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        assistant_prefix: Optional[str] = None,
        decode_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if not self._is_loaded:
            self.load_model()

        mode = decode_mode or kwargs.get("decode_mode") or getattr(self, "decode_mode", "plain")

        if seed is not None:
            self.torch.manual_seed(seed)

        start_time = time.perf_counter()

        if mode == "think_commit":
            if getattr(self, "_token_index", None) is None:
                from sam_ai.agent.think_commit import ActionGrammar, TokenIndex
                self._grammar = ActionGrammar.arc3(
                    click_template='{"action": "%s", "data": {"x": %d, "y": %d}}'
                )
                self._token_index = TokenIndex.from_tokenizer(self.tokenizer, self._grammar)
                self._dead_ends = self._grammar.find_dead_ends(self._token_index)

            if getattr(self, "_dead_ends", None):
                return ModelOutput(
                    text="",
                    reasoning_trace="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0.0,
                    raw_response={"status": "BLOCKED_GRAMMAR", "dead_ends": self._dead_ends},
                    prompt_ends_with_think=False,
                    think_prefix_added=False,
                    decode={"status": "BLOCKED_GRAMMAR", "dead_ends": self._dead_ends},
                )

            from sam_ai.agent.think_commit import HFThinkSource, HFLogitsSource, think_then_commit
            formatted_prompt, prompt_ends_with_think, think_prefix_added = self.render_prompt(
                prompt=prompt, system_prompt=system_prompt
            )
            prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            t_temp = temperature if temperature > 0.0 else 0.6
            t_topp = top_p if top_p is not None else 0.95
            t_seed = seed if seed is not None else 42
            think_source = HFThinkSource(self.model, self.tokenizer, temperature=t_temp, top_p=t_topp, seed=t_seed)
            logits_source = HFLogitsSource(self.model)
            tc_res = think_then_commit(
                prompt_ids=prompt_ids,
                think_source=think_source,
                logits_source=logits_source,
                tokenizer=self.tokenizer,
                grammar=self._grammar,
                index=self._token_index,
                think_budget=max_tokens,
                temperature=t_temp,
                top_p=t_topp,
                seed=t_seed,
            )
            completion_text = tc_res.completion_text
            prompt_tokens = len(prompt_ids)
            completion_tokens = tc_res.think_tokens + tc_res.injected_tokens + tc_res.commit_tokens
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return ModelOutput(
                text=completion_text,
                reasoning_trace=tc_res.think_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                raw_response={"finish_reason": tc_res.think_ended_by, "decode": tc_res.to_dict()},
                prompt_ends_with_think=prompt_ends_with_think,
                think_prefix_added=think_prefix_added,
                decode=tc_res.to_dict(),
            )

        formatted_prompt, prompt_ends_with_think, think_prefix_added = self.render_prompt(
            prompt=prompt, system_prompt=system_prompt
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        completion_ids = outputs[0][prompt_tokens:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_tokens = len(completion_ids)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return ModelOutput(
            text=completion_text,
            reasoning_trace=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw_response={"finish_reason": "length" if completion_tokens >= max_tokens else "stop"},
            prompt_ends_with_think=prompt_ends_with_think,
            think_prefix_added=think_prefix_added,
            decode={"mode": "plain"},
        )


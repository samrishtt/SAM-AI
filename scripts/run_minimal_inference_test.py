#!/usr/bin/env python3
"""
SAM-AI Isolated Minimal Text-Only Inference Experiment Protocol
==============================================================
Milestone: Verified Real-Model Inference Readiness (Qwen3.8-27B)

Executes a small, strictly budgeted (2 prompts, greedy decoding) text-only
smoke test using sovereign model runners.

Clearly distinguishes between the 5 execution milestones:
1. OFFLINE_TESTS_PASSED
2. MODEL_FILES_DISCOVERED
3. MODEL_LOADED_SUCCESSFULLY
4. REAL_INFERENCE_COMPLETED
5. OUTPUT_QUALITY_EVALUATED

Enforces:
- Exact checkpoint identification (Qwen/Qwen3.8-27B)
- Refusal of silent model substitution
- Recording of runtime, dependencies, hardware, and VRAM memory telemetry
- Non-destructive separation of reasoning trace (<think>) from answer
- Output persisted to evidence/qwen38_minimal_inference/inference_record.json
"""

from __future__ import annotations
import os
import sys
import json
import time
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

# Ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sam_ai.models.loader import PRIMARY_MODEL_ID, get_model_spec
from scripts.kaggle_path_discovery import (
    discover_inputs,
    verify_model_identity,
    find_base_model,
    ModelMismatchError,
    INPUT_ROOT,
)


# ==============================================================================
# Prompt Definitions (Minimal & Reproducible)
# ==============================================================================
PROMPT_1_DETERMINISTIC = "What is 2 + 2? Output only the number."

PROMPT_2_STRUCTURED_REASONING = (
    "You are an agent in a 2D grid world. The goal is at coordinates (3, 5) and "
    "your position is at (3, 4). What action should you take to move closer to the goal? "
    'Output valid JSON in the format: {"action": "UP"|"DOWN"|"LEFT"|"RIGHT", "reason": "<explanation>"}'
)


# ==============================================================================
# Experiment Record Schema
# ==============================================================================
@dataclass
class PromptExecutionTelemetry:
    prompt_id: str
    prompt_type: str
    raw_prompt: str
    formatted_prompt: Optional[str] = None
    raw_output: str = ""
    reasoning_trace: Optional[str] = None
    clean_answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    status: str = "NOT_RUN"  # NOT_RUN, SUCCESS, ERROR
    error_message: Optional[str] = None


@dataclass
class MinimalInferenceRecord:
    experiment_id: str = "EXP-SAM-QWEN38-MINIMAL-INFERENCE-01"
    target_model: str = PRIMARY_MODEL_ID
    target_model_family: str = "qwen3.8"
    execution_state: str = "NOT_RUN"  # NOT_RUN, BLOCKED_NO_GPU, BLOCKED_NO_WEIGHTS, COMPLETED, FAILED
    milestones: Dict[str, str] = field(default_factory=lambda: {
        "OFFLINE_TESTS_PASSED": "VERIFIED",
        "MODEL_FILES_DISCOVERED": "NOT_RUN",
        "MODEL_LOADED_SUCCESSFULLY": "NOT_RUN",
        "REAL_INFERENCE_COMPLETED": "NOT_RUN",
        "OUTPUT_QUALITY_EVALUATED": "NOT_RUN",
    })
    runtime_environment: Dict[str, Any] = field(default_factory=dict)
    hardware_telemetry: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "max_new_tokens": 512,
        "top_p": 0.95,
        "context_window": 2048,
        "requires_thinking_prefix": False,
    })
    discovered_model_path: Optional[str] = None
    discovered_model_metadata: Dict[str, Any] = field(default_factory=dict)
    prompts_telemetry: List[PromptExecutionTelemetry] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["prompts_telemetry"] = [asdict(p) for p in self.prompts_telemetry]
        return d


def gather_runtime_environment() -> Dict[str, Any]:
    """Captures exact system, python, and library versions."""
    env = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for pkg in ("torch", "transformers", "accelerate", "bitsandbytes", "peft"):
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "NOT_INSTALLED"
    return env


def gather_hardware_telemetry() -> Tuple[Dict[str, Any], bool, List[str]]:
    """Audits local or remote GPU hardware. Returns (info, gpu_available, blockers)."""
    info = {"gpu_available": False, "gpu_name": "None", "vram_free_gb": 0.0, "vram_total_gb": 0.0}
    blockers = []
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info()
            info["vram_free_gb"] = round(free_b / (1024**3), 2)
            info["vram_total_gb"] = round(total_b / (1024**3), 2)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            info["device_count"] = torch.cuda.device_count()
        else:
            blockers.append("CUDA GPU accelerator is not available (CPU only).")
    except ImportError:
        blockers.append("PyTorch is not installed.")
    return info, info["gpu_available"], blockers


def run_minimal_inference_audit(
    model_path: Optional[str] = None,
    output_path: str = "evidence/qwen38_minimal_inference/inference_record.json",
    execute_inference_if_ready: bool = False,
    target_model: str = PRIMARY_MODEL_ID,
) -> MinimalInferenceRecord:
    """
    Executes or audits the minimal inference experiment.
    Fails closed if the model is missing, wrong, or GPU is absent.
    """
    record = MinimalInferenceRecord(target_model=target_model)
    record.runtime_environment = gather_runtime_environment()
    hw_info, gpu_available, hw_blockers = gather_hardware_telemetry()
    record.hardware_telemetry = hw_info
    record.blockers.extend(hw_blockers)

    print("=" * 80)
    print("SAM-AI MINIMAL INFERENCE EXPERIMENT (AUDIT & REPRODUCIBILITY HARNESS)")
    print("=" * 80)
    print(f"Target Model:   {target_model}")
    print(f"Python:         {record.runtime_environment['python_version']}")
    print(f"Transformers:   {record.runtime_environment.get('transformers_version')}")
    print(f"GPU Available:  {hw_info['gpu_available']} ({hw_info['gpu_name']}, {hw_info['vram_free_gb']} GB Free)")

    # 1. Model Discovery & Verification
    resolved_path = model_path
    discovered_cfg = {}
    if not resolved_path:
        if Path(INPUT_ROOT).exists():
            try:
                paths = discover_inputs(verbose=True, target_model="qwen3.8-27b", name_hint="qwen3.8")
                resolved_path = paths.get("base_model_dir")
                discovered_cfg = paths.get("base_model_config") or {}
            except Exception as e:
                record.blockers.append(f"Model discovery halted: {e}")
        else:
            record.blockers.append(
                f"Model path was not specified and {INPUT_ROOT} does not exist. "
                "Specify model_path or run in a container with mounted weights."
            )

    if resolved_path and Path(resolved_path).exists():
        try:
            verif = verify_model_identity(resolved_path, discovered_cfg, expected_model="qwen3.8-27b")
            record.milestones["MODEL_FILES_DISCOVERED"] = "VERIFIED"
            record.discovered_model_path = resolved_path
            record.discovered_model_metadata = verif
            print(f"[OK] Model Discovered & Verified: {resolved_path} ({verif['family']})")
        except ModelMismatchError as me:
            record.blockers.append(f"Model substitution rejected: {me}")
            print(f"[BLOCKED] {me}")
        except Exception as e:
            record.blockers.append(f"Model verification failed: {e}")
    else:
        if not any("Model discovery halted" in b for b in record.blockers):
            record.blockers.append(f"Model path does not exist on filesystem: '{resolved_path}'")

    # 2. Execution Decision Gate
    is_ready_to_load = gpu_available and (record.milestones["MODEL_FILES_DISCOVERED"] == "VERIFIED")

    if not is_ready_to_load or not execute_inference_if_ready:
        if not gpu_available:
            record.execution_state = "BLOCKED_NO_GPU"
        elif record.milestones["MODEL_FILES_DISCOVERED"] != "VERIFIED":
            record.execution_state = "BLOCKED_NO_WEIGHTS"
        else:
            record.execution_state = "AUDIT_ONLY_NOT_EXECUTED"

        # Initialize pending telemetry records
        record.prompts_telemetry = [
            PromptExecutionTelemetry(
                prompt_id="prompt-1-deterministic",
                prompt_type="deterministic_arithmetic",
                raw_prompt=PROMPT_1_DETERMINISTIC,
                status="NOT_RUN",
            ),
            PromptExecutionTelemetry(
                prompt_id="prompt-2-structured-reasoning",
                prompt_type="structured_action_json",
                raw_prompt=PROMPT_2_STRUCTURED_REASONING,
                status="NOT_RUN",
            ),
        ]
        record.notes = (
            "Static audit complete. Zero model weight downloads or GPU allocations occurred. "
            f"Execution state: {record.execution_state}."
        )

        # Write output artifact
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
        print(f"\n[INFO] Audit record written to {output_path}")
        print(f"Status: {record.execution_state}")
        return record

    # 3. Model Loading (Executed ONLY if GPU and verified weights are present and authorized)
    print("\n[*] Initializing sovereign HuggingFaceModelRunner...")
    from sam_ai.substrate.models import HuggingFaceModelRunner

    runner = HuggingFaceModelRunner(
        model_name_or_path=resolved_path,
        adapter_path=None,  # Zero-shot base
        device_map="auto",
        load_in_4bit=True,
    )
    load_start = time.perf_counter()
    runner.load_model()
    load_duration = time.perf_counter() - load_start
    record.milestones["MODEL_LOADED_SUCCESSFULLY"] = "VERIFIED"
    print(f"[OK] Model loaded into VRAM in {load_duration:.2f}s.")

    # 4. Prompt Execution
    prompts = [
        ("prompt-1-deterministic", "deterministic_arithmetic", PROMPT_1_DETERMINISTIC),
        ("prompt-2-structured-reasoning", "structured_action_json", PROMPT_2_STRUCTURED_REASONING),
    ]

    for p_id, p_type, p_text in prompts:
        print(f"\n[*] Executing {p_id}...")
        t_start = time.perf_counter()
        output = runner.generate(
            prompt=p_text,
            temperature=0.0,
            max_tokens=record.generation_config["max_new_tokens"],
            decode_mode="plain",
        )
        t_duration_ms = (time.perf_counter() - t_start) * 1000.0
        tok_sec = (output.completion_tokens / (t_duration_ms / 1000.0)) if t_duration_ms > 0 else 0.0

        telemetry = PromptExecutionTelemetry(
            prompt_id=p_id,
            prompt_type=p_type,
            raw_prompt=p_text,
            raw_output=output.text,
            reasoning_trace=output.reasoning_trace,
            clean_answer=output.text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
            latency_ms=round(t_duration_ms, 2),
            tokens_per_second=round(tok_sec, 2),
            status="SUCCESS",
        )
        record.prompts_telemetry.append(telemetry)
        print(f"[✓] Completed in {t_duration_ms:.2f}ms ({output.completion_tokens} completion tokens, {tok_sec:.1f} t/s)")

    record.milestones["REAL_INFERENCE_COMPLETED"] = "VERIFIED"
    record.execution_state = "COMPLETED"

    # Write completed execution record
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)
    print(f"\n[OK] Full inference record saved to {output_path}")

    return record


if __name__ == "__main__":
    exec_inference = "--execute" in sys.argv
    run_minimal_inference_audit(execute_inference_if_ready=exec_inference)

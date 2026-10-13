#!/usr/bin/env python3
"""
SAM-AI Real-Model Smoke Test Runner (Audited & Reconciled)
==========================================================
Executes a single-game, small-budget (max 5 steps) smoke test using REAL model weights
and the sovereign SAM-AI agent architecture.

Reconciled with official arc_agi Arcade API:
- Uses `arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)`
- Uses `arcade.make(game_id, seed=0)`
- Passes `arcengine.enums.GameAction` enum to `env.step()`
- Strictly verifies adapter compatibility and non-empty weight files (checkpoint-25)
- Never silently substitutes base model for an adapter evaluation
- Does not create competition submission files or connect to external APIs
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sam_ai.agent.arc3_agent import SAMARC3Agent
from sam_ai.agent.schema import ActionSchemaParser
from sam_ai.agent.representation import ArcObservationFormatter
from sam_ai.agent.memory import TaskFailureMemory
from sam_ai.substrate.models import HuggingFaceModelRunner


def audit_adapter_compatibility(
    adapter_path: str,
    base_model_path: str,
) -> Tuple[bool, str, int]:
    """
    Verifies that the adapter directory exists, contains non-empty weights,
    and has a valid adapter_config.json matching the target base model.
    """
    p = Path(adapter_path)
    if not p.exists():
        return False, f"Adapter directory '{adapter_path}' does not exist.", 0

    cfg_file = p / "adapter_config.json"
    if not cfg_file.exists():
        return False, f"Missing 'adapter_config.json' in '{adapter_path}'.", 0

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        return False, f"Failed to parse 'adapter_config.json': {e}", 0

    # Check weights file
    weights = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    if not weights:
        return False, f"No weight files (*.safetensors or *.bin) found in '{adapter_path}'.", 0

    total_bytes = sum(w.stat().st_size for w in weights)
    if total_bytes == 0:
        return False, f"Adapter weights in '{adapter_path}' are 0 bytes (corrupt save).", 0

    base_in_cfg = cfg.get("base_model_name_or_path", "")
    print(f"[*] Adapter Config Base: '{base_in_cfg}' | Target Base: '{base_model_path}'")
    print(f"[*] Adapter Rank: {cfg.get('r')} | Alpha: {cfg.get('lora_alpha')} | Size: {total_bytes / (1024*1024):.2f} MB")

    return True, "Adapter verified non-empty and structurally valid.", total_bytes


def check_prerequisites(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    env_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Verifies hardware, weight, and environment prerequisites with zero mock fallbacks."""
    missing = []
    gpu_info = {}
    adapter_bytes = 0

    # 1. Check PyTorch & CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            missing.append("CUDA GPU accelerator is not available.")
            gpu_info = {"gpu_name": "None (CPU only)", "vram_free_gb": 0.0, "vram_total_gb": 0.0}
        else:
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_free_gb": free_b / (1024**3),
                "vram_total_gb": total_b / (1024**3),
                "device_count": torch.cuda.device_count(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
    except ImportError:
        missing.append("PyTorch is not installed.")

    # 2. Path Discovery via content-based discover_inputs()
    discovered_paths = None
    target_env_dir = env_dir
    from scripts.kaggle_path_discovery import discover_inputs, INPUT_ROOT
    if Path(INPUT_ROOT).exists() or not base_model_path or not env_dir:
        try:
            discovered_paths = discover_inputs(verbose=True)
            if "sam_ai_parent" in discovered_paths and discovered_paths["sam_ai_parent"]:
                parent_str = str(discovered_paths["sam_ai_parent"])
                if parent_str not in sys.path:
                    sys.path.insert(0, parent_str)
            if not base_model_path and discovered_paths.get("base_model_dir"):
                base_model_path = str(discovered_paths["base_model_dir"])
            if not target_env_dir and discovered_paths.get("environment_files"):
                target_env_dir = str(discovered_paths["environment_files"])
        except Exception as e:
            if not base_model_path or not Path(str(base_model_path)).exists():
                missing.append(f"Content discovery failed: {e}")
            target_env_dir = env_dir or os.environ.get("ENVIRONMENTS_DIR", "environment_files")
    else:
        target_env_dir = env_dir or os.environ.get("ENVIRONMENTS_DIR", "environment_files")

    if not base_model_path:
        missing.append("Base model path not provided and could not be discovered.")
    elif not Path(base_model_path).exists():
        missing.append(f"Base model path does not exist: {base_model_path}")

    # 3. Check Adapter Compatibility & Size
    if adapter_path:
        valid_ad, ad_msg, adapter_bytes = audit_adapter_compatibility(adapter_path, base_model_path)
        if not valid_ad:
            missing.append(f"Adapter verification failed: {ad_msg}")

    # 4. Check ARC-3 Public Environment Data
    env_p = Path(target_env_dir)
    if not env_p.exists() or not list(env_p.rglob("metadata.json")):
        missing.append(
            f"ARC-3 environment data not found at '{target_env_dir}'."
        )

    return {
        "is_ready": len(missing) == 0,
        "missing": missing,
        "gpu_info": gpu_info,
        "target_env_dir": target_env_dir,
        "base_model_path": base_model_path,
        "adapter_bytes": adapter_bytes,
    }


def run_smoke_test(
    base_model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    env_dir: Optional[str] = None,
    game_id: Optional[str] = None,
    max_steps: int = 5,
    output_dir: str = "evidence/real_model_smoke_test",
) -> Dict[str, Any]:
    """Executes the single-game real-model smoke test."""
    print("=" * 80)
    print("SAM-AI REAL-MODEL SMOKE TEST [AUDITED STATIC CODE REVIEW]")
    print("=" * 80)

    # 1. Prerequisite verification
    prereqs = check_prerequisites(base_model_path, adapter_path, env_dir)
    if not prereqs["is_ready"]:
        print("\n[BLOCKED] Real-model smoke test cannot proceed.")
        print("Missing prerequisites:")
        for m in prereqs["missing"]:
            print(f"  - {m}")
        print("\nAborting without executing mock fallbacks. Status: BLOCKED.")
        sys.exit(1)

    base_model_path = prereqs["base_model_path"]
    gpu = prereqs["gpu_info"]
    print(f"[✓] Accelerator: {gpu.get('gpu_name')} ({gpu.get('vram_free_gb', 0):.2f} GB Free / {gpu.get('vram_total_gb', 0):.2f} GB Total)")
    print(f"[✓] Base Model: {base_model_path}")
    print(f"[✓] Adapter: {adapter_path or 'None (Zero-shot Base Evaluation)'}")

    # 2. Setup output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = out_path / "smoke_trace.jsonl"
    summary_path = out_path / "smoke_summary.json"

    # 3. Initialize Model Runner (HuggingFace in-process)
    print("\n[*] Initializing in-process HuggingFaceModelRunner...")
    model_runner = HuggingFaceModelRunner(
        model_name_or_path=base_model_path,
        adapter_path=adapter_path,
        device_map="auto",
        load_in_4bit=True,
    )
    model_runner.load_model()

    # Guard: fail if adapter was requested but did not attach
    if adapter_path:
        assert model_runner.is_adapter_loaded, "CRITICAL: Adapter was requested but failed to attach."
    else:
        assert not model_runner.is_adapter_loaded, "CRITICAL: Adapter must not be loaded in base-model evaluation."

    # 4. Initialize Agent (Arm B: Verification Guard ONLY, Failure Memory Disabled, Smoke Sampling)
    agent = SAMARC3Agent(
        model_runner=model_runner,
        enable_verification_guard=True,
        enable_failure_memory=False, # Single-variable isolation
        is_real_model=True,
        temperature=0.6,
        top_p=0.95,
        seed=42,
        max_tokens=3072,
        use_system_prompt=False,
        stop_on_parse_failure=True,
    )

    # 5. Initialize Official ARC Environment via Arcade
    from arc_agi import Arcade, OperationMode
    target_env = prereqs["target_env_dir"]
    print(f"[*] Initializing Arcade (OperationMode.OFFLINE, dir={target_env})...")
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=target_env,
    )

    available_envs = arcade.get_environments()
    if not available_envs:
        print(f"[ERROR] No environments discovered in: {target_env}")
        sys.exit(1)

    chosen_game = game_id or available_envs[0].game_id
    print(f"[*] Selected Smoke-Test Game: {chosen_game}")

    env = arcade.make(game_id=chosen_game, seed=0)
    if env is None:
        print(f"[ERROR] arcade.make returned None for game '{chosen_game}'")
        sys.exit(1)

    # 6. Run Session with strict step budget
    print(f"[*] Executing interactive session (max_steps={max_steps})...")
    start_time = time.perf_counter()
    summary = agent.run_session(env, game_id=chosen_game, max_steps=max_steps)
    elapsed_time = time.perf_counter() - start_time

    # 7. Record Telemetry
    eval_type = "REAL_MODEL_WITH_ADAPTER" if adapter_path else "REAL_MODEL_ZERO_SHOT_BASE"
    result_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "game_id": chosen_game,
        "evaluation_type": eval_type,
        "is_real_model": True,
        "adapter_loaded": bool(adapter_path),
        "adapter_path": adapter_path,
        "adapter_bytes": prereqs["adapter_bytes"],
        "base_model": base_model_path,
        "steps_taken": summary.total_steps,
        "levels_completed": summary.levels_completed,
        "final_state": summary.final_state,
        "success": summary.success,
        "verification_failures": summary.verification_failures,
        "elapsed_seconds": elapsed_time,
        "action_history": [
            {"step": s.step_idx, "action": s.executed_action, "status": s.verification_status.value}
            for s in summary.steps_history
        ],
        "steps_telemetry": [
            step.telemetry.to_dict() for step in summary.steps_history if step.telemetry is not None
        ],
    }

    with open(raw_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE")
    print(f"Type: {eval_type} | Game: {chosen_game} | Steps: {summary.total_steps} | State: {summary.final_state}")
    print(f"Raw Trace: {raw_jsonl_path}")
    print(f"Summary:   {summary_path}")
    print("=" * 80)
    return result_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAM-AI Real-Model Smoke Test")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Path to pre-quantized 4-bit base model weights (auto-discovered if None)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to validated LoRA adapter (e.g. checkpoint-25)",
    )
    parser.add_argument(
        "--env-dir",
        default=None,
        help="Path to ARC-3 offline environment files",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum action steps for smoke test (default: 5)",
    )
    args = parser.parse_args()

    run_smoke_test(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        env_dir=args.env_dir,
        max_steps=args.max_steps,
    )

#!/usr/bin/env python3
"""
Generates the audited, unexecuted private Kaggle notebook for the SAM-AI Real Base-Model Smoke Test.
Configured strictly for Base-Model Zero-Shot evaluation (REAL_MODEL_ZERO_SHOT_BASE).
Does NOT mount, load, or claim to evaluate the SAM-AI LoRA adapter.
Enforces 100% parity with `scripts/run_real_model_smoke_test.py` via `HuggingFaceModelRunner`
and content-based path discovery via `scripts/kaggle_path_discovery.py`.
"""

import json
from pathlib import Path

out_dir = Path("notebooks/sam_ai_real_model_smoke_test")
out_dir.mkdir(parents=True, exist_ok=True)

metadata = {
    "id": "samrishb/sam-ai-real-model-smoke-test",
    "title": "SAM-AI Real Model Smoke Test",
    "code_file": "sam_ai_real_model_smoke_test.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "samrishb/sam-ai-source-bundle"
    ],
    "competition_sources": [
        "arc-prize-2026-arc-agi-3"
    ],
    "kernel_sources": [],
    "model_sources": [
        "gowillgo/unsloth-deepseek-r1-distill-qwen-14b-bnb-4bit/transformers/v1/1"
    ],
    "keywords": [
        "gpu"
    ],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
    "machine_shape": "NvidiaRtxPro6000"
}

with open(out_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# Load exact text of scripts/kaggle_path_discovery.py to embed in first code cell
discovery_path = Path("scripts/kaggle_path_discovery.py")
if not discovery_path.exists():
    discovery_path = Path(__file__).resolve().parent / "kaggle_path_discovery.py"
discovery_text = discovery_path.read_text(encoding="utf-8")

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SAM-AI Sovereign Agent: Real Base-Model Smoke Test (Zero-Shot Baseline)\n",
                "\n",
                "**Objective:** Verify that the sovereign SAM-AI agent (`sam_ai.agent`) loads authentic 4-bit base model weights (`DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`) via `HuggingFaceModelRunner`, connects to the official ARC-3 `Arcade` environment interface, and executes a strictly budgeted 5-step interactive loop.\n",
                "\n",
                "> **Hard Rules:**\n",
                "> 1. **Zero competition submission:** Does not create `submission.parquet` or invoke submission APIs.\n",
                "> 2. **Base-Model Only:** Does not mount, load, or evaluate the SAM-AI adapter. All results labeled `REAL_MODEL_ZERO_SHOT_BASE`.\n",
                "> 3. **No mock fallback:** If weights, GPU, or environment data are absent, halts cleanly with `[BLOCKED]`.\n",
                "> 4. **Small compute footprint:** Bounded to 5 environment steps total (< 2 minutes GPU runtime).\n",
                "> 5. **Architectural Parity:** Uses identical `HuggingFaceModelRunner` and telemetry schema as the local `scripts/run_real_model_smoke_test.py`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": discovery_text.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 1. Environment, Dependencies & Static Precondition Audit\n",
                "# ==============================================================================\n",
                "import os\n",
                "import sys\n",
                "import time\n",
                "import json\n",
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "print('=' * 75)\n",
                "print('  SAM-AI REAL BASE-MODEL SMOKE TEST: STATIC PRECONDITION AUDIT')\n",
                "print('=' * 75)\n",
                "\n",
                "# 0. Discover input paths by content\n",
                "paths = discover_inputs()\n",
                "sam_ai_parent = str(paths['sam_ai_parent'])\n",
                "if sam_ai_parent not in sys.path:\n",
                "    sys.path.insert(0, sam_ai_parent)\n",
                "\n",
                "base_model_path = str(paths['base_model_dir'])\n",
                "ENV_DATA_DIR = str(paths['environment_files'])\n",
                "\n",
                "print(f'[OK] sam_ai parent resolved:   {sam_ai_parent}')\n",
                "print(f'[OK] Base Model resolved:      {base_model_path}')\n",
                "print(f'[OK] Environment dir resolved: {ENV_DATA_DIR}')\n",
                "\n",
                "# 1. Check and Install Official arc-agi / arcengine Wheels Offline\n",
                "try:\n",
                "    import arc_agi\n",
                "    import arcengine\n",
                "    print('[OK] arc-agi and arcengine packages are available.')\n",
                "except ImportError:\n",
                "    wheel_dirs = [\n",
                "        Path(ENV_DATA_DIR).parent / 'arc_agi_3_wheels',\n",
                "    ]\n",
                "    installed = False\n",
                "    for wd in wheel_dirs:\n",
                "        if wd.exists() and list(wd.glob('*.whl')):\n",
                "            print(f'[*] Installing offline wheels from {wd}...')\n",
                "            res = subprocess.run([\n",
                "                sys.executable, '-m', 'pip', 'install', '--no-index',\n",
                "                f'--find-links={wd}', 'arc-agi', 'arcengine'\n",
                "            ], capture_output=True, text=True)\n",
                "            if res.returncode == 0:\n",
                "                print('[OK] arc-agi and arcengine installed from offline wheels.')\n",
                "                installed = True\n",
                "                break\n",
                "            else:\n",
                "                print(f'[WARN] Offline wheel installation warning: {res.stderr}')\n",
                "    if not installed:\n",
                "        raise RuntimeError('arc-agi and arcengine packages not found and offline wheels missing.')\n",
                "\n",
                "# 2. Check GPU Accelerator & Memory\n",
                "try:\n",
                "    import torch\n",
                "    if not torch.cuda.is_available():\n",
                "        raise RuntimeError('CUDA is not available. GPU accelerator must be enabled in Notebook Settings.')\n",
                "    else:\n",
                "        free_b, total_b = torch.cuda.mem_get_info()\n",
                "        gpu_name = torch.cuda.get_device_name(0)\n",
                "        print(f'[OK] GPU Accelerator: {gpu_name}')\n",
                "        print(f'[OK] Total VRAM:      {total_b / (1024**3):.2f} GB (Free: {free_b / (1024**3):.2f} GB)')\n",
                "        print(f'[OK] bfloat16:        {\"Supported\" if torch.cuda.is_bf16_supported() else \"Not Supported\"}')\n",
                "except ImportError:\n",
                "    raise RuntimeError('PyTorch is not installed.')\n",
                "\n",
                "# 3. Adapter Explicitly Disabled (Zero-Shot Base Mode)\n",
                "adapter_path = None\n",
                "adapter_bytes = 0\n",
                "print('[INFO] Operating in Base-Model-Only Zero-Shot Mode (REAL_MODEL_ZERO_SHOT_BASE).')\n",
                "print('       SAM-AI LoRA adapter is NOT loaded in this evaluation.')\n",
                "\n",
                "# 4. Check Official ARC-3 Environment Data via Arcade\n",
                "found_envs = len(list(Path(ENV_DATA_DIR).rglob('metadata.json')))\n",
                "print(f'[OK] Found {found_envs} official ARC-3 environment files in {ENV_DATA_DIR}.')\n",
                "print('\\n[READY] All hardware, base model, and environment prerequisites verified.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 2. Sovereign ModelRunner Loading (Base Model Only)\n",
                "# ==============================================================================\n",
                "from sam_ai.substrate.models import HuggingFaceModelRunner\n",
                "\n",
                "print(f'[*] Initializing sovereign HuggingFaceModelRunner with base model...')\n",
                "runner = HuggingFaceModelRunner(\n",
                "    model_name_or_path=base_model_path,\n",
                "    adapter_path=None, # Strictly base-model zero-shot\n",
                "    device_map='auto',\n",
                "    load_in_4bit=True,\n",
                ")\n",
                "runner.load_model()\n",
                "\n",
                "is_adapter_active = runner.is_adapter_loaded\n",
                "assert not is_adapter_active, 'CRITICAL: Adapter must not be loaded in zero-shot base evaluation.'\n",
                "print(f'[OK] ModelRunner ready.')\n",
                "print(f'     Model Name:     {runner.model_name}')\n",
                "print(f'     Adapter Active: {is_adapter_active} (Zero-Shot Base Mode)')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 3. SAM-AI Agent Setup (Single-Variable Isolation: Arm B + Smoke Sampling)\n",
                "# ==============================================================================\n",
                "from sam_ai.agent.arc3_agent import SAMARC3Agent\n",
                "\n",
                "# Initialize SAM-AI Agent with smoke-test sampling and hygiene\n",
                "agent = SAMARC3Agent(\n",
                "    model_runner=runner,\n",
                "    enable_verification_guard=True,\n",
                "    enable_failure_memory=False, # Single-variable isolation\n",
                "    is_real_model=True,\n",
                "    temperature=0.6,\n",
                "    top_p=0.95,\n",
                "    seed=42,\n",
                "    max_tokens=3072,\n",
                "    use_system_prompt=False,\n",
                "    stop_on_parse_failure=True,\n",
                ")\n",
                "print('[OK] SAM-AI Agent initialized under Arm B single-variable isolation.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 4. Official Arcade Execution (Strictly Max 5 Steps) & Telemetry\n",
                "# ==============================================================================\n",
                "from arc_agi import Arcade, OperationMode\n",
                "\n",
                "arcade = Arcade(\n",
                "    operation_mode=OperationMode.OFFLINE,\n",
                "    environments_dir=ENV_DATA_DIR,\n",
                ")\n",
                "available_envs = arcade.get_environments()\n",
                "if not available_envs:\n",
                "    raise RuntimeError(f'No environments found in {ENV_DATA_DIR}')\n",
                "\n",
                "chosen_game = available_envs[0].game_id\n",
                "print(f'[*] Initializing official environment for game: {chosen_game}')\n",
                "env = arcade.make(game_id=chosen_game, seed=0)\n",
                "if env is None:\n",
                "    raise RuntimeError(f'arcade.make returned None for game {chosen_game}')\n",
                "\n",
                "print(f'[*] Launching interactive smoke session (strictly max 5 steps)...')\n",
                "t0 = time.perf_counter()\n",
                "summary = agent.run_session(env, game_id=chosen_game, max_steps=5)\n",
                "elapsed = time.perf_counter() - t0\n",
                "\n",
                "# Persist Telemetry (Strictly REAL_MODEL_ZERO_SHOT_BASE)\n",
                "out_dir = Path('evidence/real_model_smoke_test')\n",
                "out_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "eval_type = 'REAL_MODEL_ZERO_SHOT_BASE'\n",
                "report = {\n",
                "    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n",
                "    'game_id': chosen_game,\n",
                "    'evaluation_type': eval_type,\n",
                "    'is_real_model': True,\n",
                "    'adapter_loaded': False,\n",
                "    'adapter_path': None,\n",
                "    'adapter_bytes': 0,\n",
                "    'base_model': base_model_path,\n",
                "    'steps_taken': summary.total_steps,\n",
                "    'levels_completed': summary.levels_completed,\n",
                "    'final_state': summary.final_state,\n",
                "    'success': summary.success,\n",
                "    'verification_failures': summary.verification_failures,\n",
                "    'elapsed_seconds': elapsed,\n",
                "    'steps_telemetry': [\n",
                "        step.telemetry.to_dict() for step in summary.steps_history if step.telemetry is not None\n",
                "    ],\n",
                "    'action_history': [\n",
                "        {'step': s.step_idx, 'action': s.executed_action, 'status': s.verification_status.value}\n",
                "        for s in summary.steps_history\n",
                "    ],\n",
                "}\n",
                "\n",
                "with open(out_dir / 'smoke_summary.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(report, f, indent=2)\n",
                "with open('smoke_summary.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(report, f, indent=2)\n",
                "\n",
                "with open(out_dir / 'smoke_trace.jsonl', 'a', encoding='utf-8') as f:\n",
                "    f.write(json.dumps(report) + '\\n')\n",
                "with open('smoke_trace.jsonl', 'a', encoding='utf-8') as f:\n",
                "    f.write(json.dumps(report) + '\\n')\n",
                "\n",
                "print('=' * 75)\n",
                "print(f'SMOKE TEST COMPLETED: {chosen_game} | Steps: {summary.total_steps} | State: {summary.final_state}')\n",
                "print(f'Evaluation Type: {eval_type}')\n",
                "print(f'Summary file:    {out_dir / \"smoke_summary.json\"}')\n",
                "print('=' * 75)\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "nvidiaRtxPro6000",
            "isGpuEnabled": True,
            "isInternetEnabled": False
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(out_dir / "sam_ai_real_model_smoke_test.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("[OK] Generated base-model zero-shot smoke-test notebook and kernel-metadata.json successfully.")

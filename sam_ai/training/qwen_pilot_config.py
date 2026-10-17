"""SAM-AI Qwen3.8-27B LoRA Pilot Configuration & Pre-Flight Feasibility Audit.

Experiment Designation: EXP-SAM-QWEN38-PILOT-01
Status: NOT_RUN / SPECIFICATION_ONLY

Designed for modest, reversible capability improvement on structured action schema and reasoning.
Performs honest hardware audit and marks status BLOCKED if GPU VRAM is insufficient.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from sam_ai.models.loader import PRIMARY_MODEL_ID, get_model_spec, get_default_target_modules


@dataclass
class QwenTrainingPilotSpec:
    """Rigorous specifications for Qwen3.8-27B LoRA training pilot."""
    experiment_id: str = "EXP-SAM-QWEN38-PILOT-01"
    base_model_id: str = PRIMARY_MODEL_ID
    target_task_family: str = "Structured Action JSON & Multi-Step Reasoning"
    dataset_provenance: str = "Curated Synthetic Verified ARC-3 Action Trajectories (Apache 2.0)"
    dataset_license: str = "Apache-2.0"
    training_method: str = "QLoRA (4-bit NF4) or BF16 LoRA with Gradient Checkpointing"
    
    # Architecture and Target Modules
    target_modules: List[str] = field(default_factory=get_default_target_modules)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Modest Training Budget
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_steps: int = 50  # Modest, reversible pilot
    save_steps: int = 25
    eval_steps: int = 25
    optimizer: str = "paged_adamw_8bit"
    
    # Memory Planning (27B Parameters)
    # Full BF16: 27B * 2 bytes = ~54 GB weights alone
    # 4-bit NF4: 27B * 0.5 bytes = ~13.5 GB weights alone
    est_weights_vram_gb_bf16: float = 54.0
    est_weights_vram_gb_4bit: float = 13.5
    est_min_total_vram_gb_qlora: float = 18.0
    est_min_total_vram_gb_bf16_lora: float = 64.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_training_feasibility() -> Dict[str, Any]:
    """
    Audits actual local hardware to determine whether Qwen3.8-27B training can run.
    Returns honest feasibility analysis. Never pretends configuration is execution.
    """
    spec = QwenTrainingPilotSpec()
    missing_prereqs = []
    gpu_available = False
    vram_free_gb = 0.0
    vram_total_gb = 0.0
    gpu_name = "None"
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            free_b, total_b = torch.cuda.mem_get_info()
            vram_free_gb = free_b / (1024**3)
            vram_total_gb = total_b / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            
            # Check feasibility against QLoRA minimum
            if vram_free_gb < spec.est_min_total_vram_gb_qlora:
                missing_prereqs.append(
                    f"Insufficient free VRAM ({vram_free_gb:.1f} GB available). "
                    f"4-bit QLoRA pilot requires at least {spec.est_min_total_vram_gb_qlora:.1f} GB free."
                )
        else:
            missing_prereqs.append("No CUDA GPU detected. Local environment has CPU only.")
    except ImportError:
        missing_prereqs.append("PyTorch is not installed.")

    # Check PEFT and BitsAndBytes
    try:
        import peft
    except ImportError:
        missing_prereqs.append("PEFT library is not installed (`pip install peft`).")

    try:
        import bitsandbytes
    except ImportError:
        missing_prereqs.append("bitsandbytes library is not installed (`pip install bitsandbytes`).")

    is_feasible = len(missing_prereqs) == 0

    return {
        "status": "READY" if is_feasible else "BLOCKED",
        "is_feasible": is_feasible,
        "pilot_spec": spec.to_dict(),
        "hardware": {
            "gpu_name": gpu_name,
            "vram_free_gb": round(vram_free_gb, 2),
            "vram_total_gb": round(vram_total_gb, 2),
        },
        "blockers": missing_prereqs,
        "recommended_remote_hardware": (
            "NVIDIA RTX PRO 6000 (95 GB VRAM) on Kaggle or NVIDIA H100 (80 GB VRAM) cloud instance"
        ),
        "execution_state": "NOT_RUN",
    }

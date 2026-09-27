"""Unit tests for SAM-AI Foundation Model Loader and Config."""

import pytest
import torch
from sam_ai.models.loader import SAMModelManager, ModelConfig, get_default_target_modules


def test_model_config_initialization():
    cfg = ModelConfig(
        model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        precision="bfloat16",
        use_lora=True,
        lora_rank=32,
    )
    assert cfg.model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    assert cfg.precision == "bfloat16"
    assert cfg.lora_rank == 32
    assert "q_proj" in get_default_target_modules()


def test_sam_model_manager_dtype_resolution():
    manager = SAMModelManager(ModelConfig(precision="float32"))
    assert manager.get_torch_dtype() == torch.float32

    manager_fp16 = SAMModelManager(ModelConfig(precision="float16"))
    assert manager_fp16.get_torch_dtype() == torch.float16

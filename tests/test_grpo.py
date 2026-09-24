"""Unit tests for the DeepSeek-R1 style GRPO Reinforcement Learning Engine."""

import pytest
import numpy as np
from core.neural.transformer import SovereignNeuralTransformer, TransformerConfig
from core.neural.grpo_trainer import GRPOTrainer, GRPOSummary


@pytest.fixture
def grpo_system():
    config = TransformerConfig(
        vocab_size=128,
        seq_len=32,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1,
    )
    model = SovereignNeuralTransformer(config)
    trainer = GRPOTrainer(model=model, group_size=4)
    return model, trainer


def test_grpo_advantage_normalization(grpo_system):
    _, trainer = grpo_system
    rewards = [1.0, 0.0, 0.5, 0.5]
    adv = trainer.compute_group_advantages(rewards)
    assert len(adv) == 4
    # Highest reward should have the highest advantage
    assert adv[0] > adv[1]
    assert np.isclose(np.mean(adv), 0.0, atol=1e-5)


def test_grpo_reinforcement_training_step(grpo_system):
    _, trainer = grpo_system

    # Deterministic verifier: rewards outputs that contain specific reasoning tokens
    def mock_math_verifier(output: str) -> float:
        score = 0.1
        if "42" in output:
            score += 0.8
        if len(output) > 10:
            score += 0.1
        return score

    summary: GRPOSummary = trainer.train_grpo_step(
        prompt="Solve 6*7=",
        reward_verifier=mock_math_verifier,
        max_tokens=15,
    )

    assert summary.group_size == 4
    assert isinstance(summary.best_output, str)
    assert summary.best_reward >= 0.0
    assert trainer.step_count == 1

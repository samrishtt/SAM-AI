"""Verification Experiment for SAM-AI GRPO Training Loop.

Executes a local synthetic training experiment and records empirical metrics:
- initial loss vs final loss
- initial reward vs final reward
- gradient norms across steps
- parameter displacement ||theta_final - theta_initial||_2
- ratio statistics (mean, min, max)
- clipping fraction
"""

import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_ai.training.grpo_core import (
    GRPOTrainingConfig,
    GRPORollout,
    GRPOTrainer,
)


class SyntheticReasoningPolicy(nn.Module):
    """Tiny policy model with 16-token vocab and embedding layer."""
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        return self.fc(x)


def run_synthetic_grpo_verification():
    torch.manual_seed(42)
    model = SyntheticReasoningPolicy(vocab_size=16, hidden_dim=16)
    config = GRPOTrainingConfig(
        group_size=4,
        clip_epsilon=0.2,
        learning_rate=0.08,
        ppo_epochs_per_rollout=2,
        max_grad_norm=1.0,
    )
    trainer = GRPOTrainer(policy_model=model, config=config)

    prompt = torch.tensor([1, 2], dtype=torch.long)
    target_token = 9  # Objective: Learn to emit token 9

    def verifier(comp_ids: torch.Tensor) -> float:
        # Binary ground truth reward
        return 1.0 if target_token in comp_ids.tolist() else 0.0

    def generator(p_ids: torch.Tensor) -> torch.Tensor:
        logits = model(p_ids.unsqueeze(0))
        probs = F.softmax(logits[0, -1] / 1.0, dim=-1)
        return torch.multinomial(probs, num_samples=2)

    initial_params = [p.clone().detach() for p in model.parameters()]
    
    steps_log = []
    
    # Run 15 GRPO rollout and training steps
    for step in range(15):
        rollouts = trainer.collect_group_rollouts(prompt, generator, verifier)
        telem = trainer.train_step(rollouts)
        steps_log.append(telem)

    # Compute total parameter displacement
    final_params = [p.clone().detach() for p in model.parameters()]
    total_delta = math.sqrt(
        sum((p_f - p_i).norm().item() ** 2 for p_i, p_f in zip(initial_params, final_params))
    )

    initial_loss = steps_log[0]["loss"]
    final_loss = steps_log[-1]["loss"]
    initial_reward = steps_log[0]["mean_reward"]
    final_reward = steps_log[-1]["mean_reward"]
    final_grad_norm = steps_log[-1].get("grad_norm", 0.0)
    final_clip_fraction = steps_log[-1].get("clip_fraction", 0.0)
    final_mean_ratio = steps_log[-1].get("mean_ratio", 1.0)
    final_min_ratio = steps_log[-1].get("min_ratio", 1.0)
    final_max_ratio = steps_log[-1].get("max_ratio", 1.0)

    print("=======================================================================")
    print("           SAM-AI GRPO SYNTHETIC VERIFICATION REPORT                   ")
    print("=======================================================================")
    print(f"Initial Policy Loss:        {initial_loss:.4f}")
    print(f"Final Policy Loss:          {final_loss:.4f}")
    print(f"Initial Mean Reward:        {initial_reward:.4f}")
    print(f"Final Mean Reward:          {final_reward:.4f}")
    print(f"Final Gradient Norm:        {final_grad_norm:.4f}")
    print(f"Parameter Displacement:     {total_delta:.4f} (||theta_T - theta_0||)")
    print(f"Final Ratio Statistics:     mean={final_mean_ratio:.4f}, min={final_min_ratio:.4f}, max={final_max_ratio:.4f}")
    print(f"Final Clipping Fraction:    {final_clip_fraction:.4f}")
    print("=======================================================================")
    
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_reward": initial_reward,
        "final_reward": final_reward,
        "final_grad_norm": final_grad_norm,
        "param_delta": total_delta,
        "mean_ratio": final_mean_ratio,
        "min_ratio": final_min_ratio,
        "max_ratio": final_max_ratio,
        "clip_fraction": final_clip_fraction,
    }


if __name__ == "__main__":
    run_synthetic_grpo_verification()

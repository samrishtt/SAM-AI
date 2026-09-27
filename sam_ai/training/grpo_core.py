"""Group Relative Policy Optimization (GRPO) Core Engine.

Mathematically rigorous implementation of GRPO (DeepSeek-R1 / Shao et al., 2024).
Fixes importance sampling ratio computation, group-normalized advantage estimation,
and token-level completion masking.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOTrainingConfig:
    """Configuration parameters for GRPO training."""
    group_size: int = 4
    clip_epsilon: float = 0.2
    beta_kl: float = 0.04
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    ppo_epochs_per_rollout: int = 2
    temperature: float = 0.8
    top_p: float = 0.95


@dataclass
class GRPORollout:
    """Container holding a single generated rollout trajectory and fixed reference stats."""
    prompt_ids: torch.Tensor          # [prompt_len]
    completion_ids: torch.Tensor      # [completion_len]
    full_input_ids: torch.Tensor      # [prompt_len + completion_len]
    prompt_mask: torch.Tensor         # [prompt_len + completion_len], 1 for completion, 0 for prompt
    old_log_probs: torch.Tensor       # [completion_len], evaluated strictly at rollout time under pi_old
    reward: float = 0.0
    advantage: float = 0.0
    completion_mask: Optional[torch.Tensor] = None  # [completion_len], 1 for valid tokens, 0 for pad


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes group-normalized advantages:
    A_i = (r_i - mean(r)) / (std(r) + eps)
    If all rewards in the group are identical (std == 0), advantages are strictly zero.
    
    Args:
        rewards: 1D Tensor of shape [G]
        eps: Small constant for numerical stability
    Returns:
        advantages: 1D Tensor of shape [G]
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)

    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=True)

    if std_r < eps:
        return torch.zeros_like(rewards)

    return (rewards - mean_r) / (std_r + eps)


def extract_token_log_probs(
    logits: torch.Tensor,
    target_ids: torch.Tensor
) -> torch.Tensor:
    """
    Extracts log p(target_ids[t] | context_{<t}) from unnormalized logits.
    
    Args:
        logits: Tensor of shape [B, S, V] (predicts token at index t from context < t)
        target_ids: Tensor of shape [B, S]
    Returns:
        log_probs: Tensor of shape [B, S]
    """
    # log_softmax over vocabulary dimension
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log prob of the actual target token
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return per_token_log_probs


class GRPOTrainer:
    """
    Mathematically sound Group Relative Policy Optimization Trainer.
    
    Key Mathematical Guarantees:
    1. pi_old is recorded strictly during the rollout phase and detached.
    2. The importance sampling ratio rho_t = exp(log_pi_theta - log_pi_old) diverges from 1.0
       as theta updates, enabling the PPO clipping boundary to function.
    3. Token masking ensures only completion tokens contribute to policy loss.
    4. Group advantage normalizes rewards across each prompt's G candidate outputs.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        config: Optional[GRPOTrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.config = config or GRPOTrainingConfig()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

    def evaluate_sequence_log_probs(
        self,
        model: nn.Module,
        full_input_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Computes token log probabilities for the completion tokens of full_input_ids.
        
        Args:
            model: The Transformer policy model
            full_input_ids: Tensor of shape [1, total_len]
            prompt_len: Length of the prompt prefix
        Returns:
            completion_log_probs: Tensor of shape [completion_len]
        """
        outputs = model(input_ids=full_input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [1, total_len, vocab_size]

        # Shift logits so logits[:, t, :] predicts full_input_ids[:, t + 1]
        shift_logits = logits[:, :-1, :]          # [1, total_len - 1, vocab_size]
        shift_targets = full_input_ids[:, 1:]     # [1, total_len - 1]

        all_token_log_probs = extract_token_log_probs(shift_logits, shift_targets).squeeze(0)  # [total_len - 1]
        # Completion tokens begin at index (prompt_len - 1) in the shifted sequence
        completion_log_probs = all_token_log_probs[prompt_len - 1:]
        return completion_log_probs

    def collect_group_rollouts(
        self,
        prompt_ids: torch.Tensor,
        generate_fn: Callable[[torch.Tensor], torch.Tensor],
        verifier_fn: Callable[[torch.Tensor], float],
    ) -> List[GRPORollout]:
        """
        Samples G candidate completions from the current policy pi_old, records
        old_log_probs under no_grad(), and evaluates verifier rewards.
        """
        rollouts: List[GRPORollout] = []
        prompt_len = prompt_ids.size(0)

        with torch.no_grad():
            for _ in range(self.config.group_size):
                # Generate completion tokens
                completion_ids = generate_fn(prompt_ids)
                comp_len = completion_ids.size(0)
                
                full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
                
                # Compute and record old_log_probs strictly under pi_old
                old_logps = self.evaluate_sequence_log_probs(self.policy, full_ids, prompt_len).detach()
                
                # Mask: 0 for prompt, 1 for completion
                mask = torch.zeros(prompt_len + comp_len, device=prompt_ids.device)
                mask[prompt_len:] = 1.0

                # Compute scalar reward via deterministic verifier
                reward = verifier_fn(completion_ids)

                rollouts.append(
                    GRPORollout(
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        full_input_ids=full_ids.squeeze(0),
                        prompt_mask=mask,
                        old_log_probs=old_logps,
                        reward=reward,
                    )
                )

        # Standardize group advantages across the G completions
        rewards_tensor = torch.tensor([r.reward for r in rollouts], dtype=torch.float32, device=prompt_ids.device)
        advantages = compute_group_advantages(rewards_tensor)

        for i, adv in enumerate(advantages):
            rollouts[i].advantage = adv.item()

        return rollouts

    def compute_grpo_loss(
        self,
        rollouts: List[GRPORollout],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the clipped GRPO surrogate loss, importance ratios, and diagnostic telemetry.
        """
        total_policy_loss = torch.tensor(0.0, device=rollouts[0].full_input_ids.device)
        total_kl_loss = torch.tensor(0.0, device=rollouts[0].full_input_ids.device)
        
        all_ratios = []
        clip_flags = []
        G = len(rollouts)

        for rollout in rollouts:
            prompt_len = rollout.prompt_ids.size(0)
            full_ids = rollout.full_input_ids.unsqueeze(0)
            advantage = rollout.advantage

            # Compute log probs under CURRENT policy pi_theta
            curr_log_probs = self.evaluate_sequence_log_probs(self.policy, full_ids, prompt_len)
            old_log_probs = rollout.old_log_probs

            # Importance sampling ratio: rho_t = exp(log_pi_theta - log_pi_old)
            log_ratio = curr_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            all_ratios.append(ratio.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantage
            clipped_obj = torch.min(surr1, surr2)

            # Check clipping fraction
            is_clipped = (ratio < (1.0 - self.config.clip_epsilon)) | (ratio > (1.0 + self.config.clip_epsilon))
            clip_flags.append(is_clipped.float().detach())

            # Per-completion masked loss: sequence-length normalized over valid non-pad tokens
            if rollout.completion_mask is not None:
                mask = rollout.completion_mask.to(clipped_obj.device)
                comp_loss = -(clipped_obj * mask).sum() / (mask.sum() + 1e-8)
            else:
                comp_loss = -clipped_obj.mean()
            total_policy_loss = total_policy_loss + comp_loss

            # Reference model KL divergence (Schulman estimator) if ref_model present
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logps = self.evaluate_sequence_log_probs(self.ref_model, full_ids, prompt_len)
                log_ratio_ref = ref_logps - curr_log_probs
                kl_div = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                if rollout.completion_mask is not None:
                    mask = rollout.completion_mask.to(kl_div.device)
                    total_kl_loss = total_kl_loss + (kl_div * mask).sum() / (mask.sum() + 1e-8)
                else:
                    total_kl_loss = total_kl_loss + kl_div.mean()

        # Mean across group G
        loss = (total_policy_loss / G) + (self.config.beta_kl * total_kl_loss / G)

        # Aggregate telemetry
        cat_ratios = torch.cat(all_ratios)
        cat_clips = torch.cat(clip_flags)

        rewards_list = [r.reward for r in rollouts]
        mean_rew = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
        var_rew = sum((x - mean_rew) ** 2 for x in rewards_list) / len(rewards_list) if rewards_list else 0.0
        std_rew = math.sqrt(var_rew)

        telemetry = {
            "loss": loss.item(),
            "policy_loss": (total_policy_loss / G).item(),
            "kl_loss": (total_kl_loss / G).item() if self.ref_model is not None else 0.0,
            "mean_ratio": cat_ratios.mean().item(),
            "max_ratio": cat_ratios.max().item(),
            "min_ratio": cat_ratios.min().item(),
            "clip_fraction": cat_clips.mean().item(),
            "mean_reward": mean_rew,
            "reward_std": std_rew,
        }

        return loss, telemetry

    def train_step(self, rollouts: List[GRPORollout]) -> Dict[str, float]:
        """
        Executes one policy update cycle across the collected rollouts.
        Supports multi-epoch PPO/GRPO updates per rollout batch.
        Tracks gradient norm and parameter displacement.
        """
        last_telemetry: Dict[str, float] = {}

        # Capture parameter weights before update
        params_before = [p.clone().detach() for p in self.policy.parameters()]

        for _ in range(self.config.ppo_epochs_per_rollout):
            self.optimizer.zero_grad()
            loss, telemetry = self.compute_grpo_loss(rollouts)
            loss.backward()

            grad_norm = 0.0
            if self.config.max_grad_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item())

            self.optimizer.step()
            telemetry["grad_norm"] = grad_norm
            last_telemetry = telemetry

        # Calculate parameter displacement ||theta_after - theta_before||_2
        param_delta_sq = 0.0
        for p_before, p_after in zip(params_before, self.policy.parameters()):
            param_delta_sq += float((p_after.detach() - p_before).norm().item() ** 2)
        last_telemetry["param_delta"] = math.sqrt(param_delta_sq)

        return last_telemetry

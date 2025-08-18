"""PPO training functionality."""

from typing import Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.rl.trainer_config import TrainerConfig


def compute_ppo_losses(
    minibatch: TensorDict,
    new_logprob: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: TrainerConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute PPO losses for policy and value functions."""
    # Policy loss
    pg_loss = compute_dpo_policy_loss(
        importance_sampling_ratio,
        adv,
    )

    returns = minibatch["returns"]
    old_values = minibatch["values"]

    # Value loss
    newvalue_reshaped = newvalue.view(returns.shape)
    if trainer_cfg.ppo.clip_vloss:
        v_loss_unclipped = (newvalue_reshaped - returns) ** 2
        vf_clip_coef = trainer_cfg.ppo.vf_clip_coef
        v_clipped = old_values + torch.clamp(
            newvalue_reshaped - old_values,
            -vf_clip_coef,
            vf_clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue_reshaped - returns) ** 2).mean()

    entropy_loss = entropy.mean()

    # Compute metrics
    with torch.no_grad():
        logratio = new_logprob - minibatch["act_log_prob"]
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
        clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


def compute_dpo_policy_loss(
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    alpha: float = 2.0,
    beta: float = 0.6,
) -> Tensor:
    """
    Computes the policy loss for Discovered Policy Optimisation (DPO).

    This function implements the closed-form drift function discovered in the
    "Discovered Policy Optimisation" paper (Lu et al., 2022) and uses it
    to calculate the final policy loss.

    Args:
        importance_sampling_ratio (Tensor): The probability ratio r_t = pi_new / pi_old.
        adv (Tensor): The advantage estimate A(s, t).
        alpha (float): The alpha parameter for the positive advantage case. Defaults to 2.0.
        beta (float): The beta parameter for the negative advantage case. Defaults to 0.6.

    Returns:
        Tensor: The final scalar policy loss to be minimized.
    """
    ratio = importance_sampling_ratio
    logratio = torch.log(ratio)

    # Positive advantage case (cautious optimism)
    term1_pos = (ratio - 1) * adv
    term2_pos = alpha * torch.tanh(term1_pos / alpha)
    drift_pos = torch.relu(term1_pos - term2_pos)

    # Negative advantage case (rollback)
    term1_neg = logratio * adv
    term2_neg = beta * torch.tanh(term1_neg / beta)
    drift_neg = torch.relu(term1_neg - term2_neg)

    # The drift is the core DPO penalty, applied based on the sign of the advantage
    drift = torch.where(adv >= 0, drift_pos, drift_neg)

    # The objective to maximize is (r * A - Drift).
    # Therefore, the loss to minimize is the negative of that: (Drift - r * A).
    pg_loss = (drift - ratio * adv).mean()

    return pg_loss

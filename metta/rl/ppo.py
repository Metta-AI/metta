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
    pg_loss1 = -adv * importance_sampling_ratio
    pg_loss2 = -adv * torch.clamp(
        importance_sampling_ratio, 1 - trainer_cfg.ppo.clip_coef, 1 + trainer_cfg.ppo.clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
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

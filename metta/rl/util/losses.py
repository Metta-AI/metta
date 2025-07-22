"""Loss computation functions for PPO training."""

import logging
from typing import Any, Dict, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed

logger = logging.getLogger(__name__)


def compute_ppo_losses(
    minibatch: TensorDict,
    new_logprobs: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute PPO losses for policy and value functions."""
    # Policy loss
    pg_loss1 = -adv * importance_sampling_ratio
    pg_loss2 = -adv * torch.clamp(
        importance_sampling_ratio, 1 - trainer_cfg.ppo.clip_coef, 1 + trainer_cfg.ppo.clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
    if trainer_cfg.ppo.clip_vloss:
        v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
        vf_clip_coef = trainer_cfg.ppo.vf_clip_coef
        v_clipped = minibatch["values"] + torch.clamp(
            newvalue_reshaped - minibatch["values"],
            -vf_clip_coef,
            vf_clip_coef,
        )
        v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

    entropy_loss = entropy.mean()

    # Compute metrics
    with torch.no_grad():
        logratio = new_logprobs - minibatch["logprobs"]
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
        clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


def process_minibatch_update(
    policy: torch.nn.Module,
    experience: Experience,
    minibatch: TensorDict,
    trainer_cfg: Any,
    indices: Tensor,
    prio_weights: Tensor,
    kickstarter: Any,
    agent_step: int,
    losses: Losses,
    device: torch.device,
) -> Tensor:
    """Process a single minibatch update and return the total loss."""
    # The policy's training forward pass returns a TD with required tensors for loss calculation.
    policy_output = policy(minibatch, action=minibatch["actions"])
    new_logprobs = policy_output["action_log_prob"].reshape(minibatch["logprobs"].shape)
    entropy = policy_output["entropy"]
    newvalue = policy_output["value"]
    full_logprobs = policy_output["log_probs"]

    logratio = new_logprobs - minibatch["logprobs"]
    importance_sampling_ratio = logratio.exp()

    # Re-compute advantages with new ratios (V-trace)
    adv = compute_advantage(
        minibatch["values"],
        minibatch["rewards"],
        minibatch["dones"],
        importance_sampling_ratio,
        minibatch["advantages"],
        trainer_cfg.ppo.gamma,
        trainer_cfg.ppo.gae_lambda,
        trainer_cfg.vtrace.vtrace_rho_clip,
        trainer_cfg.vtrace.vtrace_c_clip,
        device,
    )

    # Normalize advantages with distributed support, then apply prioritized weights
    adv = normalize_advantage_distributed(adv, trainer_cfg.ppo.norm_adv)
    adv = prio_weights * adv

    # Compute losses
    pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_ppo_losses(
        minibatch, new_logprobs, entropy, newvalue, importance_sampling_ratio, adv, trainer_cfg,
    )

    # Kickstarter losses
    ks_action_loss, ks_value_loss = kickstarter.loss(
        agent_step, full_logprobs, newvalue, minibatch["obs"], teacher_lstm_state=[]
    )

    # L2 init loss
    l2_init_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if trainer_cfg.ppo.l2_init_loss_coef > 0:
        l2_init_loss = trainer_cfg.ppo.l2_init_loss_coef * policy.l2_init_loss().to(device)

    # Total loss
    loss = (
        pg_loss
        - trainer_cfg.ppo.ent_coef * entropy_loss
        + v_loss * trainer_cfg.ppo.vf_coef
        + l2_init_loss
        + ks_action_loss
        + ks_value_loss
    )

    # Update values and ratio in experience buffer
    update_td = TensorDict(
        {"values": newvalue.view(minibatch["values"].shape), "ratio": importance_sampling_ratio},
        batch_size=minibatch.batch_size,
    )
    experience.update(indices, update_td)

    # Update loss tracking
    losses.policy_loss_sum += pg_loss.item()
    losses.value_loss_sum += v_loss.item()
    losses.entropy_sum += entropy_loss.item()
    losses.approx_kl_sum += approx_kl.item()
    losses.clipfrac_sum += clipfrac.item()
    losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
    losses.ks_action_loss_sum += ks_action_loss.item()
    losses.ks_value_loss_sum += ks_value_loss.item()
    losses.importance_sum += importance_sampling_ratio.mean().item()
    losses.minibatches_processed += 1
    losses.current_logprobs_sum += new_logprobs.mean().item()

    return loss

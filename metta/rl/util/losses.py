"""Loss computation functions for PPO training."""

import logging
from typing import Any, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed

logger = logging.getLogger(__name__)


def compute_ppo_losses(
    minibatch: TensorDict,
    new_logprob: Tensor,
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


def get_loss_experience_spec(act_shape: tuple[int, ...], act_dtype: torch.dtype) -> TensorDict:
    return TensorDict(
        {
            "rewards": torch.zeros((), dtype=torch.float32),
            "dones": torch.zeros((), dtype=torch.float32),
            "truncateds": torch.zeros((), dtype=torch.float32),
            "actions": torch.zeros(act_shape, dtype=act_dtype),
            "act_log_prob": torch.zeros((), dtype=torch.float32),
            "values": torch.zeros((), dtype=torch.float32),
        },
        batch_size=[],
    )


def process_minibatch_update(
    policy: torch.nn.Module,
    experience: Experience,
    minibatch: TensorDict,
    td: TensorDict,
    trainer_cfg: Any,
    indices: Tensor,
    prio_weights: Tensor,
    kickstarter: Any,
    agent_step: int,
    losses: Losses,
    device: torch.device,
) -> Tensor:
    """Process a single minibatch update and return the total loss."""
    td = policy(td, action=minibatch["actions"])
    old_act_log_prob = minibatch["act_log_prob"]
    new_logprob = td["act_log_prob"].reshape(old_act_log_prob.shape)
    entropy = td["entropy"]
    newvalue = td["value"]
    full_logprobs = td["full_log_probs"]

    logratio = new_logprob - old_act_log_prob
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
        td,
        new_logprob,
        entropy,
        newvalue,
        importance_sampling_ratio,
        adv,
        trainer_cfg,
    )

    # Kickstarter losses
    ks_action_loss, ks_value_loss = kickstarter.loss(
        agent_step,
        full_logprobs,
        newvalue,
        td["env_obs"],
        teacher_lstm_state=[],
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
        {"values": newvalue.view(td["values"].shape), "ratio": importance_sampling_ratio},
        batch_size=td.batch_size,
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
    losses.current_logprobs_sum += new_logprob.mean().item()

    return loss

"""PPO training functionality."""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from metta.utils.batch import calculate_prioritized_sampling_params
from metta.rl.advantage import compute_advantage
from metta.rl.losses import Losses, process_minibatch_update


def compute_ppo_losses(
    minibatch: Dict[str, Tensor],
    new_logprobs: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
    device: torch.device,
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


def ppo(
    policy: Any,
    optimizer: Any,
    experience: Any,
    kickstarter: Any,
    losses: Losses,
    trainer_cfg: Any,
    agent_step: int,
    epoch: int,
    device: torch.device,
) -> int:
    """Perform training for one or more epochs on collected experience.

    Returns:
        Number of epochs trained
    """
    losses.zero()
    experience.reset_importance_sampling_ratios()

    # Calculate prioritized sampling parameters
    anneal_beta = calculate_prioritized_sampling_params(
        epoch=epoch,
        total_timesteps=trainer_cfg.total_timesteps,
        batch_size=trainer_cfg.batch_size,
        prio_alpha=trainer_cfg.prioritized_experience_replay.prio_alpha,
        prio_beta0=trainer_cfg.prioritized_experience_replay.prio_beta0,
    )

    # Compute initial advantages
    advantages = torch.zeros(experience.values.shape, device=device)
    initial_importance_sampling_ratio = torch.ones_like(experience.values)

    advantages = compute_advantage(
        experience.values,
        experience.rewards,
        experience.dones,
        initial_importance_sampling_ratio,
        advantages,
        trainer_cfg.ppo.gamma,
        trainer_cfg.ppo.gae_lambda,
        trainer_cfg.vtrace.vtrace_rho_clip,
        trainer_cfg.vtrace.vtrace_c_clip,
        device,
    )

    # Train for multiple epochs
    total_minibatches = experience.num_minibatches * trainer_cfg.update_epochs
    minibatch_idx = 0
    epochs_trained = 0

    for _update_epoch in range(trainer_cfg.update_epochs):
        for _ in range(experience.num_minibatches):
            # Sample minibatch
            minibatch = experience.sample_minibatch(
                advantages=advantages,
                prio_alpha=trainer_cfg.prioritized_experience_replay.prio_alpha,
                prio_beta=anneal_beta,
                minibatch_idx=minibatch_idx,
                total_minibatches=total_minibatches,
            )

            # Process minibatch
            loss = process_minibatch_update(
                policy=policy,
                experience=experience,
                minibatch=minibatch,
                advantages=advantages,
                trainer_cfg=trainer_cfg,
                kickstarter=kickstarter,
                agent_step=agent_step,
                losses=losses,
                device=device,
            )

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()

            if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer_cfg.ppo.max_grad_norm)
                optimizer.step()

                # Optional weight clipping
                if hasattr(policy, "clip_weights"):
                    policy.clip_weights()

                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()

            minibatch_idx += 1

        epochs_trained += 1

        # Early exit if KL divergence is too high
        if trainer_cfg.ppo.target_kl is not None:
            average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
            if average_approx_kl > trainer_cfg.ppo.target_kl:
                break

    # Calculate explained variance
    y_pred = experience.values.flatten()
    y_true = advantages.flatten() + experience.values.flatten()
    var_y = y_true.var()
    losses.explained_variance = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0

    return epochs_trained

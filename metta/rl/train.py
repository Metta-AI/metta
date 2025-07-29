"""PPO training functionality."""

from typing import Any

import torch

from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage
from metta.rl.util.batch_utils import calculate_prioritized_sampling_params
from metta.rl.util.losses import process_minibatch_update


def train_ppo(
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

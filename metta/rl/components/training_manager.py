"""Manages the training phase of RL training."""

import logging
from typing import Any, Optional

import torch

from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_config import TrainerConfig
from metta.rl.util.advantage import compute_advantage
from metta.rl.util.batch_utils import calculate_prioritized_sampling_params
from metta.rl.util.losses import process_minibatch_update
from metta.rl.util.optimization import calculate_explained_variance

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages PPO training updates on collected experience."""

    def __init__(
        self,
        trainer_config: TrainerConfig,
        device: torch.device,
        kickstarter: Optional[Kickstarter] = None,
    ):
        """Initialize training manager.

        Args:
            trainer_config: Training configuration
            device: Device to run computations on
            kickstarter: Optional kickstarter for teacher-student learning
        """
        self.trainer_config = trainer_config
        self.device = device
        self.kickstarter = kickstarter

    def train_on_experience(
        self,
        agent: Any,
        optimizer: Any,
        experience: Experience,
        losses: Losses,
        epoch: int,
        agent_step: int,
    ) -> None:
        """Run PPO training on collected experience.

        Args:
            agent: The policy/agent to train
            optimizer: Optimizer for updating agent parameters
            experience: Experience buffer containing trajectories
            losses: Loss tracker object
            epoch: Current training epoch
            agent_step: Current training step
        """
        # Reset losses and experience for training
        losses.zero()
        experience.reset_importance_sampling_ratios()

        # Calculate prioritized replay parameters
        prio_cfg = self.trainer_config.prioritized_experience_replay
        anneal_beta = calculate_prioritized_sampling_params(
            epoch=epoch,
            total_timesteps=self.trainer_config.total_timesteps,
            batch_size=self.trainer_config.batch_size,
            prio_alpha=prio_cfg.prio_alpha,
            prio_beta0=prio_cfg.prio_beta0,
        )

        # Compute advantages
        advantages = torch.zeros(experience.values.shape, device=self.device)
        initial_importance_sampling_ratio = torch.ones_like(experience.values)

        advantages = compute_advantage(
            experience.values,
            experience.rewards,
            experience.dones,
            initial_importance_sampling_ratio,
            advantages,
            self.trainer_config.ppo.gamma,
            self.trainer_config.ppo.gae_lambda,
            self.trainer_config.vtrace.vtrace_rho_clip,
            self.trainer_config.vtrace.vtrace_c_clip,
            self.device,
        )

        # Train for multiple epochs
        total_minibatches = experience.num_minibatches * self.trainer_config.update_epochs
        minibatch_idx = 0

        for _update_epoch in range(self.trainer_config.update_epochs):
            for _ in range(experience.num_minibatches):
                # Sample minibatch
                minibatch = experience.sample_minibatch(
                    advantages=advantages,
                    prio_alpha=prio_cfg.prio_alpha,
                    prio_beta=anneal_beta,
                    minibatch_idx=minibatch_idx,
                    total_minibatches=total_minibatches,
                )

                # Train on minibatch
                loss = process_minibatch_update(
                    policy=agent,
                    experience=experience,
                    minibatch=minibatch,
                    advantages=advantages,
                    trainer_cfg=self.trainer_config,
                    kickstarter=self.kickstarter,
                    agent_step=agent_step,
                    losses=losses,
                    device=self.device,
                )

                optimizer.step(loss, epoch, experience.accumulate_minibatches)
                minibatch_idx += 1

            # Early exit if KL divergence is too high
            if self.trainer_config.ppo.target_kl is not None:
                average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
                if average_approx_kl > self.trainer_config.ppo.target_kl:
                    logger.info(f"Early stopping at epoch {_update_epoch} due to KL divergence")
                    break

        # Synchronize CUDA if needed
        if minibatch_idx > 0 and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

        # Calculate explained variance
        losses.explained_variance = calculate_explained_variance(experience.values, advantages)

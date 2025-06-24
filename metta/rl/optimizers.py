"""Optimization components for RL training."""

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from metta.agent.policy_state import PolicyState
from metta.rl.configs import PPOConfig
from metta.rl.experience import Experience
from metta.rl.losses import Losses

if TYPE_CHECKING:
    from metta.agent import BaseAgent

logger = logging.getLogger(__name__)


class PPOOptimizer:
    """PPO optimizer with clean interface for custom losses."""

    def __init__(
        self,
        policy: "BaseAgent",
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Optional[PPOConfig] = None,
    ):
        """Initialize PPO optimizer.

        Args:
            policy: Policy network to optimize
            optimizer: PyTorch optimizer
            device: Device for computations
            config: PPO configuration (uses defaults if None)
        """
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.config = config or PPOConfig()
        self.losses = Losses()

    def compute_ppo_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute core PPO losses.

        Returns dict with 'policy_loss', 'value_loss', 'entropy_loss', and 'total_loss'.
        """
        # Forward pass
        lstm_state = PolicyState()
        _, new_logprobs, entropy, newvalue, _ = self.policy(obs, lstm_state, action=actions)

        # Importance sampling ratio
        new_logprobs = new_logprobs.reshape(old_logprobs.shape)
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()

        # Normalize advantages
        if self.config.norm_adv:
            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            adv = advantages

        # Policy loss (clipped surrogate objective)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(returns.shape)
        if self.config.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(
                newvalue - values,
                -self.config.vf_clip_coef,
                self.config.vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total PPO loss
        total_loss = policy_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * value_loss

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss,
            "approx_kl": ((ratio - 1) - logratio).mean(),
            "clipfrac": ((ratio - 1.0).abs() > self.config.clip_coef).float().mean(),
        }

    def update(
        self,
        experience: Experience,
        update_epochs: Optional[int] = None,
        custom_loss_fns: Optional[List[Callable]] = None,
    ) -> Dict[str, float]:
        """Perform PPO update with optional custom losses.

        Args:
            experience: Experience buffer with collected data
            update_epochs: Number of epochs (uses config default if None)
            custom_loss_fns: Optional list of custom loss functions

        Returns:
            Dictionary of loss statistics
        """
        self.losses.zero()
        update_epochs = update_epochs or self.config.update_epochs

        # Compute advantages
        advantages = self._compute_advantages(
            experience.values,
            experience.rewards,
            experience.dones,
        )

        # Training loop
        for epoch in range(update_epochs):
            for minibatch in self._iter_minibatches(experience, advantages):
                # Compute PPO losses
                ppo_losses = self.compute_ppo_loss(
                    obs=minibatch["obs"],
                    actions=minibatch["actions"],
                    old_logprobs=minibatch["logprobs"],
                    advantages=minibatch["advantages"],
                    returns=minibatch["returns"],
                    values=minibatch["values"],
                )

                total_loss = ppo_losses["total_loss"]

                # Add custom losses
                if custom_loss_fns:
                    for loss_fn in custom_loss_fns:
                        custom_loss = loss_fn(
                            policy=self.policy,
                            obs=minibatch["obs"],
                            actions=minibatch["actions"],
                            rewards=minibatch["rewards"],
                            values=minibatch["values"],
                            advantages=minibatch["advantages"],
                        )
                        total_loss = total_loss + custom_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Update loss tracking
                self._update_losses(ppo_losses)

            # Early stopping based on KL
            if self.config.target_kl is not None:
                avg_kl = self.losses.approx_kl_sum / max(1, self.losses.minibatches_processed)
                if avg_kl > self.config.target_kl:
                    break

        return self.losses.stats()

    def _compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GAE advantages."""
        # Simplified version - in practice uses CUDA kernel
        advantages = torch.zeros_like(values)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t]
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            )

        return advantages

    def _iter_minibatches(self, experience: Experience, advantages: torch.Tensor):
        """Iterate over minibatches."""
        # Simplified version
        batch_size = len(experience.obs)
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, self.config.minibatch_size):
            end = min(start + self.config.minibatch_size, batch_size)
            mb_indices = indices[start:end]

            yield {
                "obs": experience.obs[mb_indices],
                "actions": experience.actions[mb_indices],
                "logprobs": experience.logprobs[mb_indices],
                "values": experience.values[mb_indices],
                "advantages": advantages[mb_indices],
                "returns": advantages[mb_indices] + experience.values[mb_indices],
                "rewards": experience.rewards[mb_indices],
            }

    def _update_losses(self, loss_dict: Dict[str, torch.Tensor]):
        """Update loss tracking."""
        self.losses.policy_loss_sum += loss_dict["policy_loss"].item()
        self.losses.value_loss_sum += loss_dict["value_loss"].item()
        self.losses.entropy_sum += loss_dict["entropy_loss"].item()
        self.losses.approx_kl_sum += loss_dict["approx_kl"].item()
        self.losses.clipfrac_sum += loss_dict["clipfrac"].item()
        self.losses.minibatches_processed += 1

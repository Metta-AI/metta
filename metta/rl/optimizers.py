"""Optimization components for RL training."""

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from metta.agent.policy_state import PolicyState
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses

if TYPE_CHECKING:
    from metta.agent import BaseAgent

logger = logging.getLogger(__name__)


class PPOOptimizer:
    """Proximal Policy Optimization (PPO) trainer.

    This class encapsulates the PPO algorithm logic, handling
    advantage computation, loss calculation, and optimization steps.
    """

    def __init__(
        self,
        policy: "BaseAgent",
        optimizer: Optimizer,
        device: torch.device,
        # PPO hyperparameters
        clip_coef: float = 0.1,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        vf_clip_coef: float = 0.1,
        target_kl: Optional[float] = None,
        # Additional features
        l2_reg_loss_coef: float = 0.0,
        l2_init_loss_coef: float = 0.0,
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
    ):
        """Initialize PPO optimizer.

        Args:
            policy: Policy network to optimize
            optimizer: PyTorch optimizer
            device: Device for computations
            clip_coef: PPO clipping coefficient
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            norm_adv: Whether to normalize advantages
            clip_vloss: Whether to clip value loss
            vf_clip_coef: Value function clipping coefficient
            target_kl: Target KL divergence for early stopping
            l2_reg_loss_coef: L2 regularization coefficient
            l2_init_loss_coef: L2 initialization loss coefficient
            vtrace_rho_clip: V-trace rho clipping parameter
            vtrace_c_clip: V-trace c clipping parameter
        """
        self.policy = policy
        self.optimizer = optimizer
        self.device = device

        # PPO hyperparameters
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.vf_clip_coef = vf_clip_coef
        self.target_kl = target_kl

        # Regularization
        self.l2_reg_loss_coef = l2_reg_loss_coef
        self.l2_init_loss_coef = l2_init_loss_coef

        # V-trace parameters
        self.vtrace_rho_clip = vtrace_rho_clip
        self.vtrace_c_clip = vtrace_c_clip

        # Loss tracking
        self.losses = Losses()

    def update(
        self,
        experience: Experience,
        update_epochs: int = 4,
        minibatch_size: Optional[int] = None,
        kickstarter: Optional[Kickstarter] = None,
        prioritized_replay_alpha: float = 0.0,
        prioritized_replay_beta: float = 0.6,
    ) -> Dict[str, float]:
        """Perform PPO update on collected experience.

        Args:
            experience: Experience buffer with collected data
            update_epochs: Number of epochs to train on the data
            minibatch_size: Size of minibatches (if None, uses experience default)
            kickstarter: Optional kickstarter for auxiliary losses
            prioritized_replay_alpha: PER alpha parameter
            prioritized_replay_beta: PER beta parameter

        Returns:
            Dictionary of loss statistics
        """
        self.losses.zero()

        # Reset importance sampling ratios
        experience.reset_importance_sampling_ratios()

        # Compute initial advantages
        advantages = self._compute_advantages(
            experience.values,
            experience.rewards,
            experience.dones,
            torch.ones_like(experience.values),
        )

        # Training loop
        total_minibatches = experience.num_minibatches * update_epochs
        minibatch_idx = 0

        for epoch in range(update_epochs):
            for _ in range(experience.num_minibatches):
                # Sample minibatch
                minibatch = experience.sample_minibatch(
                    advantages=advantages,
                    prio_alpha=prioritized_replay_alpha,
                    prio_beta=prioritized_replay_beta,
                    minibatch_idx=minibatch_idx,
                    total_minibatches=total_minibatches,
                )

                # Forward pass with current policy
                lstm_state = PolicyState()
                _, new_logprobs, entropy, newvalue, full_logprobs = self.policy(
                    minibatch["obs"], lstm_state, action=minibatch["actions"]
                )

                # Compute importance sampling ratio
                new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
                logratio = new_logprobs - minibatch["logprobs"]
                importance_sampling_ratio = logratio.exp()
                experience.update_ratio(minibatch["indices"], importance_sampling_ratio)

                # Track KL divergence
                with torch.no_grad():
                    approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
                    clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.clip_coef).float().mean()

                # Re-compute advantages with V-trace
                adv = self._compute_advantages(
                    minibatch["values"],
                    minibatch["rewards"],
                    minibatch["dones"],
                    importance_sampling_ratio,
                    base_advantages=minibatch["advantages"],
                )

                # Normalize advantages
                if self.norm_adv:
                    adv = self._normalize_advantages(adv)
                adv = minibatch["prio_weights"] * adv

                # Policy loss
                pg_loss1 = -adv * importance_sampling_ratio
                pg_loss2 = -adv * torch.clamp(importance_sampling_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
                    v_clipped = minibatch["values"] + torch.clamp(
                        newvalue_reshaped - minibatch["values"],
                        -self.vf_clip_coef,
                        self.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

                entropy_loss = entropy.mean()

                # Optional kickstarter losses
                ks_action_loss = torch.tensor(0.0, device=self.device)
                ks_value_loss = torch.tensor(0.0, device=self.device)
                if kickstarter is not None:
                    ks_action_loss, ks_value_loss = kickstarter.loss(
                        0,  # agent_step placeholder
                        full_logprobs,
                        newvalue,
                        minibatch["obs"],
                        teacher_lstm_state=[],
                    )

                # Regularization losses
                l2_reg_loss = torch.tensor(0.0, device=self.device)
                if self.l2_reg_loss_coef > 0:
                    l2_reg_loss = self.l2_reg_loss_coef * self.policy.l2_reg_loss()

                l2_init_loss = torch.tensor(0.0, device=self.device)
                if self.l2_init_loss_coef > 0:
                    l2_init_loss = self.l2_init_loss_coef * self.policy.l2_init_loss()

                # Total loss
                loss = (
                    pg_loss
                    - self.ent_coef * entropy_loss
                    + v_loss * self.vf_coef
                    + l2_reg_loss
                    + l2_init_loss
                    + ks_action_loss
                    + ks_value_loss
                )

                # Update values in experience buffer
                experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

                # Update loss tracking
                self._update_losses(
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    clipfrac,
                    l2_reg_loss,
                    l2_init_loss,
                    ks_action_loss,
                    ks_value_loss,
                    importance_sampling_ratio.mean(),
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()

                if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    if hasattr(self.policy, "clip_weights"):
                        self.policy.clip_weights()

                    if str(self.device).startswith("cuda"):
                        torch.cuda.synchronize()

                minibatch_idx += 1

            # Early stopping based on KL divergence
            if self.target_kl is not None:
                average_kl = self.losses.approx_kl_sum / self.losses.minibatches_processed
                if average_kl > self.target_kl:
                    break

        # Calculate explained variance
        explained_var = self._calculate_explained_variance(experience.values, advantages)
        self.losses.explained_variance = explained_var

        return self.losses.stats()

    def _compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        importance_sampling_ratio: torch.Tensor,
        base_advantages: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute advantages using CUDA kernel with CPU fallback."""
        device = self.device

        # Prepare tensors
        if base_advantages is None:
            advantages = torch.zeros_like(values)
        else:
            advantages = base_advantages.clone()

        tensors = [values, rewards, dones, importance_sampling_ratio, advantages]
        tensors = [t.to(device) for t in tensors]
        values, rewards, dones, importance_sampling_ratio, advantages = tensors

        # Use CUDA kernel or CPU implementation
        device_context = torch.cuda.device(device) if str(device).startswith("cuda") else nullcontext()
        with device_context:
            torch.ops.pufferlib.compute_puff_advantage(
                values,
                rewards,
                dones,
                importance_sampling_ratio,
                advantages,
                self.gamma,
                self.gae_lambda,
                self.vtrace_rho_clip,
                self.vtrace_c_clip,
            )

        return advantages

    def _normalize_advantages(self, adv: torch.Tensor) -> torch.Tensor:
        """Normalize advantages with distributed training support."""
        if torch.distributed.is_initialized():
            # Distributed normalization
            import einops

            adv_flat = adv.view(-1)
            local_sum = einops.rearrange(adv_flat.sum(), "-> 1")
            local_sq_sum = einops.rearrange((adv_flat * adv_flat).sum(), "-> 1")
            local_count = torch.tensor([adv_flat.numel()], dtype=adv.dtype, device=adv.device)

            stats = einops.rearrange([local_sum, local_sq_sum, local_count], "one float -> (float one)")
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

            global_sum, global_sq_sum, global_count = stats[0], stats[1], stats[2]
            global_mean = global_sum / global_count
            global_var = (global_sq_sum / global_count) - (global_mean * global_mean)
            global_std = torch.sqrt(global_var.clamp(min=1e-8))

            adv = (adv - global_mean) / (global_std + 1e-8)
        else:
            # Local normalization
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return adv

    def _calculate_explained_variance(
        self,
        values: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:
        """Calculate explained variance metric."""
        y_pred = values.flatten()
        y_true = advantages.flatten() + values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        return explained_var.item() if torch.is_tensor(explained_var) else float("nan")

    def _update_losses(self, *loss_values):
        """Update loss tracking."""
        self.losses.policy_loss_sum += loss_values[0].item()
        self.losses.value_loss_sum += loss_values[1].item()
        self.losses.entropy_sum += loss_values[2].item()
        self.losses.approx_kl_sum += loss_values[3].item()
        self.losses.clipfrac_sum += loss_values[4].item()
        self.losses.l2_reg_loss_sum += loss_values[5].item() if torch.is_tensor(loss_values[5]) else loss_values[5]
        self.losses.l2_init_loss_sum += loss_values[6].item() if torch.is_tensor(loss_values[6]) else loss_values[6]
        self.losses.ks_action_loss_sum += loss_values[7].item()
        self.losses.ks_value_loss_sum += loss_values[8].item()
        self.losses.importance_sum += loss_values[9].item()
        self.losses.minibatches_processed += 1

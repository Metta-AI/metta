"""Latent-variable dynamics model loss for model-based RL."""

from typing import Any

import einops
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext
from mettagrid.base_config import Config

try:
    from metta.agent.components.dynamics.triton_kernels import compute_kl_divergence
except ImportError:
    # Fallback if triton_kernels not available
    compute_kl_divergence = None


class LatentDynamicsLossConfig(Config):
    """Configuration for latent dynamics loss.

    This loss trains a latent-variable dynamics model using:
    - Reconstruction loss (next state prediction)
    - KL divergence (regularization to prior)
    - Auxiliary loss (long-term future prediction)
    """

    # Loss weights
    beta_kl: float = Field(default=0.01, ge=0, le=1.0, description="KL divergence weight")
    gamma_auxiliary: float = Field(default=1.0, ge=0, le=10.0, description="Auxiliary task weight")

    # Future prediction
    future_horizon: int = Field(default=5, ge=1, le=100, description="Steps ahead to predict")
    future_type: str = Field(
        default="returns", description="Type of future to predict: returns, rewards, or observations"
    )

    # Training
    use_auxiliary: bool = Field(default=True, description="Whether to use auxiliary task")
    reconstruction_coef: float = Field(default=1.0, ge=0, le=1.0, description="Reconstruction loss weight")

    # Performance
    use_triton: bool = Field(default=True, description="Use Triton kernels if available")

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create LatentDynamicsLoss instance."""
        return LatentDynamicsLoss(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_cfg=loss_config,
        )


class LatentDynamicsLoss(Loss):
    """Loss function for latent-variable dynamics model.

    Trains the model to:
    1. Reconstruct next observations from latent variables
    2. Maintain informative latent distributions (KL regularization)
    3. Predict long-term future information (auxiliary task)
    """

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute latent dynamics loss.

        Args:
            shared_loss_data: Dictionary containing policy outputs and batch data
            context: Training context
            mb_idx: Minibatch index

        Returns:
            total_loss: Combined loss value
            shared_loss_data: Updated loss data
            skip_backward: Whether to skip backward pass
        """
        policy_td = shared_loss_data["policy_td"]

        # Check if latent dynamics outputs are present
        if "latent_mean" not in policy_td or "latent_logvar" not in policy_td:
            # Model not configured with latent dynamics
            return torch.tensor(0.0, device=context.device), shared_loss_data, True

        # Get latent distribution parameters
        z_mean = policy_td["latent_mean"].to(dtype=torch.float32)
        z_logvar = policy_td["latent_logvar"].to(dtype=torch.float32)

        # 1. KL Divergence Loss (regularization to N(0,1) prior)
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Use Triton kernel if available for better performance
        if self.loss_cfg.use_triton and compute_kl_divergence is not None:
            kl_loss = compute_kl_divergence(z_mean, z_logvar, use_triton=True)
        else:
            kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
            kl_loss = kl_loss.mean()
        kl_loss = kl_loss * self.loss_cfg.beta_kl

        # 2. Reconstruction Loss (next state prediction)
        recon_loss = torch.tensor(0.0, device=context.device)
        if "obs_next_pred" in policy_td:
            obs_next_pred = policy_td["obs_next_pred"].to(dtype=torch.float32)
            sampled_mb = shared_loss_data["sampled_mb"]

            # Get actual next observations
            # Assuming observations are stored with proper time indexing
            if "obs_next" in sampled_mb:
                obs_next = sampled_mb["obs_next"].to(dtype=torch.float32)
                recon_loss = F.mse_loss(obs_next_pred, obs_next) * self.loss_cfg.reconstruction_coef

        # 3. Auxiliary Loss (long-term future prediction)
        aux_loss = torch.tensor(0.0, device=context.device)
        if self.loss_cfg.use_auxiliary and "future_pred" in policy_td:
            future_pred = policy_td["future_pred"].to(dtype=torch.float32)
            sampled_mb = shared_loss_data["sampled_mb"]

            # Get future target based on future_type
            if self.loss_cfg.future_type == "returns":
                if "returns" in sampled_mb:
                    returns = sampled_mb["returns"].to(dtype=torch.float32)
                    # Predict returns future_horizon steps ahead
                    look_ahead = self.loss_cfg.future_horizon
                    if len(returns) > look_ahead:
                        future_returns = returns[look_ahead:]
                        future_pred_aligned = future_pred[:-look_ahead]
                        # Reshape if needed using einops
                        if future_pred_aligned.shape != future_returns.shape:
                            if future_pred_aligned.dim() > future_returns.dim():
                                future_pred_aligned = einops.rearrange(future_pred_aligned, "... 1 -> ...")
                        aux_loss = F.mse_loss(future_pred_aligned, future_returns) * self.loss_cfg.gamma_auxiliary

            elif self.loss_cfg.future_type == "rewards":
                if "rewards" in sampled_mb:
                    rewards = sampled_mb["rewards"].to(dtype=torch.float32)
                    # Predict reward at next timestep
                    if len(rewards) > 1:
                        future_rewards = rewards[1:]
                        future_pred_aligned = future_pred[:-1]
                        # Reshape if needed using einops
                        if future_pred_aligned.shape != future_rewards.shape:
                            if future_pred_aligned.dim() > future_rewards.dim():
                                future_pred_aligned = einops.rearrange(future_pred_aligned, "... 1 -> ...")
                        aux_loss = F.mse_loss(future_pred_aligned, future_rewards) * self.loss_cfg.gamma_auxiliary

        # Track individual losses
        self.loss_tracker["latent_dynamics_kl_loss"].append(float(kl_loss.item()))
        self.loss_tracker["latent_dynamics_recon_loss"].append(float(recon_loss.item()))
        self.loss_tracker["latent_dynamics_aux_loss"].append(float(aux_loss.item()))

        # Total loss
        total_loss = kl_loss + recon_loss + aux_loss

        return total_loss, shared_loss_data, False

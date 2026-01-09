from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class FutureLatentEMALossConfig(LossConfig):
    """Configuration for the EMA-based future latent state prediction loss."""

    ema_decay: float = Field(default=0.9, ge=0.0, lt=1.0, description="Exponential moving average decay factor.")
    prediction_horizon: int = Field(default=4, ge=1, description="Number of future steps to include in the EMA target.")
    loss_coef: float = Field(default=1.0, ge=0.0, description="Multiplier applied to the computed loss value.")

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "FutureLatentEMALoss":
        """Factory hook invoked by the loss registry."""
        return FutureLatentEMALoss(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            cfg=self,
        )


class FutureLatentEMALoss(Loss):
    """Encourages policies to predict an EMA of future latent states."""

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"future_latent_pred", "core"}

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        policy_td = shared_loss_data.get("policy_td")
        if policy_td is None:
            return self._zero(), shared_loss_data, False

        if "future_latent_pred" not in policy_td.keys():
            return self._zero(), shared_loss_data, False
        if "core" not in policy_td.keys():
            return self._zero(), shared_loss_data, False

        future_pred: Tensor = policy_td["future_latent_pred"].to(dtype=torch.float32)
        latent_core: Tensor = policy_td["core"].to(dtype=torch.float32)

        if future_pred.dim() != 3 or latent_core.dim() != 3:
            # Expected shape: (batch, time, hidden)
            return self._zero(), shared_loss_data, False

        batch_size, time_steps, hidden_dim = latent_core.shape
        if time_steps < 2:
            return self._zero(), shared_loss_data, False

        horizon = min(self.cfg.prediction_horizon, time_steps - 1)
        if horizon < 1:
            return self._zero(), shared_loss_data, False

        ema_weights = (1.0 - self.cfg.ema_decay) * torch.pow(
            torch.full((horizon,), self.cfg.ema_decay, device=latent_core.device, dtype=latent_core.dtype),
            torch.arange(horizon, device=latent_core.device, dtype=latent_core.dtype),
        )

        # Extract future latent slices (skip the current timestep).
        future_latents = latent_core[:, 1:, :]
        if future_latents.size(1) < horizon:
            return self._zero(), shared_loss_data, False

        conv_input = future_latents.transpose(1, 2)
        conv_weight = ema_weights.view(1, 1, horizon).expand(hidden_dim, -1, -1).contiguous()
        ema_targets = F.conv1d(conv_input, conv_weight, stride=1, padding=0, groups=hidden_dim)
        ema_targets = ema_targets.transpose(1, 2)

        normalisation = 1.0 - self.cfg.ema_decay**horizon
        if normalisation > 0:
            ema_targets = ema_targets / normalisation

        ema_targets = ema_targets.detach()

        target_time_len = ema_targets.size(1)
        aligned_predictions = future_pred[:, :target_time_len, :]

        loss = F.mse_loss(aligned_predictions, ema_targets) * self.cfg.loss_coef
        self.loss_tracker["future_latent_ema_mse"].append(float(loss.item()))
        return loss, shared_loss_data, False

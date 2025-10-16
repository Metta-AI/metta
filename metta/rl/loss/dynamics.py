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


class DynamicsConfig(Config):
    returns_step_look_ahead: int = Field(default=1)
    returns_pred_coef: float = Field(default=1.0, ge=0, le=1.0)
    reward_pred_coef: float = Field(default=1.0, ge=0, le=1.0)
    value_pred_coef: float = Field(default=1.0, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create Dynamics loss instance."""
        return Dynamics(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_cfg=loss_config,
        )


class Dynamics(Loss):
    """The dynamics term in the Muesli loss."""

    # Loss calls this method
    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        policy_td = shared_loss_data["policy_td"]

        returns_pred: Tensor = policy_td["returns_pred"].to(dtype=torch.float32)
        reward_pred: Tensor = policy_td["reward_pred"].to(dtype=torch.float32)
        value_pred: Tensor = policy_td["value_pred"].to(dtype=torch.float32)

        # need to reshape from (B, T, 1) to (B, T)
        returns_pred = einops.rearrange(returns_pred, "b t 1 -> b (t 1)")
        reward_pred = einops.rearrange(reward_pred, "b t 1 -> b (t 1)")
        value_pred = einops.rearrange(value_pred, "b t 1 -> b (t 1)")

        # targets
        returns = shared_loss_data["sampled_mb"]["returns"]
        rewards = shared_loss_data["sampled_mb"]["rewards"]
        baseline_values = shared_loss_data["sampled_mb"]["values"].to(dtype=torch.float32)

        # The model predicts future returns and rewards.
        # We align the predictions with the future targets by slicing the tensors.
        look_ahead = self.loss_cfg.returns_step_look_ahead

        # Predict returns `look_ahead` steps into the future.
        future_returns = returns[look_ahead:]
        returns_pred_aligned = returns_pred[:-look_ahead]
        returns_loss = F.mse_loss(returns_pred_aligned, future_returns) * self.loss_cfg.returns_pred_coef

        # Predict reward at the next timestep.
        future_rewards = rewards[1:]
        reward_pred_aligned = reward_pred[:-1]
        reward_loss = F.mse_loss(reward_pred_aligned, future_rewards) * self.loss_cfg.reward_pred_coef

        # Predict value baseline at the next timestep.
        future_values = baseline_values[1:]
        value_pred_aligned = value_pred[:-1]
        if value_pred_aligned.numel() > 0:
            value_loss = F.mse_loss(value_pred_aligned, future_values) * self.loss_cfg.value_pred_coef
        else:
            value_loss = torch.zeros((), device=value_pred.device)

        self.loss_tracker["dynamics_returns_loss"].append(float(returns_loss.item()))
        self.loss_tracker["dynamics_reward_loss"].append(float(reward_loss.item()))
        self.loss_tracker["dynamics_value_loss"].append(float(value_loss.item()))

        return returns_loss + reward_loss + value_loss, shared_loss_data, False

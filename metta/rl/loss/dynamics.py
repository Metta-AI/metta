from typing import Any, Optional

import einops
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext


class DynamicsConfig(LossConfig):
    returns_step_look_ahead: int = Field(default=1)
    returns_pred_coef: float = Field(default=1.0, ge=0, le=1.0)
    reward_pred_coef: float = Field(default=1.0, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "Dynamics":
        """Create Dynamics loss instance."""
        return Dynamics(policy, trainer_cfg, vec_env, device, instance_name, self)


class Dynamics(Loss):
    """The dynamics term in the Muesli loss."""

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"returns_pred", "reward_pred"}

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

        # need to reshape from (B, T, 1) to (B, T)
        returns_pred = einops.rearrange(returns_pred, "b t 1 -> b (t 1)")
        reward_pred = einops.rearrange(reward_pred, "b t 1 -> b (t 1)")

        # targets
        returns = shared_loss_data["sampled_mb"]["returns"]
        rewards = shared_loss_data["sampled_mb"]["rewards"]

        # The model predicts future returns and rewards.
        # We align the predictions with the future targets by slicing the tensors.
        look_ahead = self.cfg.returns_step_look_ahead

        # Predict returns `look_ahead` steps into the future.
        future_returns = returns[look_ahead:]
        returns_pred_aligned = returns_pred[:-look_ahead]
        returns_loss = F.mse_loss(returns_pred_aligned, future_returns) * self.cfg.returns_pred_coef

        # Predict reward at the next timestep.
        future_rewards = rewards[1:]
        reward_pred_aligned = reward_pred[:-1]
        reward_loss = F.mse_loss(reward_pred_aligned, future_rewards) * self.cfg.reward_pred_coef

        self.track_metric("dynamics_returns_loss", returns_loss)
        self.track_metric("dynamics_reward_loss", reward_loss)

        return returns_loss + reward_loss, shared_loss_data, False

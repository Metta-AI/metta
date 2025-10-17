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
    action_pred_coef: float = Field(default=1.0, ge=0)

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
        action_pred_logits: Tensor | None = policy_td.get("action_pred_logits")

        # need to reshape from (B, T, 1) to (B, T)
        returns_pred = einops.rearrange(returns_pred, "b t 1 -> b (t 1)")
        reward_pred = einops.rearrange(reward_pred, "b t 1 -> b (t 1)")

        # targets
        returns = shared_loss_data["sampled_mb"]["returns"]
        rewards = shared_loss_data["sampled_mb"]["rewards"]

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

        self.loss_tracker["dynamics_returns_loss"].append(float(returns_loss.item()))
        self.loss_tracker["dynamics_reward_loss"].append(float(reward_loss.item()))

        action_loss = torch.tensor(0.0, device=self.device)
        if action_pred_logits is not None and self.loss_cfg.action_pred_coef > 0:
            action_loss = self._compute_action_loss(action_pred_logits, shared_loss_data)
            self.loss_tracker.setdefault("dynamics_action_loss", []).append(float(action_loss.item()))

        total_loss = returns_loss + reward_loss + action_loss
        return total_loss, shared_loss_data, False

    def _compute_action_loss(self, action_pred_logits: Tensor, shared_loss_data: TensorDict) -> Tensor:
        logits = action_pred_logits.to(dtype=torch.float32)
        actions = shared_loss_data["sampled_mb"]["actions"].to(dtype=torch.long)

        if logits.dim() != 3:
            raise ValueError("action_pred_logits must have shape [segments, horizon, num_actions]")

        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        elif actions.dim() != 2:
            raise ValueError("actions must have shape [segments, horizon] or [segments, horizon, 1]")

        segments, horizon, num_actions = logits.shape
        if actions.shape[0] != segments or actions.shape[1] != horizon:
            raise ValueError("actions and action_pred_logits must share batch shape")

        # Predict next action -> align predictions with future actions
        if horizon <= 1:
            return torch.tensor(0.0, device=self.device)

        next_actions = actions[:, 1:]
        pred_logits = logits[:, :-1]

        loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, num_actions),
            next_actions.reshape(-1),
        )
        return loss * self.loss_cfg.action_pred_coef

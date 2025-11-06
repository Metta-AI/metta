import typing

import einops
import pydantic
import tensordict
import torch
import torch.nn.functional as F

import metta.agent.policy
import metta.rl.loss.loss
import metta.rl.training.component_context as training_component_context
import mettagrid.base_config


class DynamicsConfig(mettagrid.base_config.Config):
    returns_step_look_ahead: int = pydantic.Field(default=1)
    returns_pred_coef: float = pydantic.Field(default=1.0, ge=0, le=1.0)
    reward_pred_coef: float = pydantic.Field(default=1.0, ge=0, le=1.0)

    def create(
        self,
        policy: metta.agent.policy.Policy,
        trainer_cfg: typing.Any,
        vec_env: typing.Any,
        device: torch.device,
        instance_name: str,
        loss_config: typing.Any,
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


class Dynamics(metta.rl.loss.loss.Loss):
    """The dynamics term in the Muesli loss."""

    # Loss calls this method
    def run_train(
        self,
        shared_loss_data: tensordict.TensorDict,
        context: training_component_context.ComponentContext,
        mb_idx: int,
    ) -> tuple[torch.Tensor, tensordict.TensorDict, bool]:
        policy_td = shared_loss_data["policy_td"]

        returns_pred: torch.Tensor = policy_td["returns_pred"].to(dtype=torch.float32)
        reward_pred: torch.Tensor = policy_td["reward_pred"].to(dtype=torch.float32)

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

        return returns_loss + reward_loss, shared_loss_data, False

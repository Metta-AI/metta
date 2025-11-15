from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment
from mettagrid.base_config import Composite, UnboundedContinuous


class PPOValueConfig(LossConfig):
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value term weight from sweep
    vf_coef: float = Field(default=0.897619, ge=0)
    # Value loss clipping toggle
    clip_vloss: bool = True

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "PPOValue":
        """Points to the PPO class for initialization."""
        return PPOValue(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class PPOValue(Loss):
    """PPO value loss."""

    __slots__ = ()

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_config)

    def get_experience_spec(self) -> Composite:
        return Composite(
            values=UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32),
        )

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        old_values = minibatch["values"]
        returns = minibatch["returns"]
        # returns = shared_loss_data["PPOActor"]["advantages"] + old_values # av fresher update?
        policy_td = shared_loss_data.get("policy_td", None)
        if policy_td is not None:
            newvalue = policy_td["values"]
            newvalue_reshaped = newvalue.view(returns.shape)

        if self.cfg.clip_vloss and newvalue_reshaped is not None:
            v_loss_unclipped = (newvalue_reshaped - returns) ** 2
            vf_clip_coef = self.cfg.vf_clip_coef
            v_clipped = old_values + torch.clamp(
                newvalue_reshaped - old_values,
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((old_values - returns) ** 2).mean()

        return v_loss, shared_loss_data, False

    def on_train_phase_end(self, context: ComponentContext) -> None:
        """Compute value-function explained variance for logging, mirroring monolithic PPO."""
        if self.replay is None:
            return
        with torch.no_grad():
            y_pred = self.replay.buffer["values"].flatten()
            y_true = self.advantages.flatten() + self.replay.buffer["values"].flatten()
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

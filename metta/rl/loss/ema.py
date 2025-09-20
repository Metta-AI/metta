import copy
from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F

from metta.agent.metta_agent import PolicyAgent
from metta.rl.loss.loss import Loss
from metta.rl.training.component_context import ComponentContext
from mettagrid.config import Config


class EMAConfig(Config):
    decay: float = Field(default=0.995, ge=0, le=1.0)
    loss_coef: float = Field(default=1.0, ge=0, le=1.0)

    def create(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create EMA loss instance."""
        return EMA(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class EMA(Loss):
    __slots__ = (
        "target_model",
        "ema_decay",
        "ema_coef",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)

        self.target_model = copy.deepcopy(self.policy)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.ema_decay = self.loss_cfg.decay
        self.ema_coef = self.loss_cfg.loss_coef

    def update_target_model(self):
        """Update target model with exponential moving average"""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.policy.parameters(), strict=False
            ):
                target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        self.update_target_model()
        policy_td = shared_loss_data["policy_td"]
        batch_shape = tuple(int(dim) for dim in policy_td.batch_size)
        if not batch_shape:
            raise RuntimeError("EMA requires a batch dimension")
        batch_size = batch_shape[0]
        time_steps = batch_shape[1] if len(batch_shape) > 1 else 1

        pred = policy_td["EMA_pred_output_2"].to(dtype=torch.float32).reshape(batch_size, time_steps, -1)

        target_td = policy_td.select(*self.policy_experience_spec.keys(include_nested=True)).clone()

        with torch.no_grad():
            self.target_model(target_td)
            target_pred = target_td["EMA_pred_output_2"].to(dtype=torch.float32).reshape(batch_size, time_steps, -1)

        shared_loss_data["EMA"]["pred"] = pred
        shared_loss_data["EMA"]["target_pred"] = target_pred

        loss = F.mse_loss(pred, target_pred) * self.ema_coef
        self.loss_tracker["EMA_mse_loss"].append(float(loss.item()))
        return loss, shared_loss_data, False

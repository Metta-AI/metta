import copy
from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.loss_config import LossSchedule
from metta.rl.training import ComponentContext
from metta.rl.utils import ensure_sequence_metadata
from mettagrid.base_config import Config


class EMAConfig(Config):
    schedule: LossSchedule | None = None
    decay: float = Field(default=0.995, ge=0, le=1.0)
    loss_coef: float = Field(default=1.0, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
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
        policy: Policy,
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
        B, TT = policy_td.batch_size
        policy_td = policy_td.reshape(B * TT)
        ensure_sequence_metadata(policy_td, batch_size=B, time_steps=TT)

        pred_flat: Tensor = policy_td["EMA_pred_output_2"].to(dtype=torch.float32)

        target_td = policy_td.select(*self.policy_experience_spec.keys(include_nested=True)).clone()
        ensure_sequence_metadata(target_td, batch_size=B, time_steps=TT)

        with torch.no_grad():
            self.target_model(target_td)
            target_pred_flat: Tensor = target_td["EMA_pred_output_2"].to(dtype=torch.float32)

        shared_loss_data["EMA"]["pred"] = pred_flat.reshape(B, TT, -1)
        shared_loss_data["EMA"]["target_pred"] = target_pred_flat.reshape(B, TT, -1)

        loss = F.mse_loss(pred_flat, target_pred_flat) * self.ema_coef
        self.loss_tracker["EMA_mse_loss"].append(float(loss.item()))
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)
        return loss, shared_loss_data, False

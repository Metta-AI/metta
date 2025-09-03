import copy
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F

from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.base_loss import BaseLoss

# from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState


class EMA(BaseLoss):
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
        checkpoint_manager: CheckpointManager,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, checkpoint_manager, instance_name, loss_config)

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

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        self.update_target_model()
        policy_td = shared_loss_data["policy_td"]

        # reshape to 1D for the head ie flatten the batch and time dimension
        B, TT = policy_td.batch_size[0], policy_td.batch_size[1]
        policy_td = policy_td.reshape(B * TT)
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
        policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))

        self.policy.policy.components["EMA_pred_output_2"](policy_td)
        pred: Tensor = policy_td["EMA_pred_output_2"].to(dtype=torch.float32)

        # target prediction: you need to clear all keys except env_obs and then clone
        target_td = policy_td.select(*self.policy_experience_spec.keys(include_nested=True)).clone()
        target_td.set("bptt", torch.full((B * TT,), TT, device=target_td.device, dtype=torch.long))
        target_td.set("batch", torch.full((B * TT,), B, device=target_td.device, dtype=torch.long))

        with torch.no_grad():
            self.target_model.components["EMA_pred_output_2"](target_td)
            target_pred: Tensor = target_td["EMA_pred_output_2"].to(dtype=torch.float32)

        # Store only tensors in shared_loss_data for downstream consumers
        shared_loss_data["EMA"]["pred"] = pred.reshape(B, TT, -1)
        shared_loss_data["EMA"]["target_pred"] = target_pred.reshape(B, TT, -1)

        loss = F.mse_loss(pred, target_pred) * self.ema_coef
        self.loss_tracker["EMA_mse_loss"].append(float(loss.item()))
        return loss, shared_loss_data

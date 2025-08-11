import copy
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.rl.loss.base_loss import BaseLoss, LossTracker
from metta.rl.trainer_config import TrainerConfig
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
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
        policy_store: PolicyStore,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, loss_tracker, policy_store)

        self.target_model = copy.deepcopy(self.policy)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.ema_decay = self.policy_cfg.losses.EMA.decay
        self.ema_coef = self.policy_cfg.losses.EMA.loss_coef

    def update_target_model(self):
        """Update target model with exponential moving average"""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.policy.parameters(), strict=False
            ):
                target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        self.update_target_model()
        policy_td = shared_loss_data["PPO"].select(*self.policy_experience_spec.keys(include_nested=True))
        pred = self.policy.components["pred_output"](policy_td)  # here, we run this head on our own
        with torch.no_grad():
            target_pred = self.target_model.components["pred_output"](policy_td)
            target_pred = target_pred.detach()
        shared_loss_data["BYOL"]["target_pred"] = target_pred  # add these in case other losses want them next
        shared_loss_data["BYOL"]["pred"] = pred
        loss = F.mse_loss(pred, target_pred) * self.ema_coef
        return loss, shared_loss_data

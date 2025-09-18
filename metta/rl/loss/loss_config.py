from typing import TYPE_CHECKING, Any, Dict

import torch
from pydantic import Field

from metta.mettagrid.config import Config
from metta.rl.loss.ppo import PPOConfig

if TYPE_CHECKING:
    from metta.agent.policy_base import Policy
    from metta.rl.training.training_environment import TrainingEnvironment


class LossSchedule(Config):
    start_epoch: int | None = Field(default=None)
    end_epoch: int | None = Field(default=None)


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(
        default={
            "ppo": PPOConfig(),
        }
    )

    def init_losses(
        self,
        policy: "Policy",
        trainer_cfg: Any,
        env: "TrainingEnvironment",
        device: torch.device,
    ):
        losses = {}
        for loss_name, loss_config in self.loss_configs.items():
            losses[loss_name] = loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
        return losses

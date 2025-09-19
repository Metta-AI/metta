from typing import Any, Dict

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.mettagrid.config import Config
from metta.rl.loss.ppo import PPOConfig
from metta.rl.training.training_environment import TrainingEnvironment
from metta.agent.metta_agent import PolicyAgent
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.ppo_config import PPOConfig
from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.ppo_config import PPOConfig
from mettagrid.config import Config


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
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
    ):
        losses = {}
        for loss_name, loss_config in self.loss_configs.items():
            losses[loss_name] = loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
        return losses

from typing import Any, Dict

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.ppo_config import PPOConfig
from mettagrid.config import Config


class LossSchedule(Config):
    start_epoch: int | None = Field(default=None)
    end_epoch: int | None = Field(default=None)


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(default={"ppo": PPOConfig(), "contrastive": ContrastiveConfig()})

    def init_losses(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
    ):
        losses = {}
        for loss_name, loss_config in self.loss_configs.items():
            losses[loss_name] = loss_config.init_loss(
                policy, trainer_cfg, vec_env, device, checkpoint_manager, loss_name, loss_config
            )
        return losses

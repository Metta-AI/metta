from typing import Any

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.sl_kickstarter import SLKickstarter
from metta.rl.trainer_config import TrainerConfig


class SLKickstarterConfig(Config):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)
    anneal_ratio: float = Field(default=0.995, ge=0, le=1.0)

    def init_loss(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        instance_name: str,
    ):
        """Points to the SLKickstarter class for initialization."""
        return SLKickstarter(policy, trainer_cfg, vec_env, device, checkpoint_manager, instance_name=instance_name)

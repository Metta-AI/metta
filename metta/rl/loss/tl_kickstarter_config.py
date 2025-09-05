from typing import Any

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.tl_kickstarter import TLKickstarter
from metta.rl.trainer_config import TrainerConfig


class TLKickstarterConfig(Config):
    teacher_uri: str = Field(default="")
    action_loss_coef: float = Field(default=0.995, ge=0, le=1.0)
    value_loss_coef: float = Field(default=1.0, ge=0, le=1.0)

    def init_loss(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        instance_name: str,
    ):
        """Points to the TLKickstarter class for initialization."""
        return TLKickstarter(policy, trainer_cfg, vec_env, device, checkpoint_manager, instance_name=instance_name)

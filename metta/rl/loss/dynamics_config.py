from typing import Any

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from metta.mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.dynamics import Dynamics

# from metta.rl.trainer_config import TrainerConfig


class DynamicsConfig(Config):
    returns_step_look_ahead: int = Field(default=1)
    returns_pred_coef: float = Field(default=1.0, ge=0, le=1.0)
    reward_pred_coef: float = Field(default=1.0, ge=0, le=1.0)

    def init_loss(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        instance_name: str,
        loss_config: Any,
    ):
        """Points to the EMA class for initialization."""
        return Dynamics(
            policy,
            trainer_cfg,
            vec_env,
            device,
            checkpoint_manager,
            instance_name=instance_name,
            loss_config=loss_config,
        )

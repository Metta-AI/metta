from typing import Any

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.common.config import Config
from metta.rl.loss.ema import EMA
from metta.rl.trainer_config import TrainerConfig


class EMAConfig(Config):
    decay: float = Field(default=0.995, ge=0, le=1.0)
    loss_coef: float = Field(default=1.0, ge=0, le=1.0)

    def init_loss(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        policy_store: PolicyStore,
        instance_name: str,
    ):
        """Points to the EMA class for initialization."""
        return EMA(policy, trainer_cfg, vec_env, device, policy_store, instance_name=instance_name)

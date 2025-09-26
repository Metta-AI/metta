# metta/rl/loss/contrastive_config.py
from typing import Any

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss.contrastive import ContrastiveLoss
from metta.rl.training import TrainingEnvironment
from mettagrid.config import Config


class ContrastiveConfig(Config):
    """Configuration for contrastive loss."""

    temperature: float = Field(default=0.07, gt=0, description="Temperature for contrastive learning")
    contrastive_coef: float = Field(default=0.1, ge=0, description="Coefficient for contrastive loss")
    discount: float = Field(
        default=0.977, ge=0, lt=1, description="Discount factor (gamma) used for geometric positive sampling"
    )
    embedding_dim: int = Field(default=128, gt=0, description="Dimension of contrastive embeddings")
    use_projection_head: bool = Field(default=True, description="Whether to use projection head")
    log_similarities: bool = Field(
        default=False, description="Whether to log positive/negative similarities to console"
    )
    log_frequency: int = Field(default=100, gt=0, description="Log similarities every N training steps")

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create the contrastive loss instance."""
        return ContrastiveLoss(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )

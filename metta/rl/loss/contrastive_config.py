# metta/rl/loss/contrastive_config.py
from typing import Any

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from mettagrid.config import Config
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.contrastive import ContrastiveLoss


class ContrastiveConfig(Config):
    """Configuration for contrastive loss."""

    temperature: float = Field(default=0.07, gt=0, description="Temperature for contrastive learning")
    contrastive_coef: float = Field(default=0.1, ge=0, description="Coefficient for contrastive loss")
    embedding_dim: int = Field(default=128, gt=0, description="Dimension of contrastive embeddings")
    use_projection_head: bool = Field(default=True, description="Whether to use projection head")

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
        """Initialize the contrastive loss."""
        return ContrastiveLoss(
            policy,
            trainer_cfg,
            vec_env,
            device,
            checkpoint_manager,
            instance_name=instance_name,
            loss_config=loss_config,
        )

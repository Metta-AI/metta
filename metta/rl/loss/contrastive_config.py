from typing import Any

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss.contrastive import ContrastiveLoss
from metta.rl.training.training_environment import TrainingEnvironment
from mettagrid.config import Config


class ContrastiveConfig(Config):
    """Configuration for contrastive loss."""

    # Contrastive learning hyperparameters
    temperature: float = Field(default=0.07, gt=0, description="Temperature scaling for InfoNCE loss")
    contrastive_coef: float = Field(default=0.1, ge=0, description="Coefficient for contrastive loss weight")
    embedding_dim: int = Field(default=128, gt=0, description="Dimension of contrastive embeddings")
    use_projection_head: bool = Field(default=True, description="Whether to use a projection head for embeddings")

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create a ContrastiveLoss instance."""
        return ContrastiveLoss(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=env,
            device=device,
            instance_name=instance_name,
            loss_config=loss_config,
        )

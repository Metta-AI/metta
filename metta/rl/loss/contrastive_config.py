# metta/rl/loss/contrastive_config.py
import typing

import pydantic
import torch

import metta.agent.policy
import metta.rl.loss.contrastive
import metta.rl.training.training_environment as training_environment
import mettagrid.base_config


class ContrastiveConfig(mettagrid.base_config.Config):
    """Configuration for contrastive loss."""

    temperature: float = pydantic.Field(
        default=0.1902943104505539, gt=0, description="Temperature for contrastive learning"
    )
    contrastive_coef: float = pydantic.Field(
        default=0.0006806607125326991, ge=0, description="Coefficient for contrastive loss"
    )
    discount: float = pydantic.Field(
        default=0.977, ge=0, lt=1, description="Discount factor (gamma) used for geometric positive sampling"
    )
    embedding_dim: int = pydantic.Field(default=128, gt=0, description="Dimension of contrastive embeddings")
    use_projection_head: bool = pydantic.Field(default=True, description="Whether to use projection head")
    log_similarities: bool = pydantic.Field(
        default=False, description="Whether to log positive/negative similarities to console"
    )
    log_frequency: int = pydantic.Field(default=100, gt=0, description="Log similarities every N training steps")

    def create(
        self,
        policy: metta.agent.policy.Policy,
        trainer_cfg: typing.Any,
        env: training_environment.TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: typing.Any,
    ):
        """Create the contrastive loss instance."""
        return metta.rl.loss.contrastive.ContrastiveLoss(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )

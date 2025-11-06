from __future__ import annotations

import typing

import pydantic
import torch

import metta.agent.policy
import metta.rl.loss
import mettagrid.base_config
import metta.rl.training.training_environment

if typing.TYPE_CHECKING:
    import metta.rl.training


class LossSchedule(mettagrid.base_config.Config):
    start_epoch: int | None = pydantic.Field(default=None)
    end_epoch: int | None = pydantic.Field(default=None)


class LossConfig(mettagrid.base_config.Config):
    loss_configs: typing.Dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    enable_contrastive: bool = pydantic.Field(default=False, description="Whether to enable contrastive loss")

    # Contrastive loss hyperparameters (only used when enable_contrastive=True)
    contrastive_temperature: float = pydantic.Field(
        default=0.07, gt=0, description="Temperature for contrastive learning"
    )
    contrastive_coef: float = pydantic.Field(default=0.1, ge=0, description="Coefficient for contrastive loss")
    contrastive_embedding_dim: int = pydantic.Field(
        default=128, gt=0, description="Dimension of contrastive embeddings"
    )
    contrastive_use_projection_head: bool = pydantic.Field(default=True, description="Whether to use projection head")

    def model_post_init(self, __context: typing.Any) -> None:
        """Called after the model is initialized."""
        super().model_post_init(__context)

        # If loss_configs is empty, add default PPO config
        if not self.loss_configs:
            self.loss_configs = {"ppo": metta.rl.loss.PPOConfig()}

        # Add contrastive config only if enabled to avoid inconsistent behavior
        if self.enable_contrastive and "contrastive" not in self.loss_configs:
            self.loss_configs["contrastive"] = metta.rl.loss.ContrastiveConfig(
                temperature=self.contrastive_temperature,
                contrastive_coef=self.contrastive_coef,
                embedding_dim=self.contrastive_embedding_dim,
                use_projection_head=self.contrastive_use_projection_head,
            )

    def init_losses(
        self,
        policy: metta.agent.policy.Policy,
        trainer_cfg: typing.Any,
        env: metta.rl.training.training_environment.TrainingEnvironment,
        device: torch.device,
    ):
        losses = {}
        for loss_name, loss_config in self.loss_configs.items():
            # Explicit check for inconsistent config
            if loss_name == "contrastive":
                if not self.enable_contrastive:
                    continue

            losses[loss_name] = loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
        return losses

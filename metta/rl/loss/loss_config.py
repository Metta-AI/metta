from typing import TYPE_CHECKING, Any, Dict

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss import ContrastiveConfig, PPOConfig
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.training import TrainingEnvironment


class LossSchedule(Config):
    start_epoch: int | None = Field(default=None)
    end_epoch: int | None = Field(default=None)


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(default_factory=dict)
    enable_contrastive: bool = Field(default=False, description="Whether to enable contrastive loss")
    enable_future_latent_ema: bool = Field(default=False, description="Whether to enable future latent EMA loss")

    # Contrastive loss hyperparameters (only used when enable_contrastive=True)
    contrastive_temperature: float = Field(default=0.07, gt=0, description="Temperature for contrastive learning")
    contrastive_coef: float = Field(default=0.1, ge=0, description="Coefficient for contrastive loss")
    contrastive_embedding_dim: int = Field(default=128, gt=0, description="Dimension of contrastive embeddings")
    contrastive_use_projection_head: bool = Field(default=True, description="Whether to use projection head")

    # Future latent EMA hyperparameters (only used when enable_future_latent_ema=True)
    future_latent_ema_decay: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description="Decay parameter for the EMA target when predicting future latent states",
    )
    future_latent_ema_horizon: int = Field(
        default=4,
        ge=1,
        description="How many future steps to include when building the EMA latent target",
    )
    future_latent_ema_coef: float = Field(
        default=1.0,
        ge=0.0,
        description="Scaling applied to the future latent EMA loss contribution",
    )

    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized."""
        super().model_post_init(__context)

        # If loss_configs is empty, add default PPO config
        if not self.loss_configs:
            self.loss_configs = {"ppo": PPOConfig()}

        # Add contrastive config only if enabled to avoid inconsistent behavior
        if self.enable_contrastive and "contrastive" not in self.loss_configs:
            self.loss_configs["contrastive"] = ContrastiveConfig(
                temperature=self.contrastive_temperature,
                contrastive_coef=self.contrastive_coef,
                embedding_dim=self.contrastive_embedding_dim,
                use_projection_head=self.contrastive_use_projection_head,
            )

        if self.enable_future_latent_ema and "future_latent_ema" not in self.loss_configs:
            from metta.rl.loss.future_latent_ema import FutureLatentEMALossConfig

            self.loss_configs["future_latent_ema"] = FutureLatentEMALossConfig(
                ema_decay=self.future_latent_ema_decay,
                prediction_horizon=self.future_latent_ema_horizon,
                loss_coef=self.future_latent_ema_coef,
            )

    def init_losses(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: "TrainingEnvironment",
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

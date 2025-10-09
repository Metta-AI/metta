from typing import TYPE_CHECKING, Any, Dict

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss import ContrastiveConfig
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.training import TrainingEnvironment


class LossSchedule(Config):
    start_epoch: int | None = Field(default=None)
    end_epoch: int | None = Field(default=None)


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(default_factory=dict)
    enable_contrastive: bool = Field(default=False, description="Whether to enable contrastive loss")

    # Contrastive loss hyperparameters (only used when enable_contrastive=True)
    contrastive_temperature: float = Field(default=0.07, gt=0, description="Temperature for contrastive learning")
    contrastive_coef: float = Field(default=0.1, ge=0, description="Coefficient for contrastive loss")
    contrastive_embedding_dim: int = Field(default=128, gt=0, description="Dimension of contrastive embeddings")
    contrastive_use_projection_head: bool = Field(default=True, description="Whether to use projection head")

    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized."""
        super().model_post_init(__context)

        # If loss_configs is empty, add default PPO config
        if not self.loss_configs:
            # Import here to avoid circular dependency
            from metta.rl.loss.grpo import GRPOConfig

            self.loss_configs = {"grpo": GRPOConfig()}
            # self.loss_configs = {"ppo": PPOConfig()}

        # Add contrastive config only if enabled to avoid inconsistent behavior
        if self.enable_contrastive and "contrastive" not in self.loss_configs:
            self.loss_configs["contrastive"] = ContrastiveConfig(
                temperature=self.contrastive_temperature,
                contrastive_coef=self.contrastive_coef,
                embedding_dim=self.contrastive_embedding_dim,
                use_projection_head=self.contrastive_use_projection_head,
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

from typing import TYPE_CHECKING, Any, Dict, List

import torch
from pydantic import Field, field_validator

from metta.agent.policy import Policy
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.training import TrainingEnvironment


class LossPhaseSchedule(Config):
    begin_at_epoch: int | None = Field(default=None, ge=0)
    end_at_epoch: int | None = Field(default=None, ge=0)
    cycle_length: int | None = Field(default=None, gt=0)
    active_in_cycle: List[int] | None = Field(default=None)

    @field_validator("active_in_cycle")
    @classmethod
    def _validate_active(cls, value: List[int] | None, info):
        if value is None:
            return None
        if any(step < 1 for step in value):
            raise ValueError("active_in_cycle entries must be >= 1")
        cycle_length = info.data.get("cycle_length")
        if cycle_length is not None and any(step > cycle_length for step in value):
            raise ValueError("active_in_cycle entries must be <= cycle_length")
        return value


class LossSchedule(Config):
    rollout: LossPhaseSchedule | None = None
    train: LossPhaseSchedule | None = None


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(default_factory=dict)
    enable_contrastive: bool = Field(default=False, description="Whether to enable contrastive loss")
    loss_schedules: Dict[str, LossSchedule] = Field(default_factory=dict)

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
            from metta.rl.loss.ppo import PPOConfig

            self.loss_configs = {"ppo": PPOConfig()}

        # Add contrastive config only if enabled to avoid inconsistent behavior
        if self.enable_contrastive and "contrastive" not in self.loss_configs:
            from metta.rl.loss.contrastive_config import ContrastiveConfig

            self.loss_configs["contrastive"] = ContrastiveConfig(
                temperature=self.contrastive_temperature,
                contrastive_coef=self.contrastive_coef,
                embedding_dim=self.contrastive_embedding_dim,
                use_projection_head=self.contrastive_use_projection_head,
            )

        for loss_name, schedule in self.loss_schedules.items():
            if loss_name not in self.loss_configs:
                raise ValueError(f"Schedule provided for unknown loss '{loss_name}'")
            loss_cfg = self.loss_configs[loss_name]
            if not hasattr(loss_cfg, "schedule"):
                raise ValueError(
                    f"Loss config '{loss_name}' does not support scheduling; define a 'schedule' field on the config."
                )
            loss_cfg.schedule = schedule

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

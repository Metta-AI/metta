from typing import TYPE_CHECKING, Any, Dict

import torch
from pydantic import Field

from metta.agent.policy import Policy
from mettagrid.config import Config

if TYPE_CHECKING:
    from metta.rl.training import TrainingEnvironment


class LossSchedule(Config):
    start_epoch: int | None = Field(default=None)
    end_epoch: int | None = Field(default=None)


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(default_factory=dict)
    enable_contrastive: bool = Field(default=False, description="Whether to enable contrastive loss")

    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized."""
        super().model_post_init(__context)

        # If loss_configs is empty, add default PPO config
        if not self.loss_configs:
            # Import here to avoid circular dependency
            from metta.rl.loss.ppo import PPOConfig

            self.loss_configs = {"ppo": PPOConfig()}

        # Add contrastive config only if enabled to avoid inconsistent behavior
        if self.enable_contrastive and "contrastive" not in self.loss_configs:
            # Import here to avoid circular dependency
            from metta.rl.loss.contrastive_config import ContrastiveConfig
            self.loss_configs["contrastive"] = ContrastiveConfig()

    def init_losses(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: "TrainingEnvironment",
        device: torch.device,
    ):
        losses = {}
        for loss_name, loss_config in self.loss_configs.items():
            # Explicit check with warning for inconsistent config
            if loss_name == "contrastive":
                if not self.enable_contrastive:
                    print(
                        "WARNING: 'contrastive' found in loss_configs but enable_contrastive=False. "
                        "Skipping contrastive loss."
                    )
                    continue
                else:
                    print("Initializing contrastive loss (enable_contrastive=True)")

            losses[loss_name] = loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
        return losses

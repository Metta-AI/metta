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

    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized."""
        super().model_post_init(__context)

        # If loss_configs is empty, add default PPO config
        if not self.loss_configs:
            # Import here to avoid circular dependency
            from metta.rl.loss.ppo import PPOConfig

            self.loss_configs = {"ppo": PPOConfig()}

    def init_losses(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: "TrainingEnvironment",
        device: torch.device,
    ):
        return {
            loss_name: loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
            for loss_name, loss_config in self.loss_configs.items()
        }

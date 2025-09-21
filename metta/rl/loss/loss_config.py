from typing import Any, Dict

import torch
from pydantic import Field

from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.contrastive_config import ContrastiveConfig
from metta.rl.loss.ppo_config import PPOConfig
from mettagrid.config import Config


class LossSchedule(Config):
    start_epoch: int | None = Field(default=None)
    end_epoch: int | None = Field(default=None)


class LossConfig(Config):
    loss_configs: Dict[str, Any] = Field(default_factory=lambda: {"ppo": PPOConfig()})
    enable_contrastive: bool = Field(default=False, description="Whether to enable contrastive loss")

    def __post_init__(self):
        """Add contrastive config only if enabled to avoid inconsistent behavior."""
        super().__post_init__()
        if self.enable_contrastive and "contrastive" not in self.loss_configs:
            self.loss_configs["contrastive"] = ContrastiveConfig()

    def init_losses(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
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

            losses[loss_name] = loss_config.init_loss(
                policy, trainer_cfg, vec_env, device, checkpoint_manager, loss_name, loss_config
            )
        return losses

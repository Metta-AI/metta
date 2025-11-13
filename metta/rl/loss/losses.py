from typing import TYPE_CHECKING

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss import contrastive_config
from metta.rl.loss.action_supervised import ActionSupervisedConfig
from metta.rl.loss.alternating_kickstarter import AlternatingKickstarterConfig
from metta.rl.loss.grpo import GRPOConfig
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.loss.sl_checkpointed_kickstarter import SLCheckpointedKickstarterConfig
from metta.rl.loss.sl_kickstarter import SLKickstarterConfig
from metta.rl.loss.tl_kickstarter import TLKickstarterConfig
from metta.rl.training import TrainingEnvironment
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class LossesConfig(Config):
    # PPO (Proximal Policy Optimization) is enabled by default as it's the primary
    # reinforcement learning algorithm used in most training scenarios
    ppo: PPOConfig = Field(default_factory=lambda: PPOConfig(enabled=True))
    contrastive: contrastive_config.ContrastiveConfig = Field(
        default_factory=lambda: contrastive_config.ContrastiveConfig(enabled=False)
    )
    supervisor: ActionSupervisedConfig = Field(default_factory=lambda: ActionSupervisedConfig(enabled=False))
    grpo: GRPOConfig = Field(default_factory=lambda: GRPOConfig(enabled=False))
    tl_kickstarter: TLKickstarterConfig = Field(default_factory=lambda: TLKickstarterConfig(enabled=False))
    sl_kickstarter: SLKickstarterConfig = Field(default_factory=lambda: SLKickstarterConfig(enabled=False))
    sl_checkpointed_kickstarter: SLCheckpointedKickstarterConfig = Field(
        default_factory=lambda: SLCheckpointedKickstarterConfig(enabled=False)
    )
    alternating_kickstarter: AlternatingKickstarterConfig = Field(
        default_factory=lambda: AlternatingKickstarterConfig(enabled=False)
    )

    def _configs(self) -> dict[str, LossConfig]:
        loss_configs: dict[str, LossConfig] = {}
        if self.ppo.enabled:
            loss_configs["ppo"] = self.ppo
        if self.contrastive.enabled:
            loss_configs["contrastive"] = self.contrastive
        if self.supervisor.enabled:
            loss_configs["supervisor"] = self.supervisor
        if self.grpo.enabled:
            loss_configs["grpo"] = self.grpo
        if self.tl_kickstarter.enabled:
            loss_configs["tl_kickstarter"] = self.tl_kickstarter
        if self.sl_kickstarter.enabled:
            loss_configs["sl_kickstarter"] = self.sl_kickstarter
        if self.sl_checkpointed_kickstarter.enabled:
            loss_configs["sl_checkpointed_kickstarter"] = self.sl_checkpointed_kickstarter
        if self.alternating_kickstarter.enabled:
            loss_configs["alternating_kickstarter"] = self.alternating_kickstarter
        return loss_configs

    def init_losses(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        env: TrainingEnvironment,
        device: torch.device,
    ) -> dict[str, Loss]:
        return {
            loss_name: loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
            for loss_name, loss_config in self._configs().items()
        }

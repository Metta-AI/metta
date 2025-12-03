from typing import TYPE_CHECKING

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.loss import contrastive_config
from metta.rl.loss.action_supervised import ActionSupervisedConfig
from metta.rl.loss.grpo import GRPOConfig
from metta.rl.loss.kickstarter import KickstarterConfig
from metta.rl.loss.logit_kickstarter import LogitKickstarterConfig
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.loss.ppo_actor import PPOActorConfig
from metta.rl.loss.ppo_critic import PPOCriticConfig
from metta.rl.loss.quantile_ppo_critic import QuantilePPOCriticConfig
from metta.rl.loss.sliced_kickstarter import SlicedKickstarterConfig
from metta.rl.loss.sliced_scripted_cloner import SlicedScriptedClonerConfig
from metta.rl.loss.vit_reconstruction import ViTReconstructionLossConfig
from metta.rl.training import TrainingEnvironment
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class LossesConfig(Config):
    # ENABLED BY DEFAULT: PPO split into two terms for flexibility, simplicity, and separation of concerns
    ppo_actor: PPOActorConfig = Field(default_factory=lambda: PPOActorConfig(enabled=True))
    ppo_critic: PPOCriticConfig = Field(default_factory=lambda: PPOCriticConfig(enabled=True))
    quantile_ppo_critic: QuantilePPOCriticConfig = Field(default_factory=lambda: QuantilePPOCriticConfig(enabled=False))

    # our original PPO in a single file
    ppo: PPOConfig = Field(default_factory=lambda: PPOConfig(enabled=False))

    # other aux losses below
    contrastive: contrastive_config.ContrastiveConfig = Field(
        default_factory=lambda: contrastive_config.ContrastiveConfig(enabled=False)
    )
    supervisor: ActionSupervisedConfig = Field(default_factory=lambda: ActionSupervisedConfig(enabled=False))
    grpo: GRPOConfig = Field(default_factory=lambda: GRPOConfig(enabled=False))
    kickstarter: KickstarterConfig = Field(default_factory=lambda: KickstarterConfig(enabled=False))
    sliced_kickstarter: SlicedKickstarterConfig = Field(default_factory=lambda: SlicedKickstarterConfig(enabled=False))
    logit_kickstarter: LogitKickstarterConfig = Field(default_factory=lambda: LogitKickstarterConfig(enabled=False))
    sliced_scripted_cloner: SlicedScriptedClonerConfig = Field(
        default_factory=lambda: SlicedScriptedClonerConfig(enabled=False)
    )
    vit_reconstruction: ViTReconstructionLossConfig = Field(
        default_factory=lambda: ViTReconstructionLossConfig(enabled=False)
    )

    def init_losses(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        env: TrainingEnvironment,
        device: torch.device,
    ) -> dict[str, Loss]:
        # losses are run in the order they are listed here
        configs: list[tuple[str, LossConfig]] = [
            ("sliced_kickstarter", self.sliced_kickstarter),
            ("sliced_scripted_cloner", self.sliced_scripted_cloner),
            ("ppo_critic", self.ppo_critic),
            ("quantile_ppo_critic", self.quantile_ppo_critic),
            ("ppo_actor", self.ppo_actor),
            ("ppo", self.ppo),
            ("vit_reconstruction", self.vit_reconstruction),
            ("contrastive", self.contrastive),
            ("grpo", self.grpo),
            ("action_supervisor", self.supervisor),
            ("kickstarter", self.kickstarter),
            ("logit_kickstarter", self.logit_kickstarter),
        ]
        return {
            name: cfg.create(policy, trainer_cfg, env, device, name)
            for name, cfg in configs
            if cfg.enabled
        }

from typing import TYPE_CHECKING, ClassVar

import torch
from pydantic import Field

from metta.agent.policy import Policy
from metta.rl.nodes import contrastive_config
from metta.rl.nodes.action_supervised import ActionSupervisedConfig
from metta.rl.nodes.cmpo import CMPOConfig
from metta.rl.nodes.dynamics import DynamicsConfig
from metta.rl.nodes.ema import EMAConfig
from metta.rl.nodes.eer_cloner import EERClonerConfig
from metta.rl.nodes.eer_kickstarter import EERKickstarterConfig
from metta.rl.nodes.grpo import GRPOConfig
from metta.rl.nodes.kickstarter import KickstarterConfig
from metta.rl.nodes.logit_kickstarter import LogitKickstarterConfig
from metta.rl.nodes.base import NodeBase, NodeConfig
from metta.rl.nodes.ppo_actor import PPOActorConfig
from metta.rl.nodes.ppo_critic import PPOCriticConfig
from metta.rl.nodes.quantile_ppo_critic import QuantilePPOCriticConfig
from metta.rl.nodes.sl_checkpointed_kickstarter import SLCheckpointedKickstarterConfig
from metta.rl.nodes.sliced_kickstarter import SlicedKickstarterConfig
from metta.rl.nodes.sliced_scripted_cloner import SlicedScriptedClonerConfig
from metta.rl.nodes.vit_reconstruction import ViTReconstructionLossConfig
from metta.rl.training import TrainingEnvironment
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class LossesConfig(Config):
    _LOSS_ORDER: ClassVar[tuple[str, ...]] = (
        "sliced_kickstarter",
        "sliced_scripted_cloner",
        "eer_kickstarter",
        "eer_cloner",
        "ppo_critic",
        "quantile_ppo_critic",
        "ppo_actor",
        "cmpo",
        "vit_reconstruction",
        "contrastive",
        "ema",
        "dynamics",
        "grpo",
        "supervisor",
        "sl_checkpointed_kickstarter",
        "kickstarter",
        "logit_kickstarter",
    )

    # ENABLED BY DEFAULT: PPO split into two terms for flexibility, simplicity, and separation of concerns
    ppo_actor: PPOActorConfig = Field(default_factory=lambda: PPOActorConfig(enabled=True))
    ppo_critic: PPOCriticConfig = Field(default_factory=lambda: PPOCriticConfig(enabled=True))

    quantile_ppo_critic: QuantilePPOCriticConfig = Field(default_factory=lambda: QuantilePPOCriticConfig(enabled=False))
    cmpo: CMPOConfig = Field(default_factory=lambda: CMPOConfig(enabled=False))

    # other aux losses below
    contrastive: contrastive_config.ContrastiveConfig = Field(
        default_factory=lambda: contrastive_config.ContrastiveConfig(enabled=False)
    )
    ema: EMAConfig = Field(default_factory=lambda: EMAConfig(enabled=False))
    dynamics: DynamicsConfig = Field(default_factory=lambda: DynamicsConfig(enabled=False))
    supervisor: ActionSupervisedConfig = Field(default_factory=lambda: ActionSupervisedConfig(enabled=False))
    grpo: GRPOConfig = Field(default_factory=lambda: GRPOConfig(enabled=False))
    kickstarter: KickstarterConfig = Field(default_factory=lambda: KickstarterConfig(enabled=False))
    sliced_kickstarter: SlicedKickstarterConfig = Field(default_factory=lambda: SlicedKickstarterConfig(enabled=False))
    logit_kickstarter: LogitKickstarterConfig = Field(default_factory=lambda: LogitKickstarterConfig(enabled=False))
    sliced_scripted_cloner: SlicedScriptedClonerConfig = Field(
        default_factory=lambda: SlicedScriptedClonerConfig(enabled=False)
    )
    sl_checkpointed_kickstarter: SLCheckpointedKickstarterConfig = Field(
        default_factory=lambda: SLCheckpointedKickstarterConfig(enabled=False)
    )
    eer_kickstarter: EERKickstarterConfig = Field(default_factory=lambda: EERKickstarterConfig(enabled=False))
    eer_cloner: EERClonerConfig = Field(default_factory=lambda: EERClonerConfig(enabled=False))
    vit_reconstruction: ViTReconstructionLossConfig = Field(
        default_factory=lambda: ViTReconstructionLossConfig(enabled=False)
    )

    def _configs(self) -> dict[str, NodeConfig]:
        # losses are run in the order they are listed here. This is not ideal and we should refactor this config.
        # also, the way it's setup doesn't let the experimenter give names to losses.
        loss_configs = {
            name: cfg for name, cfg in ((name, getattr(self, name)) for name in self._LOSS_ORDER) if cfg.enabled
        }
        return loss_configs

    @property
    def loss_configs(self) -> dict[str, NodeConfig]:
        return self._configs()

    def init_losses(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        env: TrainingEnvironment,
        device: torch.device,
    ) -> dict[str, NodeBase]:
        return {
            loss_name: loss_cfg.create(policy, trainer_cfg, env, device, loss_name)
            for loss_name, loss_cfg in self._configs().items()
        }

    def __iter__(self):
        """Iterate over (name, config) pairs for all loss configs."""
        for name in self._LOSS_ORDER:
            yield name, getattr(self, name)

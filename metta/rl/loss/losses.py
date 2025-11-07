import typing

import pydantic
import torch

import metta.agent.policy
import metta.rl.loss
import metta.rl.loss.action_supervised
import metta.rl.loss.grpo
import metta.rl.loss.loss
import metta.rl.loss.ppo
import metta.rl.training
import mettagrid.base_config

if typing.TYPE_CHECKING:
    import metta.rl.trainer_config

    TrainerConfig = metta.rl.trainer_config.TrainerConfig


class LossesConfig(mettagrid.base_config.Config):
    # PPO (Proximal Policy Optimization) is enabled by default as it's the primary
    # reinforcement learning algorithm used in most training scenarios
    ppo: metta.rl.loss.ppo.PPOConfig = pydantic.Field(default_factory=lambda: metta.rl.loss.ppo.PPOConfig(enabled=True))
    contrastive: metta.rl.loss.contrastive_config.ContrastiveConfig = pydantic.Field(
        default_factory=lambda: metta.rl.loss.contrastive_config.ContrastiveConfig(enabled=False)
    )
    supervisor: metta.rl.loss.action_supervised.ActionSupervisedConfig = pydantic.Field(
        default_factory=lambda: metta.rl.loss.action_supervised.ActionSupervisedConfig(enabled=False)
    )
    grpo: metta.rl.loss.grpo.GRPOConfig = pydantic.Field(
        default_factory=lambda: metta.rl.loss.grpo.GRPOConfig(enabled=False)
    )

    def _configs(self) -> dict[str, metta.rl.loss.loss.LossConfig]:
        loss_configs: dict[str, metta.rl.loss.loss.LossConfig] = {}
        if self.ppo.enabled:
            loss_configs["ppo"] = self.ppo
        if self.contrastive.enabled:
            loss_configs["contrastive"] = self.contrastive
        if self.supervisor.enabled:
            loss_configs["supervisor"] = self.supervisor
        if self.grpo.enabled:
            loss_configs["grpo"] = self.grpo
        return loss_configs

    def init_losses(
        self,
        policy: metta.agent.policy.Policy,
        trainer_cfg: "TrainerConfig",
        env: metta.rl.training.TrainingEnvironment,
        device: torch.device,
    ) -> dict[str, metta.rl.loss.loss.Loss]:
        return {
            loss_name: loss_config.create(policy, trainer_cfg, env, device, loss_name, loss_config)
            for loss_name, loss_config in self._configs().items()
        }

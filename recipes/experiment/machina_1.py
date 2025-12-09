"""Train only on the Machina v1 open world map with vibe-biased init."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.agent.policy import PolicyArchitecture
from metta.agent.policies.vit import ViTDefaultConfig
from metta.sim.simulation_config import SimulationConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import make_training_env, train_single_mission
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig


class VibeLogitBiasConfig(ComponentConfig):
    in_key: str = "logits"
    name: str = "vibe_logit_bias"

    def make_component(self, env: PolicyEnvInterface | None = None):
        if env is None:
            raise ValueError("VibeLogitBiasConfig requires PolicyEnvInterface")
        return VibeLogitBias(self, env)


class VibeLogitBias(nn.Module):
    """Add a constant bias so all vibe actions share one action's probability mass."""

    def __init__(self, config: VibeLogitBiasConfig, env: PolicyEnvInterface):
        super().__init__()
        self.in_key = config.in_key

        vibe_indices = [i for i, name in enumerate(env.action_names) if name.startswith("change_vibe_")]
        bias = torch.zeros(len(env.action_names), dtype=torch.float32)
        if vibe_indices:
            bias_value = -math.log(len(vibe_indices))
            bias[vibe_indices] = bias_value
        self.register_buffer("bias", bias)

    def forward(self, td: TensorDict) -> TensorDict:
        logits = td[self.in_key]
        if logits.shape[-1] == self.bias.shape[0]:
            td[self.in_key] = logits + self.bias.to(logits.device, logits.dtype)
        return td


class ViTWithVibeBiasConfig(ViTDefaultConfig):
    """ViT default policy with vibe logits down-weighted at init."""

    # Need to instantiate to access Pydantic field defaults
    components: list[ComponentConfig] = ViTDefaultConfig().components + [VibeLogitBiasConfig()]


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
) -> TrainTool:
    """Entrypoint that locks training to ``machina_1.open_world`` with CVC defaults and adds a matching eval."""

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )

    # Apply CVC sweep defaults to the trainer (mirrors recipes.experiment.cogs_v_clips)
    trainer_cfg = TrainerConfig(losses=LossesConfig())
    trainer_cfg.optimizer.learning_rate = 0.00737503357231617
    trainer_cfg.optimizer.eps = 5.0833278919526e-07

    trainer_cfg.losses.ppo.clip_coef = 0.22017136216163635
    trainer_cfg.losses.ppo.gae_lambda = 0.9900000095367432
    trainer_cfg.losses.ppo.vf_coef = 0.49657103419303894

    trainer_cfg.losses.ppo_actor.clip_coef = 0.22017136216163635

    trainer_cfg.losses.ppo_critic.gae_lambda = 0.9900000095367432
    trainer_cfg.losses.ppo_critic.vf_coef = 0.49657103419303894

    trainer_cfg.losses.quantile_ppo_critic.gae_lambda = 0.9900000095367432
    trainer_cfg.losses.quantile_ppo_critic.vf_coef = 0.49657103419303894

    tt.trainer = trainer_cfg

    tt.policy_architecture = policy_architecture or ViTWithVibeBiasConfig()

    # Replace eval suite with a single machina_1.open_world eval
    eval_env = make_training_env(num_cogs=num_cogs, mission="machina_1.open_world", variants=eval_variants)
    tt.evaluator.simulations = [
        SimulationConfig(
            suite="cogs_vs_clips",
            name=f"machina_1_open_world_{num_cogs}cogs",
            env=eval_env,
        )
    ]
    return tt


__all__ = ["train"]

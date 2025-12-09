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
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import train_single_mission


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

    components: list[ComponentConfig] = list(ViTDefaultConfig.components) + [VibeLogitBiasConfig()]


def train(
    num_cogs: int = 20,
    variants: Optional[Sequence[str]] = ("heart_chorus",),
    eval_variants: Optional[Sequence[str]] = ("heart_chorus",),
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
) -> TrainTool:
    """Entrypoint that locks training to ``machina_1.open_world`` with heart_chorus by default."""

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )

    tt.policy_architecture = policy_architecture or ViTWithVibeBiasConfig()
    return tt


__all__ = ["train"]

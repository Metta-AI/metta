"""Machina v1 open-world recipe with vibe bias and sweep helpers."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from recipes.experiment.cogs_v_clips import (
    apply_cvc_sweep_defaults,
    make_training_env,
    train_single_mission,
)


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

    components: list[ComponentConfig] = ViTDefaultConfig().components + [VibeLogitBiasConfig()]


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
) -> TrainTool:
    """Train on machina_1.open_world with sweep-tuned defaults and single-map eval."""

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
    )

    apply_cvc_sweep_defaults(tt.trainer)
    tt.policy_architecture = policy_architecture or ViTWithVibeBiasConfig()

    eval_env = make_training_env(num_cogs=num_cogs, mission="machina_1.open_world", variants=eval_variants)
    tt.evaluator.simulations = [
        SimulationConfig(
            suite="cogs_vs_clips",
            name=f"machina_1_open_world_{num_cogs}cogs",
            env=eval_env,
        )
    ]
    # Slow down evals for long runs
    tt.evaluator.epoch_interval = 3000
    return tt


def train_sweep(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
) -> TrainTool:
    """Sweep-friendly train with heart_chorus baked in."""

    base_variants = ["heart_chorus"]
    if variants:
        for v in variants:
            if v not in base_variants:
                base_variants.append(v)

    return train(
        num_cogs=num_cogs,
        variants=base_variants,
        eval_variants=eval_variants or base_variants,
        eval_difficulty=eval_difficulty,
        policy_architecture=policy_architecture,
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    """No-op evaluator for sweeps."""

    return StubTool()


def sweep(
    sweep_name: str,
    num_cogs: int = 4,
    eval_difficulty: str | None = "standard",
    max_trials: int = 80,
    num_parallel_trials: int = 4,
) -> SweepTool:
    """Hyperparameter sweep targeting train_sweep (heart_chorus baked in)."""

    search_space = {
        **SP.LEARNING_RATE,
        **SP.PPO_CLIP_COEF,
        **SP.PPO_GAE_LAMBDA,
        **SP.PPO_VF_COEF,
        **SP.ADAM_EPS,
        **SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=1e9,
        ),
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.machina_1",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        metric_key="env_agent/heart.gained",
        search_space=search_space,
        cost_key="metric/total_time",
        max_trials=max_trials,
        num_parallel_trials=num_parallel_trials,
    )


__all__ = [
    "train",
    "train_sweep",
    "evaluate_stub",
    "sweep",
]

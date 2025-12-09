"""Machina v1 open-world recipe using training-vibe subset and sweep helpers."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import MettaGridConfig
from recipes.experiment.cogs_v_clips import (
    apply_cvc_sweep_defaults,
    make_training_env,
    train_single_mission,
)


TRAINING_VIBE_NAMES: list[str] = [vibe.name for vibe in vibes.TRAINING_VIBES]


def _restrict_to_training_vibes(env: MettaGridConfig) -> None:
    """Use the TRAINING_VIBES subset and align vibe actions."""
    env.game.vibe_names = list(TRAINING_VIBE_NAMES)

    change_vibe = getattr(getattr(env.game, "actions", None), "change_vibe", None)
    if change_vibe is not None:
        change_vibe.number_of_vibes = len(TRAINING_VIBE_NAMES)

    if env.game.agent.initial_vibe >= len(TRAINING_VIBE_NAMES):
        env.game.agent.initial_vibe = 0


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
    bc_policy_uri: str | None = None,
    bc_mode: Literal["sliced_cloner", "supervisor"] = "sliced_cloner",
    bc_steps: int | None = None,
    bc_teacher_lead_prob: float = 1.0,
) -> TrainTool:
    """Train on machina_1.open_world with sweep-tuned defaults and single-map eval."""

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        bc_policy_uri=bc_policy_uri,
        bc_mode=bc_mode,
        bc_steps=bc_steps,
        bc_teacher_lead_prob=bc_teacher_lead_prob,
    )

    training_env_cfg = tt.training_env.curriculum.task_generator.env
    _restrict_to_training_vibes(training_env_cfg)

    apply_cvc_sweep_defaults(tt.trainer)
    tt.policy_architecture = policy_architecture or ViTDefaultConfig()

    eval_env = make_training_env(num_cogs=num_cogs, mission="machina_1.open_world", variants=eval_variants)
    _restrict_to_training_vibes(eval_env)
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
    bc_policy_uri: str | None = None,
    bc_mode: Literal["sliced_cloner", "supervisor"] = "sliced_cloner",
    bc_steps: int | None = None,
    bc_teacher_lead_prob: float = 1.0,
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
        bc_policy_uri=bc_policy_uri,
        bc_mode=bc_mode,
        bc_steps=bc_steps,
        bc_teacher_lead_prob=bc_teacher_lead_prob,
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

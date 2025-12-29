"""Machina v1 open-world recipe using the full vibe set and sweep helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from metta.agent.policy import PolicyArchitecture
    from metta.rl.training.teacher import TeacherConfig
    from metta.tools.stub import StubTool
    from metta.tools.sweep import SweepTool
    from metta.tools.train import TrainTool


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = None,
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Train on machina_1.open_world with leaderboard-aligned defaults and single-map eval."""
    from metta.agent.policies.vit import ViTDefaultConfig
    from metta.sim.simulation_config import SimulationConfig
    from mettagrid.config import vibes
    from recipes.experiment.cogs_v_clips import (
        _normalize_variant_names,
        make_training_env,
        train_single_mission,
    )

    if eval_variants is None:
        eval_variants = variants

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        teacher=teacher,
    )
    tt.policy_architecture = policy_architecture or ViTDefaultConfig()

    # Explicitly keep full vibe/action definitions so saved checkpoints remain compatible.
    env_cfg = tt.training_env.curriculum.task_generator.env
    env_cfg.game.vibe_names = [v.name for v in vibes.VIBES]
    change_vibe = getattr(env_cfg.game.actions, "change_vibe", None)
    if change_vibe is not None:
        change_vibe.vibes = list(vibes.VIBES)
    if env_cfg.game.agent.initial_vibe >= len(vibes.VIBES):
        env_cfg.game.agent.initial_vibe = 0

    eval_variant_names = _normalize_variant_names(
        initial=[eval_difficulty] if eval_difficulty else None,
        variants=eval_variants,
    )
    eval_env = make_training_env(
        num_cogs=num_cogs,
        mission="machina_1.open_world",
        variants=eval_variant_names or None,
    )
    tt.evaluator.simulations = [
        SimulationConfig(
            suite="cogs_vs_clips",
            name=f"machina_1_open_world_{num_cogs}cogs",
            env=eval_env,
        )
    ]
    # Run evals periodically during long runs
    tt.evaluator.epoch_interval = 150
    return tt


def train_sweep(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = None,
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Sweep-friendly train with heart_chorus baked in."""
    from recipes.experiment.cogs_v_clips import _normalize_variant_names

    base_variants = _normalize_variant_names(initial=["heart_chorus"], variants=variants)

    tt = train(
        num_cogs=num_cogs,
        variants=base_variants,
        eval_variants=eval_variants or base_variants,
        eval_difficulty=eval_difficulty,
        policy_architecture=policy_architecture,
        teacher=teacher,
    )
    # Sweep-friendly default (kept consistent with the shared CvC sweep search space).
    tt.trainer.total_timesteps = 1_000_000_000
    return tt


def evaluate_stub(*args, **kwargs) -> StubTool:
    """No-op evaluator for sweeps."""
    from metta.tools.stub import StubTool

    return StubTool()


def sweep(
    sweep_name: str,
    num_cogs: int = 4,
    eval_difficulty: str | None = "standard",
    max_trials: int = 80,
    num_parallel_trials: int = 4,
) -> SweepTool:
    """Hyperparameter sweep targeting train_sweep (heart_chorus baked in)."""
    from metta.sweep.core import make_sweep
    from recipes.experiment.cogs_v_clips import get_cvc_sweep_search_space

    search_space = get_cvc_sweep_search_space()

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.machina_1",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        metric_key="env_game/assembler.heart.created",
        search_space=search_space,
        cost_key="metric/total_time",
        max_trials=max_trials,
        num_parallel_trials=num_parallel_trials,
    )

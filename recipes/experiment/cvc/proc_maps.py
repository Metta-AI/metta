"""Procedural-map CoGs vs Clips entry points (experiment only)."""

from __future__ import annotations

from typing import Optional, Sequence

from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import evaluate
from recipes.experiment.cogs_v_clips import play
from recipes.experiment.cogs_v_clips import train_proc_maps as train


def train_sweep(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    mission: str | None = None,
    maps_cache_size: Optional[int] = 50,
) -> TrainTool:
    """Train with the heart_chorus variant baked in (CLI-friendly for sweeps)."""
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
        mission=mission,
        maps_cache_size=maps_cache_size,
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    """No-op evaluator for sweeps (avoids dispatching eval jobs)."""
    return StubTool()


def sweep(
    sweep_name: str,
    num_cogs: int = 4,
    eval_difficulty: str | None = "standard",
    max_trials: int = 80,
    num_parallel_trials: int = 4,
) -> SweepTool:
    """Hyperparameter sweep that targets train_sweep (heart_chorus baked in)."""
    parameters = [
        SP.LEARNING_RATE,
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.ADAM_EPS,
        SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=1e9,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.cvc.proc_maps",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        objective="env_agent/heart.gained",
        cost_key="metric/total_time",
        parameters=parameters,
        max_trials=max_trials,
        num_parallel_trials=num_parallel_trials,
        # Keep eval aligned with the baked-in training variant
        train_overrides={
            "num_cogs": num_cogs,
            "eval_difficulty": eval_difficulty,
        },
    )


__all__ = ["train", "play", "evaluate", "train_sweep", "sweep"]

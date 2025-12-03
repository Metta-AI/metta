"""
User-level sweep helpers for learning-progress experiments.

Expose lightweight factories so `./tools/run.py recipes.experiment.user.lp_sweep.cvc_arena_lp_grid`
can spin up a reusable grid-search sweep.
"""

from __future__ import annotations

from typing import Dict, Sequence

from metta.sweep.core import grid_search
from metta.tools.sweep import SweepTool

LP_SWEEP_PARAMETERS: Dict[str, Sequence[float]] = {
    "training_env.curriculum.algorithm_config.ema_timescale": [0.001, 0.005, 0.01],
    "training_env.curriculum.algorithm_config.z_score_amplification": [5.0, 10.0, 15.0],
    "training_env.curriculum.algorithm_config.exploration_bonus": [0.05, 0.1],
}

DEFAULT_TRAIN_OVERRIDES: Dict[str, object] = {
    "trainer.total_timesteps": 1_000_000,
    "checkpointer.epoch_interval": 10,
    "group": "lp_grid_sweep",
}


def _grid_cardinality(parameters: Dict[str, Sequence[object]]) -> int:
    total = 1
    for choices in parameters.values():
        total *= len(choices)
    return total


def cvc_arena_lp_grid() -> SweepTool:
    """
    Grid-search the main learning-progress knobs inside the CvC arena recipe.

    The sweep enumerates the Cartesian product of EMA timescales, z-score
    amplification factors, and exploration bonuses. Each configuration launches
    via the `grid_search` scheduler to keep orchestration simple while still
    recording metrics through the sweep toolchain.
    """

    return grid_search(
        name="lp_cvc_arena_grid",
        recipe="recipes.experiment.cvc_arena",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        objective="evaluator/eval_cvc_arena/score",
        parameters=LP_SWEEP_PARAMETERS,
        max_trials=_grid_cardinality(LP_SWEEP_PARAMETERS),
        num_parallel_trials=3,
        train_overrides=DEFAULT_TRAIN_OVERRIDES,
    )


def cvc_proc_maps_lp_grid() -> SweepTool:
    """Grid-search LP hypers for the procedural CVC recipe with throttled parallelism."""

    train_overrides = dict(DEFAULT_TRAIN_OVERRIDES)
    train_overrides.update(
        {
            "num_cogs": 4,
            "variants": ["heart_chorus", "lonely_heart"],
        }
    )

    return grid_search(
        name="lp_cvc_proc_grid",
        recipe="recipes.experiment.cvc.proc_maps",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        objective="evaluator/eval_cogs_vs_clips/score",
        parameters=LP_SWEEP_PARAMETERS,
        max_trials=_grid_cardinality(LP_SWEEP_PARAMETERS),
        num_parallel_trials=2,
        train_overrides=train_overrides,
    )


__all__ = ["cvc_arena_lp_grid", "cvc_proc_maps_lp_grid"]

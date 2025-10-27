# metta/adaptive/controller/ray_controller.py
from __future__ import annotations

import json
import sys
from typing import Any, Dict

from pydantic import Field
from ray import init, tune
from ray.tune import TuneConfig, Tuner, RunConfig

from metta.sweep.ray.ray_run_trial import metta_train_fn
from mettagrid.base_config import Config


class SweepConfig(Config):
    """
    Configuration for Ray-based adaptive controller.
    This is everything that pertains to **how** we're sweeping.
    """

    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    stats_server_uri: str | None = None
    experiment_id: str = "ray_v1"
    cpus_per_trial: float = 1.0
    gpus_per_trial: float = 0.0

    # New settings
    max_concurrent_trials: int = 1


def ray_sweep(
    *,
    param_space: Dict[str, Any] | None = None,
    num_samples: int = 1,
    sweep_config: SweepConfig | None = None,
) -> None:
    """
    Toy-phase controller: run a small local Ray Tune sweep (no pruning).
    Expects the trial to report "reward".
    """
    if sweep_config is None:
        sweep_config = SweepConfig()

    init(runtime_env={"working_dir": None})

    default_space: Dict[str, Any] = {
        # Example knobs; add/remove to taste
        "params": {
            "trainer.optimizer.learning_rate": tune.loguniform(1e-5, 3e-3),
            "trainer.total_timesteps": 5_000_000_000,
        },
        "sweep_config": sweep_config.model_dump(),
        # Optional extras the builder respects:
        # "recipe": tune.choice(["arena", "navigation"]),
        # "experiment_id": "toy-exp-001",
    }
    space = param_space or default_space
    trial_resources: dict[str, float] = {}
    if sweep_config.cpus_per_trial:
        trial_resources["cpu"] = sweep_config.cpus_per_trial
    if sweep_config.gpus_per_trial:
        trial_resources["gpu"] = sweep_config.gpus_per_trial

    trainable = tune.with_resources(metta_train_fn, trial_resources) if trial_resources else metta_train_fn

    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(
            num_samples=num_samples,
            metric="reward",  # must match the key reported by the runner
            mode="max",
            max_concurrent_trials=sweep_config.max_concurrent_trials,
        ),
        run_config=RunConfig(log_to_file=True, name=sweep_config.experiment_id),
        param_space=space,
    )
    tuner.fit()


if __name__ == "__main__":
    ray_sweep()

# metta/adaptive/controller/ray_controller.py
from __future__ import annotations

from typing import Any, Dict

from pydantic import Field
from ray import init, tune
from ray.tune import TuneConfig, Tuner

from metta.sweep.ray.ray_run_trial import metta_train_fn
from mettagrid.base_config import Config


class SweepConfig(Config):
    """
    Configuration for Ray-based adaptive controller.
    This is everything that pertains to **how** we're sweeping.
    """

    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped"

    # And we could add those in to the search space??
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"

    # We can get rid of the train_overrids I think now
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = Field(default=1)

    # TODO: Obviously not this
    sweep_id: str = Field(default='sweep_id_unset')

    cpus_per_trial: int = 48
    gpus_per_trial: int = 4
    max_concurrent_trials: int = 4

def ray_sweep(
    *,
    search_space: Dict[str, Any] | None = None,
    sweep_config: SweepConfig | None = None,
    ray_address: str | None = None,
) -> None:
    """
    Run a Ray Tune sweep using the provided configuration.

    Args:
        param_space: Optional Ray Tune parameter space. Defaults to a simple preset.
        num_samples: Number of Tune samples; falls back to sweep_config.max_trials.
        sweep_config: Sweep configuration; if omitted, uses defaults.
        static_overrides: Additional overrides applied to training jobs.
        ray_address: Optional Ray cluster address (e.g. ray://host:port).
    """
    sweep_config = sweep_config or SweepConfig()

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}

    if ray_address:
        init_kwargs["address"] = ray_address
    init_kwargs["runtime_env"] = {"working_dir": None}

    init(**init_kwargs)

    default_space: Dict[str, Any] = {
        "params": {
            "trainer.optimizer.learning_rate": tune.loguniform(1e-5, 3e-3),
            "trainer.total_timesteps": 50_000,
        },
        "sweep_config": sweep_config.model_dump(),
    }

    if not search_space:
        space = default_space
    else:
        space = {
            "params": search_space,
            "sweep_config": sweep_config.model_dump(),
        }

    trial_resources: dict[str, float] = {}
    if sweep_config.cpus_per_trial:
        trial_resources["cpu"] = sweep_config.cpus_per_trial
    if sweep_config.gpus_per_trial:
        trial_resources["gpu"] = sweep_config.gpus_per_trial

    trainable = tune.with_resources(metta_train_fn, trial_resources) if trial_resources else metta_train_fn


    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(
            num_samples=sweep_config.num_samples,
            metric="reward",
            mode="max",
            max_concurrent_trials=sweep_config.max_concurrent_trials,
        ),
        param_space=space,
    )
    tuner.fit()

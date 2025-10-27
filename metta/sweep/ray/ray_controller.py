# metta/adaptive/controller/ray_controller.py
from __future__ import annotations

import argparse
import ast
from typing import Any, Dict, Sequence

from pydantic import Field
from ray import init, tune
from ray.tune import RunConfig, TuneConfig, Tuner

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
    cpus_per_trial: int = 2
    gpus_per_trial: int = 1
    max_concurrent_trials: int = 1


def _parse_key_values(arguments: Sequence[str]) -> dict[str, Any]:
    """Parse key=value arguments into a dictionary with best-effort typing."""
    parsed: dict[str, Any] = {}
    for arg in arguments:
        if "=" not in arg:
            raise ValueError(f"Expected key=value format, received '{arg}'.")
        key, raw_value = arg.split("=", 1)
        try:
            parsed_value = ast.literal_eval(raw_value)
        except Exception:
            parsed_value = raw_value
        parsed[key] = parsed_value
    return parsed


def ray_sweep(
    *,
    param_space: Dict[str, Any] | None = None,
    num_samples: int | None = None,
    sweep_config: SweepConfig | None = None,
    static_overrides: Dict[str, Any] | None = None,
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
    if static_overrides:
        sweep_config.train_overrides.update(static_overrides)

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if ray_address:
        init_kwargs["address"] = ray_address
    else:
        init_kwargs["runtime_env"] = {"working_dir": None}
    init(**init_kwargs)

    default_space: Dict[str, Any] = {
        "params": {
            "trainer.optimizer.learning_rate": tune.loguniform(1e-5, 3e-3),
            "trainer.total_timesteps": 5_000_000_000,
        },
        "sweep_config": sweep_config.model_dump(),
    }
    space = param_space or default_space

    trial_resources: dict[str, float] = {}
    if sweep_config.cpus_per_trial:
        trial_resources["cpu"] = sweep_config.cpus_per_trial
    if sweep_config.gpus_per_trial:
        trial_resources["gpu"] = sweep_config.gpus_per_trial

    trainable = tune.with_resources(metta_train_fn, trial_resources) if trial_resources else metta_train_fn

    total_samples = num_samples or sweep_config.max_trials

    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(
            num_samples=total_samples,
            metric="reward",
            mode="max",
            max_concurrent_trials=sweep_config.max_concurrent_trials,
        ),
        run_config=RunConfig(log_to_file=True, name=sweep_config.experiment_id),
        param_space=space,
    )
    tuner.fit()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch a Metta Ray Tune sweep.")
    parser.add_argument("--ray-address", type=str, help="Ray cluster address (e.g., ray://host:6379).")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of Ray Tune samples to execute (defaults to sweep_config.max_trials).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Override SweepConfig.experiment_id.",
    )
    parser.add_argument(
        "--module-path",
        type=str,
        help="Module path of the training entrypoint (e.g. experiments.recipes.arena.train).",
    )
    known_args, remaining = parser.parse_known_args(argv)

    overrides = _parse_key_values(remaining)

    sweep_config = SweepConfig()
    if known_args.experiment_id:
        sweep_config.experiment_id = known_args.experiment_id

    if known_args.module_path:
        module_path = known_args.module_path.strip()
        if "." in module_path:
            module, entrypoint = module_path.rsplit(".", 1)
            sweep_config.recipe_module = module
            sweep_config.train_entrypoint = entrypoint

    try:
        ray_sweep(
            num_samples=known_args.num_samples,
            sweep_config=sweep_config,
            static_overrides=overrides,
            ray_address=known_args.ray_address,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

# metta/adaptive/controller/ray_controller.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import ray

from pydantic import Field
from ray import init, tune
from ray.tune import TuneConfig, Tuner

from metta.sweep.ray.ray_run_trial import metta_train_fn
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


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
    num_samples: int = Field(default=12)

    # TODO: Obviously not this
    sweep_id: str = Field(default="sweep_id_unset")

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

    cluster_resources = ray.cluster_resources()
    total_cpus = float(cluster_resources.get("CPU", 0.0))
    total_gpus = float(cluster_resources.get("GPU", 0.0))

    accelerator_keys = [k for k in cluster_resources if k.startswith("accelerator_type:")]
    accelerator_resource = os.getenv("RAY_ACCELERATOR_RESOURCE")
    if accelerator_resource and accelerator_resource not in cluster_resources:
        accelerator_resource = None
    if not accelerator_resource and accelerator_keys:
        accelerator_resource = accelerator_keys[0]

    logger.info(
        "Connected to Ray cluster: CPUs=%s, GPUs=%s, accelerator_resource=%s",
        total_cpus,
        total_gpus,
        accelerator_resource,
    )

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

    trial_bundle: dict[str, float] = {}
    if sweep_config.cpus_per_trial:
        trial_bundle["CPU"] = float(sweep_config.cpus_per_trial)
    if sweep_config.gpus_per_trial:
        trial_bundle["GPU"] = float(sweep_config.gpus_per_trial)
        if accelerator_resource:
            trial_bundle[accelerator_resource] = float(sweep_config.gpus_per_trial)

    effective_max_concurrent = max(int(sweep_config.max_concurrent_trials), 1)

    if sweep_config.cpus_per_trial:
        if total_cpus <= 0:
            logger.warning("Cluster reports zero CPUs; cannot derive CPU-based concurrency limit.")
        else:
            cpu_limit = int(total_cpus // sweep_config.cpus_per_trial)
            if cpu_limit == 0:
                raise ValueError(
                    "Requested %.2f CPUs per trial, but the cluster only reports %.2f CPUs."
                    % (sweep_config.cpus_per_trial, total_cpus)
                )
            effective_max_concurrent = min(effective_max_concurrent, cpu_limit)

    if sweep_config.gpus_per_trial:
        if total_gpus <= 0:
            logger.warning("Cluster reports zero GPUs; cannot derive GPU-based concurrency limit.")
        else:
            gpu_limit = int(total_gpus // sweep_config.gpus_per_trial)
            if gpu_limit == 0:
                raise ValueError(
                    "Requested %.2f GPUs per trial, but the cluster only reports %.2f GPUs."
                    % (sweep_config.gpus_per_trial, total_gpus)
                )
            effective_max_concurrent = min(effective_max_concurrent, gpu_limit)

    logger.info(
        "Trials will request resources: %s; max concurrent trials capped at %d",
        trial_bundle if trial_bundle else "(none)",
        effective_max_concurrent,
    )

    tune_config_kwargs: dict[str, Any] = dict(
        num_samples=sweep_config.num_samples,
        metric="reward",
        mode="max",
        max_concurrent_trials=effective_max_concurrent,
    )

    if trial_bundle:
        tune_config_kwargs["resources_per_trial"] = tune.PlacementGroupFactory([trial_bundle])

    trainable = metta_train_fn

    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(**tune_config_kwargs),
        param_space=space,
    )
    tuner.fit()

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

    cpus_per_trial: int = 1
    gpus_per_trial: int = 0
    max_concurrent_trials: int = 4
    max_failures_per_trial: int = 3  # Max retries for failed trials (e.g., spot terminations)
    fail_fast: bool = False  # Whether to stop the sweep if any trial fails permanently


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
        ray_address: Optional Ray cluster address (e.g. host:port or ray://host:port).
    """
    sweep_config = sweep_config or SweepConfig()

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}

    if ray_address:
        # Check if this is a client mode address (ray://) or local mode address
        if ray_address.startswith("ray://"):
            # Client mode - use as is but may have GPU allocation issues
            init_kwargs["address"] = ray_address
            logger.warning(
                "Using Ray client mode (ray://) which may not properly allocate GPUs to trials. "
                "Consider using local mode (host:port) instead."
            )
        else:
            # Local mode - better for GPU allocation
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
        "Connected to Ray cluster: CPUs=%s, GPUs=%s, accelerator_resource=%s, mode=%s",
        total_cpus,
        total_gpus,
        accelerator_resource,
        "client" if ray_address and ray_address.startswith("ray://") else "local",
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

    trial_resources: dict[str, float] = {}
    if sweep_config.cpus_per_trial:
        trial_resources["cpu"] = float(sweep_config.cpus_per_trial)
    if sweep_config.gpus_per_trial:
        trial_resources["gpu"] = float(sweep_config.gpus_per_trial)

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
        trial_resources if trial_resources else "(none)",
        effective_max_concurrent,
    )

    if trial_resources:
        trainable = tune.with_resources(metta_train_fn, resources=trial_resources)
    else:
        trainable = metta_train_fn

    # Configure failure handling for spot instances
    failure_config = tune.FailureConfig(
        max_failures=sweep_config.max_failures_per_trial,  # Retry failed trials
        fail_fast=sweep_config.fail_fast,  # Whether to stop on permanent failures
    )

    logger.info(
        "Failure handling configured: max_failures=%d, fail_fast=%s",
        sweep_config.max_failures_per_trial,
        sweep_config.fail_fast,
    )

    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(
            num_samples=sweep_config.num_samples,
            metric="reward",
            mode="max",
            max_concurrent_trials=effective_max_concurrent,
            failure_config=failure_config,
        ),
        param_space=space,
    )
    tuner.fit()

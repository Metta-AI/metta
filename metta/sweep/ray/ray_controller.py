# metta/adaptive/controller/ray_controller.py
from __future__ import annotations

import logging
import os
import time
from itertools import count
from typing import TYPE_CHECKING, Any, Dict

import ray
from pydantic import Field
from ray import init, tune
from ray.tune import RunConfig, TuneConfig, Tuner
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.repeater import Repeater

from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.sweep.ray.ray_run_trial import metta_train_fn
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

logger = logging.getLogger(__name__)


class SweepConfig(Config):
    """
    Configuration for Ray-based adaptive controller.
    This is everything that pertains to **how** we're sweeping.
    """

    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped"

    # And we could add those in to the search space??
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate_in_sweep"
    stats_server_uri: str | None = PROD_STATS_SERVER_URI

    # We can get rid of the train_overrids I think now
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = Field(default=12)

    # TODO: Obviously not this
    sweep_id: str = Field(default="sweep_id_unset")

    cpus_per_trial: int | str = "auto"  # Can be int or "auto" to use all available per node
    gpus_per_trial: int | str = "auto"  # Can be int or "auto" to use all available per node
    max_concurrent_trials: int = 4
    max_failures_per_trial: int = 3  # Max retries for failed trials (e.g., spot terminations)
    fail_fast: bool = False  # Whether to stop the sweep if any trial fails permanently

    # TODO I don't like having a default score key
    score_key: str = Field(default="evaluator/eval_sweep/score")

    # Number of times to repeat each suggested configuration (e.g., multi-seed evaluation)
    num_seeds_per_trial: int = Field(default=1, gt=0)


def ray_sweep(
    *,
    search_space: Dict[str, Any],
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

    if sweep_config.gpus_per_trial == "auto":
        gpus_from_env = os.getenv("METTA_DETECTED_GPUS_PER_NODE")
        if gpus_from_env:
            sweep_config.gpus_per_trial = int(gpus_from_env)
            logger.info(f"Auto-detected GPUs per trial from node hardware: {sweep_config.gpus_per_trial}")
        else:
            logger.warning(
                "'auto' specified for gpus_per_trial but METTA_DETECTED_GPUS_PER_NODE not set, defaulting to 0"
            )
            sweep_config.gpus_per_trial = 0

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}

    if ray_address:
        init_kwargs["address"] = ray_address
    init_kwargs["runtime_env"] = {"working_dir": None}

    init(**init_kwargs)

    cluster_resources = ray.cluster_resources()

    # Leave 4GPUs for other processes
    total_cpus = max(float(cluster_resources.get("CPU", 0.0)) - 4, 1)
    total_gpus = float(cluster_resources.get("GPU", 0.0))

    logger.info(
        "Connected to Ray cluster: CPUs=%s, GPUs=%s, mode=%s",
        total_cpus,
        total_gpus,
        "client" if ray_address and ray_address.startswith("ray://") else "local",
    )

    # Handle auto-number of CPUs
    if sweep_config.cpus_per_trial == "auto":
        sweep_config.cpus_per_trial = int(total_cpus / sweep_config.max_concurrent_trials)
        logger.info("Auto-set CPUs per trial to %s", sweep_config.cpus_per_trial)


    space = {
        "params": search_space,
        "sweep_config": sweep_config.model_dump(),
    }

    trial_resources: dict[str, float] = {}
    # At this point, cpus_per_trial and gpus_per_trial should be integers (auto already resolved)
    if isinstance(sweep_config.cpus_per_trial, int) and sweep_config.cpus_per_trial > 0:
        trial_resources["cpu"] = float(sweep_config.cpus_per_trial)
    else:
        resources = ray.cluster_resources()
        available_cpus = resources.get("CPU", 0)

        # We save 4 cores for other processed
        trial_resources["cpu"] = float(available_cpus - 4)//sweep_config.max_concurrent_trials

    if isinstance(sweep_config.gpus_per_trial, int) and sweep_config.gpus_per_trial > 0:
        trial_resources["gpu"] = float(sweep_config.gpus_per_trial)

    effective_max_concurrent = max(int(sweep_config.max_concurrent_trials), 1)

    if isinstance(sweep_config.cpus_per_trial, int) and sweep_config.cpus_per_trial > 0:
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

    if isinstance(sweep_config.gpus_per_trial, int) and sweep_config.gpus_per_trial > 0:
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

    optuna_search = OptunaSearch(metric="reward", mode="max")

    search_alg = optuna_search
    if sweep_config.num_seeds_per_trial > 1:
        repeated_search = Repeater(
            optuna_search,
            repeat=sweep_config.num_seeds_per_trial,
        )
        # Limit Optuna to a single suggestion's repeats at a time so it averages seeds before proposing new configs
        search_alg = ConcurrencyLimiter(
            repeated_search,
            max_concurrent=sweep_config.num_seeds_per_trial,
        )

    trial_counter = count()

    def trial_name_creator(trial: Trial) -> str:
        trial_index = next(trial_counter)
        return f"{sweep_config.sweep_id}.{trial_index}.{trial.trial_id}"

    tuner = Tuner(
        trainable,
        tune_config=TuneConfig(
            num_samples=sweep_config.num_samples,
            metric="reward",
            mode="max",
            # max_concurrent_trials=effective_max_concurrent,
            search_alg=search_alg,
            trial_name_creator=trial_name_creator,
        ),
        run_config=RunConfig(
            failure_config=failure_config,
        ),
        param_space=space,
    )
    tuner.fit()

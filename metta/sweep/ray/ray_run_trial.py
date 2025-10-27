from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

from ray import tune

from metta.adaptive.dispatcher import LocalDispatcher
from metta.adaptive.stores import WandbStore
from metta.adaptive.utils import create_training_job


def _fallback_run_id() -> str:
    return f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def metta_train_fn(config: dict[str, Any]) -> None:

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    """
    Train function for the metta model.
    Keys needed in config:
        - serialized_job_definition,
        - experiment_id
    """
    dispatcher = LocalDispatcher(capture_output=True)

    # Ray config should provide a dict payload under "serialized_job_definition".
    sweep_config = config["sweep_config"]

    # Get run name from Ray Tune
    ctx = tune.get_context()
    trial_name = ctx.get_trial_name()

    merged_overrides = dict(sweep_config.get("train_overrides", {}))
    merged_overrides.update(config["params"])

    job = create_training_job(
        run_id=trial_name,
        experiment_id=sweep_config.get("experiment_id"),
        recipe_module=sweep_config.get("recipe_module"),
        train_entrypoint=sweep_config.get("train_entrypoint"),
        stats_server_uri=sweep_config.get("stats_server_uri"),
        train_overrides=merged_overrides,
    )
    job.metadata["sweep/suggestion"] = config["params"]

    # TODO: Register SIGINTs for pruning
    job_pid = dispatcher.dispatch(job)

    print(f"Job ID: {job_pid}")
    latest_reward = 0

    # polling returns None as long as the process is running
    while dispatcher.get_process(job_pid).poll() is None:
        time.sleep(10)
        # Report latest reward to Ray

        # Increment stub
        latest_reward += 1

        # Poll WandB
        store = WandbStore(entity="metta-research", project=os.environ.get("WANDB_PROJECT", "metta"))
        summary = store.get_run_summary(trial_name)
        current_timestep = summary.get("metric/agent_step")
        current_reward = summary.get("metric/reward")
        tune.report({"reward": current_reward, "timestep": current_timestep})

    store = WandbStore(entity="metta-research", project=os.environ.get("WANDB_PROJECT", "metta"))
    summary = store.get_run_summary(trial_name)
    current_timestep = summary.get("metric/agent_step")
    current_reward = summary.get("metric/reward")
    tune.report({"reward": current_reward, "current_timestep": current_timestep})

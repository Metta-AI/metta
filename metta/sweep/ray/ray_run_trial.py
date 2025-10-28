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
from webbrowser import get


def _fallback_run_id() -> str:
    return f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _report_metrics(trial_name: str) -> dict[str, Any]:
    store = WandbStore(entity="metta-research", project=os.environ.get("WANDB_PROJECT", "metta"))
    try:
        summary = store.get_run_summary(trial_name)
        current_timestep = summary.get("metric/agent_step", 0)
        current_reward = summary.get("experience/rewards", 0)
        tune.report({"reward": current_reward, "timestep": current_timestep})
        return {"reward": current_reward, "timestep": current_timestep}
    except Exception as e:
        print(f"Error polling WandB: {e}")
        return {}


def metta_train_fn(config: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    """
    Train function for the metta model
    """
    dispatcher = LocalDispatcher(capture_output=True, use_torchrun=True)

    # Ray config should provide a dict payload under "serialized_job_definition".
    sweep_config = config["sweep_config"]

    # Get run name from Ray Tune
    ctx = tune.get_context()
    trial_name = ctx.get_trial_name()

    merged_overrides = dict(sweep_config.get("train_overrides", {}))
    merged_overrides.update(config["params"])

    job = create_training_job(
        run_id=trial_name,
        experiment_id=sweep_config.get("sweep_id"),
        recipe_module=sweep_config.get("recipe_module"),
        train_entrypoint=sweep_config.get("train_entrypoint"),
        stats_server_uri=sweep_config.get("stats_server_uri"),
        train_overrides=merged_overrides,
    )
    job.metadata["sweep/suggestion"] = config["params"]

    # TODO: Register SIGINTs for pruning
    job_pid = dispatcher.dispatch(job)

    print(f"Job ID: {job_pid}")

    # polling returns None as long as the process is running

    # Fetch process as it might have been reaped already
    training_proc = dispatcher.get_process(job_pid)

    # Wait for the process to finish
    while training_proc and training_proc.poll() is None:
        time.sleep(10)

        # Poll WandB
        _report_metrics(trial_name)

    # Give WandB a few seconds to sync
    time.sleep(20)
    _report_metrics(trial_name)

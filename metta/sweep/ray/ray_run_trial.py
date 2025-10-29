from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

from ray import get_gpu_ids, tune
from ray.runtime_context import get_runtime_context

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

    runtime_ctx = get_runtime_context()
    assigned_resources = None
    resource_ids = None
    gpu_id_strings: list[str] = []

    if runtime_ctx is not None:
        get_assigned = getattr(runtime_ctx, "get_assigned_resources", None)
        if callable(get_assigned):
            try:
                assigned_resources = get_assigned()
            except Exception as exc:
                logging.warning("Failed to read assigned resources from runtime context: %s", exc)

        get_resource_ids = getattr(runtime_ctx, "get_resource_ids", None)
        if callable(get_resource_ids):
            try:
                resource_ids = get_resource_ids()
                raw_gpu_ids = resource_ids.get("GPU")
                if raw_gpu_ids:
                    gpu_id_strings = [str(int(g)) for g in raw_gpu_ids]
            except Exception as exc:
                logging.warning("Failed to read resource IDs from runtime context: %s", exc)
        else:
            legacy_get_gpu_ids = getattr(runtime_ctx, "get_gpu_ids", None)
            if callable(legacy_get_gpu_ids):
                try:
                    raw_legacy_ids = legacy_get_gpu_ids()
                    if raw_legacy_ids:
                        gpu_id_strings = [str(int(g)) for g in raw_legacy_ids]
                except Exception as exc:
                    logging.warning("Failed to read GPU IDs from runtime context: %s", exc)

    if not gpu_id_strings:
        try:
            ray_gpu_ids = get_gpu_ids()
            if ray_gpu_ids:
                gpu_id_strings = [str(int(g)) for g in ray_gpu_ids]
        except Exception as exc:
            logging.warning("ray.get_gpu_ids() failed: %s", exc)

    if gpu_id_strings:
        cuda_visible = ",".join(gpu_id_strings)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        os.environ["RAY_GPU_IDS"] = cuda_visible
        os.environ["NUM_GPUS"] = str(len(gpu_id_strings))
        logging.info(
            "Configured CUDA visibility for trial %s: %s (num_gpus=%s)",
            sweep_config.get("sweep_id"),
            cuda_visible,
            len(gpu_id_strings),
        )
    else:
        logging.warning("Ray did not provide GPU IDs; CUDA visibility remains unchanged for this trial.")

    logging.info(
        "Ray runtime assigned resources: %s; resource_ids: %s; gpu_id_strings: %s",
        assigned_resources,
        resource_ids,
        gpu_id_strings,
    )

    # Get run name from Ray Tune
    ctx = tune.get_context()
    trial_name = ctx.get_trial_name()

    merged_overrides = dict(sweep_config.get("train_overrides", {}))
    merged_overrides.update(config["params"])

    requested_gpus = sweep_config.get("gpus_per_trial")
    if (requested_gpus is None or requested_gpus == 0) and gpu_id_strings:
        requested_gpus = len(gpu_id_strings)

    job = create_training_job(
        run_id=trial_name,
        experiment_id=sweep_config.get("sweep_id"),
        recipe_module=sweep_config.get("recipe_module"),
        train_entrypoint=sweep_config.get("train_entrypoint"),
        gpus=requested_gpus or 0,
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

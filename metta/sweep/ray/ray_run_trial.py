from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any

from ray import get_gpu_ids, tune
from ray.runtime_context import get_runtime_context

from metta.adaptive.dispatcher import LocalDispatcher
from metta.adaptive.models import JobDefinition
from metta.adaptive.stores import WandbStore
from metta.adaptive.utils import create_training_job

logger = logging.getLogger(__name__)

def _fallback_run_id() -> str:
    return f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _report_metrics(trial_name: str, score_key: str | None = None) -> dict[str, Any]:
    store = WandbStore(entity="metta-research", project=os.environ.get("WANDB_PROJECT", "metta"))
    try:
        summary = store.get_run_summary(trial_name)
        current_timestep = summary.get("metric/agent_step", 0)
        current_reward = summary.get("experience/rewards", 0)
    except Exception as e:
        print(f"Error polling WandB: {e}")
        return {}


def _save_metadata(job: JobDefinition) -> None:
    store = WandbStore(entity="metta-research", project=os.environ.get("WANDB_PROJECT", "metta"))
    try:
        store.update_run_summary(run_id=job.run_id, summary_update=job.metadata)
    except Exception as e:
        print(f"Error saving metadata: {e}")


def metta_train_fn(config: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    """
    Train function for the metta model
    """

    # Track the training process and termination status
    # TODO: Can we refactor this  (COOLING)
    training_proc = None
    spot_termination = False

    def handle_sigterm(signum, frame):
        """Handle SIGTERM signal (spot instance termination)"""
        nonlocal spot_termination
        spot_termination = True
        logging.warning("SIGTERM received - likely spot instance termination. Attempting graceful shutdown...")

        # If we have a running training process, terminate it gracefully
        if training_proc and training_proc.poll() is None:
            logging.info("Terminating training process...")
            training_proc.terminate()
            # Give it some time to save checkpoint
            time.sleep(10)
            if training_proc.poll() is None:
                training_proc.kill()

        # Exit with special code to indicate spot termination
        # Ray Tune can use this to determine if it should retry
        sys.exit(124)  # 124 = timeout/spot termination

    # Register SIGTERM handler only if we're in the main thread
    # Ray Tune may run trials in worker threads where signal handlers can't be registered
    import threading

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handle_sigterm)
    else:
        logger.info("Not in main thread, skipping SIGTERM handler registration")

    # Ray config should provide a dict payload under "serialized_job_definition".
    sweep_config = config["sweep_config"]
    ctx = tune.get_context()
    trial_name = ctx.get_trial_name()

    # Check if Ray assigned GPUs to this trial
    runtime_ctx = get_runtime_context()
    assigned_gpus = 0

    if runtime_ctx is not None:
        # Try to get GPU assignment from Ray
        try:
            ray_gpu_ids = get_gpu_ids()
            if ray_gpu_ids:
                assigned_gpus = len(ray_gpu_ids)
                logging.info(f"Ray assigned {assigned_gpus} GPU(s) to trial {trial_name}: {ray_gpu_ids}")
            else:
                logging.warning(f"No GPUs assigned by Ray to trial {trial_name}")
        except Exception as e:
            logging.warning(f"Failed to get GPU IDs from Ray: {e}")

    # Use assigned GPUs or fall back to sweep config
    gpus_for_job = assigned_gpus if assigned_gpus > 0 else sweep_config.get("gpus_per_trial", 0)
    training_dispatcher = LocalDispatcher(capture_output=True, use_torchrun=(gpus_for_job > 0))

    # TODO We can refactor this now
    merged_overrides.update(config["params"])

    job = create_training_job(
        run_id=trial_name,
        experiment_id=sweep_config.get("sweep_id"),
        recipe_module=sweep_config.get("recipe_module"),
        train_entrypoint=sweep_config.get("train_entrypoint"),
        stats_server_uri=sweep_config.get("stats_server_uri"),
        train_overrides=config["params"]
    )
    job.metadata["sweep/suggestion"] = config["params"]
    job.metadata["sweep/assigned_gpus"] = assigned_gpus
    job.metadata["sweep/trial_id"] = ctx.get_trial_id()
    print(f"Job ID: {job_pid}")

    training_proc = training_dispatcher.get_process(job_pid)

    # polling returns None as long as the process is running
    # Wait for the process to finish
    while training_proc and training_proc.poll() is None:
        # TODO Make this configurable
        time.sleep(60)

        # Poll WandB
        _report_metrics(trial_name)

    # Give WandB a few seconds to sync
    time.sleep(20)
    _report_metrics(trial_name)
    _save_metadata(job)

    # Check exit code to determine if this was a normal exit or failure
    if training_proc:
        exit_code = training_proc.returncode
        if exit_code == 124:
            # Spot termination - should be retried
            logging.info("Trial terminated due to spot instance preemption (exit code 124)")
            sys.exit(124)
        elif exit_code != 0:
            # Other failure
            logging.error(f"Trial failed with exit code {exit_code}")
            sys.exit(exit_code if exit_code else 1)

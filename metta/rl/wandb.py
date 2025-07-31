from __future__ import annotations

import json
import logging
import os
import time
import weakref
from typing import Any, Dict

import torch.nn as nn
import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Use WeakKeyDictionary to associate state with each wandb.Run without mutating the object
_ABORT_STATE: weakref.WeakKeyDictionary[wandb.sdk.wandb_run.Run, Dict[str, Any]] = weakref.WeakKeyDictionary()


def abort_requested(wandb_run: wandb.sdk.wandb_run.Run | None, min_interval_sec: int = 60) -> bool:
    """Return True if the WandB run has an "abort" tag.

    The API call is throttled to *min_interval_sec* seconds.
    """
    if wandb_run is None:
        return False

    state = _ABORT_STATE.setdefault(wandb_run, {"last_check": 0.0, "cached_result": False})
    now = time.time()

    # Return cached result if within throttle interval
    if now - state["last_check"] < min_interval_sec:
        return state["cached_result"]

    # Time to check again
    state["last_check"] = now
    try:
        run_obj = wandb.Api().run(wandb_run.path)
        state["cached_result"] = "abort" in run_obj.tags
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        state["cached_result"] = False

    return state["cached_result"]


def upload_env_configs(env_configs: dict[str, Any], wandb_run: wandb.sdk.wandb_run.Run | None) -> None:
    """Serialize resolved env configs and upload to run files.

    Args:
        env_configs: Dictionary mapping bucket names to their environment configurations
        wandb_run: The wandb run to upload to
    """
    if wandb_run is None:
        return

    try:
        resolved = {k: OmegaConf.to_container(v, resolve=True) for k, v in env_configs.items()}
        payload = json.dumps(resolved, indent=2)
        file_path = os.path.join(wandb_run.dir, "env_configs.json")
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(payload)
        try:
            wandb_run.save(file_path, base_path=wandb_run.dir, policy="now")
        except Exception:
            pass  # offline mode
    except Exception as e:
        logger.warning(f"Failed to upload env configs: {e}")


# Metrics functions moved from metrics.py
def setup_wandb_metrics(wandb_run: Any) -> None:
    """Set up wandb metric definitions for consistent tracking across runs.

    Args:
        wandb_run: The wandb run object
    """
    # Define base metrics
    metrics = ["agent_step", "epoch", "total_time", "train_time"]
    for metric in metrics:
        wandb_run.define_metric(f"metric/{metric}")

    # Set agent_step as the default x-axis for all metrics
    wandb_run.define_metric("*", step_metric="metric/agent_step")

    # Define special metric for reward vs total time
    wandb_run.define_metric("overview/reward_vs_total_time", step_metric="metric/total_time")


def log_model_parameters(policy: nn.Module, wandb_run: Any) -> None:
    """Log model parameter count to wandb summary.

    Args:
        policy: The policy model
        wandb_run: The wandb run object
    """
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


def log_training_metrics(
    wandb_run: Any,
    metrics: dict[str, Any],
    step: int,
) -> None:
    """Log training metrics to wandb.

    Args:
        wandb_run: The wandb run object
        metrics: Dictionary of metrics to log
        step: The current training step
    """
    wandb_run.log(metrics, step=step)


def define_custom_metric(
    wandb_run: Any,
    metric_name: str,
    step_metric: str | None = None,
) -> None:
    """Define a custom metric with optional step metric.

    Args:
        wandb_run: The wandb run object
        metric_name: Name of the metric to define
        step_metric: Optional step metric to use as x-axis
    """
    if step_metric:
        wandb_run.define_metric(metric_name, step_metric=step_metric)
    else:
        wandb_run.define_metric(metric_name)


def upload_policy_artifact(
    wandb_run: Any,
    policy_store: Any,
    policy_record: Any,
    force: bool = False,
) -> str | None:
    """Upload policy to WandB as artifact.

    Args:
        wandb_run: WandB run object
        policy_store: Policy store
        policy_record: Policy record to upload
        force: Force upload even if already uploaded (currently unused)

    Returns:
        WandB policy name or None if failed
    """
    if not wandb_run or not policy_record:
        return None

    try:
        wandb_policy_name = policy_store.add_to_wandb_run(wandb_run.id, policy_record)
        logger.info(f"Uploaded policy to wandb: {wandb_policy_name}")
        return wandb_policy_name
    except Exception as e:
        logger.warning(f"Failed to upload policy to wandb: {e}")
        return None

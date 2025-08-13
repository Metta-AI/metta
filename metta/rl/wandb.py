from __future__ import annotations

import logging
import time
import weakref
from typing import Dict

import torch.nn as nn
import wandb

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbRun

logger = logging.getLogger(__name__)

# Use WeakKeyDictionary to associate state with each wandb.Run without mutating the object
_ABORT_STATE: weakref.WeakKeyDictionary[WandbRun, Dict[str, float | bool]] = weakref.WeakKeyDictionary()


def abort_requested(wandb_run: WandbRun | None, min_interval_sec: int = 60) -> bool:
    """Return True if the WandB run has an "abort" tag.

    The API call is throttled to *min_interval_sec* seconds.
    """
    if wandb_run is None:
        return False

    state = _ABORT_STATE.setdefault(wandb_run, {"last_check": 0.0, "cached_result": False})
    now = time.time()

    # Return cached result if within throttle interval
    if now - state["last_check"] < min_interval_sec:
        return bool(state["cached_result"])

    # Time to check again
    state["last_check"] = now
    try:
        run_obj = wandb.Api().run(wandb_run.path)
        state["cached_result"] = "abort" in run_obj.tags
    except Exception as e:
        logger.debug(f"Abort tag check failed: {e}")
        state["cached_result"] = False

    return bool(state["cached_result"])


POLICY_EVALUATOR_METRIC_PREFIX = "policy_evaluator"
POLICY_EVALUATOR_STEP_METRIC = "metric/policy_evaluator_agent_step"
POLICY_EVALUATOR_EPOCH_METRIC = "metric/policy_evaluator_epoch"


# Metrics functions moved from metrics.py
def setup_wandb_metrics(wandb_run: WandbRun) -> None:
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
    setup_policy_evaluator_metrics(wandb_run)


def setup_policy_evaluator_metrics(wandb_run: WandbRun) -> None:
    # Separate step metric for remote evaluation allows evaluation results to be logged without conflicts
    wandb_run.define_metric(POLICY_EVALUATOR_STEP_METRIC)
    wandb_run.define_metric(f"{POLICY_EVALUATOR_METRIC_PREFIX}/*", step_metric=POLICY_EVALUATOR_STEP_METRIC)


def log_model_parameters(policy: nn.Module, wandb_run: WandbRun) -> None:
    """Log model parameter count to wandb summary.

    Args:
        policy: The policy model
        wandb_run: The wandb run object
    """
    num_params = sum(p.numel() for p in policy.parameters())
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params


def upload_policy_artifact(
    wandb_run: WandbRun | None,
    policy_store: PolicyStore,
    policy_record: PolicyRecord,
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

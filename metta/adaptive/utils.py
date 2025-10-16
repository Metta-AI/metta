"""Utility functions for adaptive experiment orchestration."""

import hashlib
import logging
import time
from typing import Any, Dict, Optional

from metta.adaptive.models import JobDefinition, JobTypes, RunInfo
from metta.common.util.constants import PROD_STATS_SERVER_URI, SOFTMAX_S3_POLICY_PREFIX

logger = logging.getLogger(__name__)


def make_monitor_table(
    runs: list["RunInfo"],
    title: str = "Run Status Table",
    logger_prefix: str = "",
    include_score: bool = True,
    truncate_run_id: bool = True,
) -> list[str]:
    """Create a formatted table showing run status.

    Args:
        runs: List of RunInfo objects to display
        title: Title for the table
        logger_prefix: Prefix to add to each log line (e.g., "[OptimizingScheduler]")
        include_score: Whether to include the score column
        truncate_run_id: Whether to truncate run IDs to just show trial numbers

    Returns:
        List of formatted lines that can be logged
    """
    lines = []
    prefix = f"{logger_prefix} " if logger_prefix else ""

    # Title
    lines.append(f"{prefix}{title}:")
    lines.append(f"{prefix}{'=' * 110}")

    # Header
    if include_score:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30} {'Score':<15} {'Cost':<10}")
    else:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30}")
    lines.append(f"{prefix}{'-' * 110}")

    # Rows
    for run in runs:
        # Format run ID
        display_id = get_display_id(run.run_id) if truncate_run_id else run.run_id

        # Format progress in Gsteps
        if run.total_timesteps and run.current_steps is not None:
            current_gsteps = run.current_steps / 1_000_000_000
            total_gsteps = run.total_timesteps / 1_000_000_000
            progress_pct = (run.current_steps / run.total_timesteps) * 100
            progress_str = f"{current_gsteps:.3g}/{total_gsteps:.3g} Gsteps ({progress_pct:.1f}%)"
        elif run.current_steps is not None:
            current_gsteps = run.current_steps / 1_000_000_000
            progress_str = f"{current_gsteps:.3g}/? Gsteps"
        else:
            progress_str = "-"

        # Format score and cost
        if include_score:
            # Try to get score/cost from sweep namespace first, then from observation field (backwards compat)
            summary = run.summary if isinstance(run.summary, dict) else {}
            score = summary.get("sweep/score")
            cost = summary.get("sweep/cost")

            # Backwards compatibility: check old observation field
            if score is None and hasattr(run, "observation") and run.observation:
                score = run.observation.score
                cost = run.observation.cost

            score_str = f"{float(score):.4f}" if score is not None else "N/A"
            cost_str = f"${float(cost):.2f}" if cost is not None else "N/A"
            lines.append(
                f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30} {score_str:<15} {cost_str:<10}"
            )
        else:
            lines.append(f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30}")

    lines.append(f"{prefix}{'=' * 110}")

    return lines


def get_display_id(run_id: str) -> str:
    """Extract clean display ID from run ID.

    Args:
        run_id: Full run ID (e.g., "experiment_name_trial_0001_a1b2c3")

    Returns:
        Cleaned display ID (e.g., "trial_0001")
    """
    if "_trial_" in run_id:
        # Extract everything after "_trial_"
        trial_part = run_id.split("_trial_")[-1]
        run_id = trial_part
    return run_id


def build_eval_overrides(
    run_id: str,
    experiment_id: str,
    stats_server_uri: Optional[str] = PROD_STATS_SERVER_URI,
    additional_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build evaluation override parameters.

    Args:
        run_id: The run ID for WandB tracking
        experiment_id: The experiment ID for grouping
        stats_server_uri: Optional stats server URI
        additional_overrides: Optional additional overrides to merge

    Returns:
        Dictionary of evaluation overrides
    """
    eval_overrides = dict(additional_overrides) if additional_overrides else {}

    # WandB configuration - simplified to match new setup
    eval_overrides["push_metrics_to_wandb"] = "True"
    # Use 'group' instead of 'wandb.group' to match train.py pattern
    eval_overrides["group"] = experiment_id

    # Stats server configuration
    if stats_server_uri:
        eval_overrides["stats_server_uri"] = stats_server_uri

    return eval_overrides


def build_train_overrides(
    stats_server_uri: Optional[str] = None,
    additional_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build training override parameters.

    Args:
        stats_server_uri: Optional stats server URI
        additional_overrides: Optional additional overrides to merge

    Returns:
        Dictionary of training overrides
    """
    overrides = dict(additional_overrides) if additional_overrides else {}

    if stats_server_uri:
        overrides["stats_server_uri"] = stats_server_uri
        overrides["evaluator.evaluate_remote"] = "True"
        overrides["evaluator.evaluate_local"] = "False"
        overrides["evaluator.skip_git_check"] = "True"

    return overrides


def create_eval_job(
    run_id: str,
    experiment_id: str,
    recipe_module: str,
    eval_entrypoint: str,
    stats_server_uri: Optional[str] = PROD_STATS_SERVER_URI,
    eval_overrides: Optional[Dict[str, Any]] = None,
) -> "JobDefinition":
    """Create an evaluation job definition.

    Args:
        run_id: The run ID to evaluate
        experiment_id: The experiment ID for grouping
        recipe_module: Module containing the evaluation function
        eval_entrypoint: Name of the evaluation function
        stats_server_uri: Optional stats server URI
        eval_args: Optional positional arguments for evaluation
        eval_overrides: Optional additional overrides

    Returns:
        JobDefinition for evaluation
    """

    overrides = build_eval_overrides(
        run_id=run_id,
        experiment_id=experiment_id,
        stats_server_uri=stats_server_uri,
        additional_overrides=eval_overrides,
    )

    return JobDefinition(
        run_id=run_id,
        cmd=f"{recipe_module}.{eval_entrypoint}",
        type=JobTypes.LAUNCH_EVAL,
        args={"policy_uri": f"{SOFTMAX_S3_POLICY_PREFIX}/{run_id}:latest"},
        overrides=overrides,
        metadata={},
    )


def create_training_job(
    run_id: str,
    experiment_id: str,
    recipe_module: str,
    train_entrypoint: str,
    gpus: int = 1,
    nodes: int = 1,
    stats_server_uri: Optional[str] = None,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> "JobDefinition":
    """Create a training job definition.

    Args:
        run_id: The unique run ID
        experiment_id: The experiment ID for grouping
        recipe_module: Module containing the training function
        train_entrypoint: Name of the training function
        config: Hyperparameter configuration from optimizer
        gpus: Number of GPUs per job
        stats_server_uri: Optional stats server URI
        train_overrides: Optional additional overrides

    Returns:
        JobDefinition for training
    """

    overrides = build_train_overrides(
        stats_server_uri=stats_server_uri,
        additional_overrides=train_overrides,
    )

    return JobDefinition(
        run_id=run_id,
        cmd=f"{recipe_module}.{train_entrypoint}",
        type=JobTypes.LAUNCH_TRAINING,
        gpus=gpus,
        nodes=nodes,
        args={"run": run_id, "group": experiment_id},
        overrides=overrides,
        metadata={"group": experiment_id},
    )


def generate_run_id(experiment_id: str, trial_num: int) -> str:
    """Generate a standardized run ID with hash to avoid collisions.

    Args:
        experiment_id: The experiment identifier
        trial_num: The trial number (1-based)

    Returns:
        Formatted run ID like "experiment_id_trial_0001_a1b2c3"
    """
    # Generate a short hash to avoid name collisions
    # Use experiment_id, trial_num, and current time to ensure uniqueness
    hash_input = f"{experiment_id}_{trial_num}_{time.time()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{experiment_id}_trial_{trial_num:04d}_{short_hash}"

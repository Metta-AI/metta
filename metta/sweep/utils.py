"""Utility functions for sweep orchestration."""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional
from metta.sweep.models import JobDefinition, RunInfo



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
    lines.append(f"{prefix}{'=' * 100}")

    # Header
    if include_score:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30} {'Score':<15}")
    else:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30}")
    lines.append(f"{prefix}{'-' * 100}")

    # Rows
    for run in runs:
        # Format run ID
        display_id = get_display_id(run.run_id) if truncate_run_id else run.run_id

        # Format progress
        if run.total_timesteps and run.current_steps is not None:
            progress_pct = (run.current_steps / run.total_timesteps) * 100
            progress_str = f"{run.current_steps:,}/{run.total_timesteps:,} ({progress_pct:.1f}%)"
        elif run.current_steps is not None:
            progress_str = f"{run.current_steps:,}/?"
        else:
            progress_str = "-"

        # Format score
        if include_score:
            score_str = f"{run.observation.score:.4f}" if run.observation else "N/A"
            lines.append(f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30} {score_str:<15}")
        else:
            lines.append(f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30}")

    lines.append(f"{prefix}{'=' * 100}")

    return lines


def get_display_id(run_id: str) -> str:
    """Extract clean display ID from run ID.

    Args:
        run_id: Full run ID (e.g., "sweep_name_trial_0001_a1b2c3")

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
    sweep_id: str,
    stats_server_uri: Optional[str] = None,
    additional_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build evaluation override parameters.

    Args:
        run_id: The run ID for WandB tracking
        sweep_id: The sweep ID for grouping
        stats_server_uri: Optional stats server URI
        additional_overrides: Optional additional overrides to merge

    Returns:
        Dictionary of evaluation overrides
    """
    eval_overrides = additional_overrides.copy() if additional_overrides else {}

    # WandB configuration
    eval_overrides["push_metrics_to_wandb"] = "True"
    eval_overrides["wandb.name"] = run_id
    eval_overrides["wandb.run_id"] = run_id
    eval_overrides["wandb.group"] = sweep_id

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
    overrides = additional_overrides.copy() if additional_overrides else {}

    if stats_server_uri:
        overrides["stats_server_uri"] = stats_server_uri
        overrides["trainer.evaluation.evaluate_remote"] = "True"
        overrides["trainer.evaluation.evaluate_local"] = "False"
        overrides["trainer.evaluation.skip_git_check"] = "True"

    return overrides


def create_eval_job(
    run_id: str,
    sweep_id: str,
    recipe_module: str,
    eval_entrypoint: str,
    stats_server_uri: Optional[str] = None,
    eval_args: Optional[List[str]] = None,
    eval_overrides: Optional[Dict[str, Any]] = None,
) -> "JobDefinition":
    """Create an evaluation job definition.

    Args:
        run_id: The run ID to evaluate
        sweep_id: The sweep ID for grouping
        recipe_module: Module containing the evaluation function
        eval_entrypoint: Name of the evaluation function
        stats_server_uri: Optional stats server URI
        eval_args: Optional positional arguments for evaluation
        eval_overrides: Optional additional overrides

    Returns:
        JobDefinition for evaluation
    """
    from metta.sweep.models import JobDefinition, JobTypes

    overrides = build_eval_overrides(
        run_id=run_id,
        sweep_id=sweep_id,
        stats_server_uri=stats_server_uri,
        additional_overrides=eval_overrides,
    )

    return JobDefinition(
        run_id=run_id,
        cmd=f"{recipe_module}.{eval_entrypoint}",
        type=JobTypes.LAUNCH_EVAL,
        args=eval_args or [],
        overrides=overrides,
        metadata={"policy_uri": f"wandb://metta/{run_id}"},
    )


def create_training_job(
    run_id: str,
    sweep_id: str,
    recipe_module: str,
    train_entrypoint: str,
    config: Dict[str, Any],
    gpus: int = 1,
    nodes: int = 1,
    stats_server_uri: Optional[str] = None,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> "JobDefinition":
    """Create a training job definition.

    Args:
        run_id: The unique run ID
        sweep_id: The sweep ID for grouping
        recipe_module: Module containing the training function
        train_entrypoint: Name of the training function
        config: Hyperparameter configuration from optimizer
        gpus_per_job: Number of GPUs per job
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
        config=config,
        overrides=overrides,
        metadata={"group": sweep_id},
    )


def generate_run_id(sweep_id: str, trial_num: int) -> str:
    """Generate a standardized run ID with hash to avoid collisions.

    Args:
        sweep_id: The sweep identifier
        trial_num: The trial number (1-based)

    Returns:
        Formatted run ID like "sweep_id_trial_0001_a1b2c3"
    """
    # Generate a short hash to avoid name collisions
    # Use sweep_id, trial_num, and current time to ensure uniqueness
    hash_input = f"{sweep_id}_{trial_num}_{time.time()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{sweep_id}_trial_{trial_num:04d}_{short_hash}"

"""Wandb integration for Metta."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def initialize_wandb(
    run_name: str,
    run_dir: str,
    enabled: bool = True,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    job_type: str = "train",
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any]]:  # Returns (wandb_run, wandb_ctx)
    """Initialize Weights & Biases logging with proper configuration.

    This helper function creates the wandb configuration in the format expected
    by WandbContext, handling both Hydra and non-Hydra use cases. It generates
    the same configuration structure that the Hydra pipeline creates via
    configs/wandb/*.yaml files.

    Args:
        run_name: Name of the run (used for group, name, and run_id)
        run_dir: Directory where run data is stored
        enabled: Whether wandb logging is enabled
        project: W&B project name (defaults to env var or "metta")
        entity: W&B entity name (defaults to env var or "metta-research")
        config: Optional configuration dict to log
        job_type: Type of job (e.g., "train", "eval")
        tags: Optional list of tags
        notes: Optional notes for the run

    Returns:
        Tuple of (wandb_run, wandb_ctx):
        - wandb_run: The W&B run object if initialized, None otherwise
        - wandb_ctx: The WandbContext object for cleanup

    Example:
        >>> wandb_run, wandb_ctx = initialize_wandb(
        ...     run_name=dirs.run_name,
        ...     run_dir=dirs.run_dir,
        ...     enabled=not os.environ.get("WANDB_DISABLED"),
        ...     config={"trainer": trainer_config.model_dump()}
        ... )

    Note:
        This function is compatible with the Hydra pipeline used in tools/train.py.
        It creates the same wandb configuration structure that would be loaded from
        configs/wandb/metta_research.yaml or configs/wandb/off.yaml.
    """
    from metta.common.wandb.wandb_context import WandbContext

    # Build wandb config
    if enabled:
        wandb_config = {
            "enabled": True,
            "project": project or os.environ.get("WANDB_PROJECT", "metta"),
            "entity": entity or os.environ.get("WANDB_ENTITY", "metta-research"),
            "group": run_name,
            "name": run_name,
            "run_id": run_name,
            "data_dir": run_dir,
            "job_type": job_type,
            "tags": tags or [],
            "notes": notes or "",
        }
    else:
        wandb_config = {"enabled": False}

    # Build global config for WandbContext
    # This mimics what Hydra would provide
    global_config = {
        "run": run_name,
        "run_dir": run_dir,
        "cmd": job_type,
        "wandb": wandb_config,
    }

    # Add any user-provided config
    if config:
        global_config.update(config)

    # Initialize wandb context
    wandb_ctx = WandbContext(DictConfig(wandb_config), DictConfig(global_config))
    wandb_run = wandb_ctx.__enter__()

    return wandb_run, wandb_ctx


def cleanup_wandb(wandb_ctx: Optional[Any]) -> None:
    """Clean up wandb context if it exists.

    Args:
        wandb_ctx: The WandbContext object returned by initialize_wandb
    """
    if wandb_ctx is not None:
        wandb_ctx.__exit__(None, None, None)

"""Helper functions for Weights & Biases integration."""

import os
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig

from metta.common.wandb.wandb_context import WandbContext


def initialize_wandb(
    run_name: str,
    run_dir: str,
    enabled: bool = True,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    job_type: str = "train",
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> Tuple[Any, WandbContext]:
    """Initialize wandb logging with simplified interface.

    Args:
        run_name: Name for the wandb run
        run_dir: Directory for run data
        enabled: Whether to enable wandb
        project: Wandb project name (defaults to WANDB_PROJECT env var or "metta")
        entity: Wandb entity name (defaults to WANDB_ENTITY env var or "metta-research")
        config: Additional configuration to merge
        job_type: Type of job (e.g., "train", "eval")
        tags: List of tags for the run
        notes: Notes for the run

    Returns:
        Tuple of (wandb_run, wandb_context)
    """
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

    global_config = {
        "run": run_name,
        "run_dir": run_dir,
        "cmd": job_type,
        "wandb": wandb_config,
    }

    if config:
        global_config.update(config)

    wandb_ctx = WandbContext(DictConfig(wandb_config), DictConfig(global_config))
    wandb_run = wandb_ctx.__enter__()

    return wandb_run, wandb_ctx


def cleanup_wandb(wandb_ctx: Optional[WandbContext]) -> None:
    """Clean up wandb context.

    Args:
        wandb_ctx: The wandb context to clean up
    """
    if wandb_ctx is not None:
        wandb_ctx.__exit__(None, None, None)

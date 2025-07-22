"""General utilities for Metta training."""

from typing import Any, Optional

from metta.common.wandb.helpers import abort_requested


def should_run(
    epoch: int,
    interval: int,
    is_master: bool = True,
    force: bool = False,
) -> bool:
    """Check if a periodic task should run based on interval and master status."""
    if not is_master or not interval:
        return False

    if force:
        return True

    return epoch % interval == 0


def check_abort(wandb_run: Optional[Any], trainer_cfg: Any, agent_step: int) -> bool:
    """Check for abort tag in wandb run."""
    if abort_requested(wandb_run, min_interval_sec=60):
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Abort tag detected. Stopping the run.")
        trainer_cfg.total_timesteps = int(agent_step)
        if wandb_run:
            wandb_run.config.update({"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True)
        return True
    return False

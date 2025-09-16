#!/usr/bin/env python3
"""Weights & Biases alert notification platform."""

from devops.skypilot.notifications.utils import format_runtime, get_notification_info
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.utils import send_wandb_alert

logger = getRankAwareLogger(__name__)


def send_wandb_notification(termination_reason: str, job_config: JobConfig) -> bool:
    """Send W&B alert."""
    if not job_config.enable_wandb_alerts or not job_config.metta_run_id:
        logger.debug("W&B alerts disabled or no run ID")
        return False

    info = get_notification_info(termination_reason, job_config)
    runtime = format_runtime(job_config)

    logger.info(f"Sending W&B alert: {info['title']}")

    # Build alert text
    alert_text = info["description"]
    if runtime:
        alert_text += f"\nRuntime: {runtime}"
    alert_text += f"\nNodes: {job_config.total_nodes}"
    alert_text += f"\nTask ID: {job_config.skypilot_task_id or 'N/A'}"

    try:
        send_wandb_alert(
            title=f"{info['emoji']} {info['title']}",
            text=alert_text,
            run_id=job_config.metta_run_id,
            project=job_config.wandb_project or METTA_WANDB_PROJECT,
            entity=job_config.wandb_entity or METTA_WANDB_ENTITY,
        )
        logger.info("W&B alert sent successfully")
        return True
    except Exception as e:
        logger.error(f"W&B alert error: {e}", exc_info=True)
        return False

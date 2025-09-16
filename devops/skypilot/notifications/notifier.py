#!/usr/bin/env python3
"""Notifier for sending notifications to multiple platforms."""

from devops.skypilot.notifications.discord import send_discord_notification
from devops.skypilot.notifications.github import send_github_status
from devops.skypilot.notifications.wandb import send_wandb_notification
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


def send_notifications(termination_reason: str, job_config: JobConfig) -> dict[str, bool]:
    """Send notifications based on termination reason."""
    jc = job_config
    if not jc.is_master:
        logger.debug("Skipping notifications on non-master node")
        return {}

    logger.info(f"Processing notifications for termination reason: {termination_reason}")

    results: dict[str, bool] = {}
    providers = [
        ("discord", jc.discord_webhook_url, send_discord_notification),
        ("github", jc.enable_github_status, send_github_status),
        ("wandb", jc.enable_wandb_alerts, send_wandb_notification),
    ]

    for name, enabled, sender in providers:
        if enabled:
            results[name] = sender(termination_reason, jc)

    # Log summary with emojis
    if results:
        summary = ", ".join(f"{name}={'✅' if success else '❌'}" for name, success in results.items())
        logger.info(f"Notification summary: {summary}")
    else:
        logger.info("Notification summary: none sent")

    return results

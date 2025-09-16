#!/usr/bin/env python3
"""Notification manager for SkyPilot jobs."""

from devops.skypilot.notifications.discord import send_discord_notification
from devops.skypilot.notifications.github import send_github_status
from devops.skypilot.notifications.utils import get_exit_code_description
from devops.skypilot.notifications.wandb import send_wandb_notification
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


class NotificationManager:
    """Manager for SkyPilot job notifications."""

    def __init__(self, job_config: JobConfig):
        """Initialize notification manager with job configuration."""
        self.job_config = job_config

    def log_config(self):
        """Log the current configuration."""
        logger.info("Run Configuration:")
        logger.info(f"  - METTA_RUN_ID: {self.job_config.metta_run_id or ''}")
        logger.info(f"  - SKYPILOT_TASK_ID: {self.job_config.skypilot_task_id or ''}")
        logger.info(f"  - NODE_INDEX: {self.job_config.node_index}")
        logger.info(f"  - IS_MASTER: {self.job_config.is_master}")
        logger.info(f"  - TOTAL_NODES: {self.job_config.total_nodes}")
        logger.info(f"  - HEARTBEAT_TIMEOUT: {self.job_config.heartbeat_timeout or 'NOT SET'}")
        logger.info(f"  - MAX_RUNTIME_HOURS: {self.job_config.max_runtime_hours or 'NOT SET'}")
        logger.info(f"  - RESTART_COUNT: {self.job_config.restart_count}")
        logger.info(f"  - TEST_NCCL: {self.job_config.test_nccl}")
        logger.info(f"  - DISCORD_ENABLED: {bool(self.job_config.discord_webhook_url)}")
        logger.info(f"  - GITHUB_STATUS_ENABLED: {self.job_config.enable_github_status}")
        logger.info(f"  - WANDB_ALERTS_ENABLED: {self.job_config.enable_wandb_alerts}")

    def log_final_summary(self, exit_code: int, termination_reason: str):
        """Log final job summary."""
        logger.info("[SUMMARY] ===== Job Summary =====")
        logger.info(f"[SUMMARY] Metta Run ID: {self.job_config.metta_run_id or 'N/A'}")
        logger.info(f"[SUMMARY] Skypilot Task ID: {self.job_config.skypilot_task_id or 'N/A'}")
        logger.info(f"[SUMMARY] Restart Count: {self.job_config.restart_count or 'N/A'}")
        logger.info(f"[SUMMARY] Exit code: {exit_code}")
        logger.info(f"[SUMMARY] Exit code description: {get_exit_code_description(exit_code)}")
        logger.info(f"[SUMMARY] Termination reason: {termination_reason or 'unknown'}")
        logger.info("[SUMMARY] ======================")

    def send_notifications(self, termination_reason: str) -> dict[str, bool]:
        """Send notifications based on termination reason."""
        if not self.job_config.is_master:
            logger.debug("Skipping notifications on non-master node")
            return {}

        logger.info(f"Processing notifications for termination reason: {termination_reason}")

        results = {}

        # Send to each platform
        if self.job_config.discord_webhook_url:
            results["discord"] = send_discord_notification(termination_reason, self.job_config)

        if self.job_config.enable_github_status:
            results["github"] = send_github_status(termination_reason, self.job_config)

        if self.job_config.enable_wandb_alerts:
            results["wandb"] = send_wandb_notification(termination_reason, self.job_config)

        logger.info(f"Notification summary: {results}")
        return results

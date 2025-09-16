#!/usr/bin/env python3
"""Notifier for sending notifications to multiple platforms."""

from devops.skypilot.notifications.discord import send_discord_notification
from devops.skypilot.notifications.github import send_github_status
from devops.skypilot.notifications.notification import get_exit_code_description
from devops.skypilot.notifications.wandb import send_wandb_notification
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


class Notifier:
    """Manager for SkyPilot job notifications."""

    def __init__(self, job_config: JobConfig) -> None:
        self.job_config = job_config

    def log_config(self) -> None:
        """Log the current configuration."""
        jc = self.job_config
        logger.info("Run Configuration:")
        entries = [
            ("METTA_RUN_ID", jc.metta_run_id),
            ("SKYPILOT_TASK_ID", jc.skypilot_task_id),
            ("NODE_INDEX", jc.node_index),
            ("IS_MASTER", jc.is_master),
            ("TOTAL_NODES", jc.total_nodes),
            ("HEARTBEAT_TIMEOUT", jc.heartbeat_timeout or "NOT SET"),
            ("MAX_RUNTIME_HOURS", jc.max_runtime_hours or "NOT SET"),
            ("RESTART_COUNT", jc.restart_count),
            ("TEST_NCCL", jc.test_nccl),
            ("DISCORD_ENABLED", bool(jc.discord_webhook_url)),
            ("GITHUB_STATUS_ENABLED", jc.enable_github_status),
            ("WANDB_ALERTS_ENABLED", jc.enable_wandb_alerts),
        ]
        for key, val in entries:
            logger.info(f"  - {key}: {val}")

    def log_final_summary(self, exit_code: int, termination_reason: str) -> None:
        """Log final job summary."""
        jc = self.job_config
        logger.info("[SUMMARY] ===== Job Summary =====")
        summary = [
            ("Metta Run ID", jc.metta_run_id),
            ("Skypilot Task ID", jc.skypilot_task_id),
            ("Restart Count", jc.restart_count),
            ("Exit code", exit_code),
            ("Exit code description", get_exit_code_description(exit_code)),
            ("Termination reason", termination_reason),
        ]
        for key, val in summary:
            logger.info(f"[SUMMARY] {key}: {val}")
        logger.info("[SUMMARY] ======================")

    def send_notifications(self, termination_reason: str) -> dict[str, bool]:
        """Send notifications based on termination reason."""
        jc = self.job_config
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

#!/usr/bin/env python3
import re
import signal
from dataclasses import dataclass
from typing import Literal

from devops.skypilot.notifications.discord import send_discord_notification
from devops.skypilot.notifications.github import set_github_status
from devops.skypilot.notifications.wandb import send_wandb_notification
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


def get_exit_code_description(exit_code: int) -> str:
    """Get human-readable description for process exit codes."""
    # Standard exit codes (same across platforms)
    standard_codes = {
        0: "Success",
        1: "General error",
        2: "Misuse of shell command",
        126: "Command cannot execute",
        127: "Command not found",
        255: "Uncaught exception or runtime error",
        -1: "Unknown termination",
    }

    if exit_code in standard_codes:
        return standard_codes[exit_code]

    # Handle signal-based exit codes (128 + signal)
    if 128 < exit_code < 256:
        sig_num = exit_code - 128
        try:
            # Try to get the signal enum
            sig = signal.Signals(sig_num)
            # Get the signal description
            sig_desc = signal.strsignal(sig_num) or sig.name
            return f"Process terminated by {sig.name} - {sig_desc}"
        except (ValueError, AttributeError):
            # Signal number not valid on this platform
            return f"Process terminated by signal {sig_num}"

    return f"Unknown exit code ({exit_code})"


@dataclass
class NotificationConfig:
    """Configuration for a notification across all services."""

    title: str
    description: str
    github_state: Literal["success", "failure", "error", "pending"]
    send_discord: bool = True
    send_wandb: bool = True
    send_github: bool = True


def get_notification_config(termination_reason: str, job_config: JobConfig) -> NotificationConfig | None:
    """Map termination reason to notification configuration."""

    # Static notification mappings
    notification_map: dict[str, NotificationConfig] = {
        "heartbeat_timeout": NotificationConfig(
            title="🚨 SkyPilot Job Heartbeat Timeout",
            description=f"Job failed - no heartbeat for {job_config.heartbeat_timeout} seconds",
            github_state="failure",
        ),
        "max_runtime_reached": NotificationConfig(
            title="✅ SkyPilot Job Completed",
            description=f"Job ran successfully for {job_config.max_runtime_hours} hours",
            github_state="success",
        ),
        "nccl_tests_failed": NotificationConfig(
            title="🔧 SkyPilot Job NCCL Test Error",
            description="NCCL tests failed",
            github_state="error",
        ),
        "rapid_restarts": NotificationConfig(
            title="⚠️ SkyPilot Job Failing Repeatedly",
            description=f"Job terminated after {job_config.restart_count} restarts with average runtime < 3 minutes",
            github_state="failure",
        ),
        "job_completed": NotificationConfig(
            title="✅ SkyPilot Job Completed",
            description="Job ran to completion.",
            github_state="success",
        ),
        "force_restart_test": NotificationConfig(
            title="🔧 SkyPilot Job Restarted",
            description="Job restarted to test automatic recovery.",
            github_state="pending",
        ),
    }

    # Handle dynamic exit code cases
    if termination_reason.startswith("job_failed_"):
        match = re.match(r"job_failed_(-?\d+)", termination_reason)
        if match:
            exit_code = int(match.group(1))
            return NotificationConfig(
                title="🚨 SkyPilot Job Failed",
                description=f"Job failed with code {exit_code}: {get_exit_code_description(exit_code)}",
                github_state="failure",
            )

    return notification_map.get(termination_reason)


def send_notifications(termination_reason: str, job_config: JobConfig) -> dict[str, bool]:
    """Send notifications based on termination reason."""
    if not job_config.is_master:
        logger.debug("Skipping notifications on non-master node")
        return {}

    logger.info(f"Processing notifications for termination reason: {termination_reason}")

    # Get notification configuration
    config = get_notification_config(termination_reason, job_config)
    if not config:
        logger.warning(f"No notification configuration found for termination reason: {termination_reason}")
        return {}

    # Send notifications to each service
    results = {}

    # Discord notification
    if config.send_discord and job_config.enable_discord_notification:
        try:
            results["discord"] = send_discord_notification(
                title=config.title, status_msg=config.description, job_config=job_config
            )
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            results["discord"] = False

    # W&B notification
    if config.send_wandb and job_config.enable_wandb_notification:
        try:
            results["wandb"] = send_wandb_notification(
                title=config.title, description=config.description, job_config=job_config
            )
        except Exception as e:
            logger.error(f"Failed to send W&B notification: {e}")
            results["wandb"] = False

    # GitHub status update
    if config.send_github and job_config.enable_github_status:
        try:
            results["github"] = set_github_status(
                state=config.github_state, description=config.description, job_config=job_config
            )
        except Exception as e:
            logger.error(f"Failed to send GitHub status: {e}")
            results["github"] = False

    # Log summary
    logger.info(
        f"Notification summary: reason={termination_reason}, "
        f"discord={results.get('discord', False)}, "
        f"wandb={results.get('wandb', False)}, "
        f"github={results.get('github', False)}"
    )

    return results

#!/usr/bin/env python3

import re
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger
from metta.common.util.retry import retry_function

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
            from devops.skypilot.notifications.discord import DiscordNotifier

            results["discord"] = DiscordNotifier().send_notification(config, job_config)
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            results["discord"] = False

    # W&B notification
    if config.send_wandb and job_config.enable_wandb_notification:
        try:
            from devops.skypilot.notifications.wandb import WandBNotifier

            results["wandb"] = WandBNotifier().send_notification(config, job_config)
        except Exception as e:
            logger.error(f"Failed to send W&B notification: {e}")
            results["wandb"] = False

    # GitHub status update
    if config.send_github and job_config.enable_github_status:
        try:
            from devops.skypilot.notifications.github import GitHubNotifier

            results["github"] = GitHubNotifier().send_notification(config, job_config)
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


class NotificationBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the notification type for logging."""
        pass

    @abstractmethod
    def get_required_fields(self, job_config: JobConfig) -> Dict[str, Any]:
        """Extract required fields from job_config. Return empty dict if validation fails."""
        pass

    @abstractmethod
    def format_notification(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Format the notification payload."""
        pass

    @abstractmethod
    def send(self, payload: Dict[str, Any]) -> None:
        """Send the actual notification. Should raise exception on failure."""
        pass

    def send_notification(self, notification_config: NotificationConfig, job_config: JobConfig) -> bool:
        """Main entry point for sending notifications."""

        # Get and validate required fields
        fields = self.get_required_fields(job_config)
        missing = [key for key, val in fields.items() if not val]
        if missing:
            logger.warning(f"Skipping {self.name} notification - missing params: {', '.join(missing)}")
            return False

        # Add notification config fields
        fields.update(
            {
                "title": notification_config.title,
                "description": notification_config.description,
                "status_msg": notification_config.description,  # For Discord compatibility
                "state": notification_config.github_state,  # For GitHub
            }
        )

        # Format the notification
        try:
            payload = self.format_notification(fields)
        except Exception as e:
            logger.error(f"Failed to format {self.name} notification: {e}")
            return False

        # Log before sending
        lines = [f"Sending {self.name} notification:"]
        for key, value in payload.items():
            if value is not None:
                lines.append(f"  {key:<12} = {value}")
        logger.info("\n".join(lines))

        # Send with retry
        try:
            retry_function(
                lambda: self.send(payload),
                max_retries=3,
                initial_delay=2.0,
                max_delay=30.0,
            )
            logger.info(f"✅ Successfully sent {self.name} notification")
            return True
        except Exception as e:
            logger.error(f"{self.name} notification failed: {e}")
            return False

    def _calculate_runtime(self, start_time: Optional[int]) -> str:
        if not start_time:
            return ""

        current_time = int(time.time())
        duration = current_time - start_time
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        return f"{hours}h {minutes}m"

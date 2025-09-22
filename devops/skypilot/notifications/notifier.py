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


@dataclass
class Notification:
    """Configuration for a notification across all services."""

    title: str
    description: str
    github_state: Literal["success", "failure", "error", "pending"]


class Notifier(ABC):
    def __init__(self, name: str):
        self._name = name

    def send(self, job_config: JobConfig, termination_reason: str) -> bool:
        """Send the notification."""
        notification = self._notification(termination_reason, job_config)
        payload = self._make_payload(notification, job_config)

        # Log before sending
        logger.info(f"Sending {self._name} notification: \n {notification} \n {payload}")

        # Send with retry
        try:
            retry_function(
                lambda: self._send(payload),
                max_retries=3,
                initial_delay=2.0,
                max_delay=30.0,
            )
            logger.info(f"✅ Successfully sent {self._name} notification")
            return True
        except Exception as e:
            logger.error(f"{self._name} notification failed: {e}")
            return False

    @abstractmethod
    def _send(self, payload: Dict[str, Any]) -> bool: ...

    @abstractmethod
    def _make_payload(self, notification: Notification, job_config: JobConfig) -> Dict[str, Any]: ...

    def _notification(self, termination_reason: str, job_config: JobConfig) -> Notification:
        """Map termination reason to notification configuration."""

        if termination_reason.startswith("job_failed_"):
            match = re.match(r"job_failed_(-?\d+)", termination_reason)
            if match:
                exit_code = int(match.group(1))
                return Notification(
                    title="🚨 SkyPilot Job Failed",
                    description=f"Job failed with code {exit_code}: {self._describe_exit_code(exit_code)}",
                    github_state="failure",
                )
        if termination_reason.startswith("heartbeat_timeout"):
            return Notification(
                title="🚨 SkyPilot Job Heartbeat Timeout",
                description=f"Job failed - no heartbeat for {job_config.heartbeat_timeout} seconds",
                github_state="failure",
            )
        if termination_reason.startswith("max_runtime_reached"):
            return Notification(
                title="✅ SkyPilot Job Completed",
                description=f"Job ran successfully for {job_config.max_runtime_hours} hours",
                github_state="success",
            )
        if termination_reason.startswith("nccl_test_failure"):
            return Notification(
                title="🔧 SkyPilot Job NCCL Test Error",
                description="NCCL tests failed",
                github_state="error",
            )
        if termination_reason.startswith("rapid_restarts"):
            return Notification(
                title="⚠️ SkyPilot Job Failing Repeatedly",
                description=f"Job terminated after {job_config.restart_count} restarts with mean runtime < 3 minutes",
                github_state="failure",
            )
        if termination_reason.startswith("job_completed"):
            return Notification(
                title="✅ SkyPilot Job Completed",
                description="Job ran to completion.",
                github_state="success",
            )
        logger.warning(f"No notification configuration found for termination reason: {termination_reason}")
        return Notification(
            title="🚨 SkyPilot Job Failed",
            description=f"Job failed with for unknown reason: {termination_reason}",
            github_state="failure",
        )

    def _calculate_runtime(self, start_time: Optional[int]) -> str:
        if not start_time:
            return ""

        current_time = int(time.time())
        duration = current_time - start_time
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        return f"{hours}h {minutes}m"

    def _describe_exit_code(self, exit_code: int) -> str:
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


# def send_notifications(termination_reason: str, job_config: JobConfig) -> dict[str, bool]:
#     """Send notifications based on termination reason."""
#     if not job_config.is_master:
#         logger.debug("Skipping notifications on non-master node")
#         return {}

#     logger.info(f"Processing notifications for termination reason: {termination_reason}")

#     # Get notification configuration
#     notification_cfg = self._get_notification_config(termination_reason, job_config)
#     if not notification_cfg:
#         logger.warning(f"No notification configuration found for termination reason: {termination_reason}")
#         return {}

#     # Send notifications to each service
#     results = {}

#     # Discord notification
#     if notification_cfg.send_discord and job_config.enable_discord_notification:
#         try:
#             from devops.skypilot.notifications.discord import DiscordNotifier

#             results["discord"] = DiscordNotifier().send_notification(notification_cfg, job_config)
#         except Exception as e:
#             logger.error(f"Failed to send Discord notification: {e}")
#             results["discord"] = False

#     # W&B notification
#     if notification_cfg.send_wandb and job_config.enable_wandb_notification:
#         try:
#             from devops.skypilot.notifications.wandb import WandBNotifier

#             results["wandb"] = WandBNotifier().send_notification(notification_cfg, job_config)
#         except Exception as e:
#             logger.error(f"Failed to send W&B notification: {e}")
#             results["wandb"] = False

#     # GitHub status update
#     if notification_cfg.send_github and job_config.enable_github_status:
#         try:
#             from devops.skypilot.notifications.github import GitHubNotifier

#             results["github"] = GitHubNotifier().send_notification(notification_cfg, job_config)
#         except Exception as e:
#             logger.error(f"Failed to send GitHub status: {e}")
#             results["github"] = False

#     # Log summary
#     logger.info(
#         f"Notification summary: reason={termination_reason}, "
#         f"discord={results.get('discord', False)}, "
#         f"wandb={results.get('wandb', False)}, "
#         f"github={results.get('github', False)}"
#     )

#     return results

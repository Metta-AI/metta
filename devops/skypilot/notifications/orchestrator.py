import importlib
import re
import signal

from devops.skypilot.notifications.notifier import NotificationConfig
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
            sig = signal.Signals(sig_num)
            sig_desc = signal.strsignal(sig_num) or sig.name
            return f"Process terminated by {sig.name} - {sig_desc}"
        except (ValueError, AttributeError):
            # Signal number not valid on this platform
            return f"Process terminated by signal {sig_num}"

    return f"Unknown exit code ({exit_code})"


def get_notification_config(termination_reason: str, job_config: JobConfig) -> NotificationConfig | None:
    # Collect exit code from job_failed_*
    if termination_reason.startswith("job_failed_"):
        match = re.match(r"job_failed_(-?\d+)", termination_reason)
        if match:
            exit_code = int(match.group(1))
            return NotificationConfig(
                title="🚨 SkyPilot Job Failed",
                description=f"Job failed with code {exit_code}: {get_exit_code_description(exit_code)}",
                github_state="failure",
            )

    # Static notification mappings with channel-specific routing
    notification_map = {
        "heartbeat_timeout": NotificationConfig(
            title="🚨 SkyPilot Job Heartbeat Timeout",
            description=f"Job failed - no heartbeat for {job_config.heartbeat_timeout} seconds",
            github_state="failure",
        ),
        "max_runtime_reached": NotificationConfig(
            title="✅ SkyPilot Job Completed",
            description=f"Job ran successfully for {job_config.max_runtime_hours} hours",
            github_state="success",
            send_discord=False,
            send_wandb=False,
        ),
        "nccl_tests_failed": NotificationConfig(
            title="🔧 SkyPilot Job NCCL Test Error",
            description="NCCL tests failed",
            github_state="error",
            send_wandb=False,
        ),
        "rapid_restarts": NotificationConfig(
            title="⚠️ SkyPilot Job Failing Repeatedly",
            description=f"Job terminated after {job_config.restart_count} restarts with average runtime < 3 minutes",
            github_state="failure",
            # Critical issue - notify all channels
        ),
        "job_completed": NotificationConfig(
            title="✅ SkyPilot Job Completed",
            description="Job ran to completion.",
            github_state="success",
            # Routine completion - skip Discord
            send_discord=False,
        ),
    }

    return notification_map.get(termination_reason)


def send_notifications(termination_reason: str, job_config: JobConfig) -> dict[str, bool]:
    """Send notifications based on termination reason."""
    results: dict[str, bool] = {}

    if not job_config.is_master:
        logger.debug("Skipping notifications on non-master node")
        return results

    logger.info(f"Processing notifications for termination reason: {termination_reason}")

    notification_config = get_notification_config(termination_reason, job_config)
    if not notification_config:
        logger.warning(f"No notification configuration found for termination reason: {termination_reason}")
        return results

    # Lazy imports to avoid circular dependencies
    notifier_mapping = {
        "discord": (
            lambda: importlib.import_module("devops.skypilot.notifications.discord").DiscordNotifier,
            job_config.enable_discord_notification and notification_config.send_discord,
        ),
        "wandb": (
            lambda: importlib.import_module("devops.skypilot.notifications.wandb").WandBNotifier,
            job_config.enable_wandb_notification and notification_config.send_wandb,
        ),
        "github": (
            lambda: importlib.import_module("devops.skypilot.notifications.github").GitHubNotifier,
            job_config.enable_github_status and notification_config.send_github,
        ),
    }

    # Send notifications and build summary
    summary_parts = []

    for channel_name, (get_notifier_class, is_enabled) in notifier_mapping.items():
        if not is_enabled:
            continue

        notifier_class = get_notifier_class()
        notifier = notifier_class()
        success = notifier.send(notification_config, job_config)
        results[channel_name] = success

        status = "sent" if success else "failed"
        summary_parts.append(f"{channel_name}:{status}")

    # Log summary
    if summary_parts:
        logger.info(f"Notifications for {termination_reason}: {' '.join(summary_parts)}")
    else:
        logger.info(f"No notifications sent for {termination_reason}")

    return results

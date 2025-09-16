#!/usr/bin/env python3
"""Notification manager for SkyPilot jobs."""

import re
import time

from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)

NOTIFICATION_METADATA = {
    "heartbeat_timeout": {
        "emoji": "🚨",
        "title": "SkyPilot Job Heartbeat Timeout",
        "state": "failure",
        "description": lambda jc: f"Job failed - no heartbeat for {jc.heartbeat_timeout} seconds",
    },
    "max_runtime_reached": {
        "emoji": "✅",
        "title": "SkyPilot Job Completed",
        "state": "success",
        "description": lambda jc: f"Job ran successfully for {jc.max_runtime_hours} hours",
    },
    "nccl_tests_failed": {
        "emoji": "🔧",
        "title": "SkyPilot Job NCCL Config Error",
        "state": "failure",
        "description": lambda _jc: "NCCL tests failed",
    },
    "rapid_restarts": {
        "emoji": "⚠️",
        "title": "SkyPilot Job Failing Repeatedly",
        "state": "failure",
        "description": lambda jc: f"Job terminated after {jc.restart_count} restarts with average runtime < 3 minutes",
    },
    "job_completed": {
        "emoji": "✅",
        "title": "SkyPilot Job Completed",
        "state": "success",
        "description": lambda _jc: "Job ran to completion",
    },
    "_default": {
        "emoji": "📢",
        "title": "SkyPilot Job Status Update",
        "state": "failure",
        "description": lambda _jc: "Job status changed",
    },
}

_EXIT_CODE_DESCRIPTIONS = {
    1: "General error",
    2: "Misuse of shell command",
    126: "Command cannot execute",
    127: "Command not found",
    129: "Process terminated (SIGHUP)",
    130: "Script terminated by Ctrl-C (SIGINT)",
    131: "Core dumped (SIGQUIT)",
    134: "Assertion failed (SIGABRT)",
    135: "Process terminated (SIGTRAP)",
    136: "Floating point exception (SIGFPE)",
    137: "Process killed (SIGKILL) - likely OOM",
    139: "Segmentation fault (SIGSEGV)",
    140: "Bad system call (SIGSYS)",
    141: "Broken pipe (SIGPIPE)",
    143: "Process terminated (SIGTERM)",
    255: "Uncaught exception or runtime error",
    -1: "Unknown termination",
}


def get_exit_code_description(exit_code: int) -> str:
    """Get human-readable description for process exit codes."""
    return _EXIT_CODE_DESCRIPTIONS.get(exit_code, f"Unknown exit code ({exit_code})")


def get_notification_info(termination_reason: str, job_config: JobConfig) -> dict:
    """Get notification information for a termination reason."""
    # Handle job_failed_<exit_code> pattern
    if termination_reason.startswith("job_failed_"):
        if m := re.match(r"job_failed_(-?\d+)", termination_reason):
            code = int(m.group(1))
            return {
                "emoji": "🚨",
                "title": "SkyPilot Job Failed",
                "state": "failure",
                "description": f"Job failed with code {code}: {get_exit_code_description(code)}",
            }

    # Look up in metadata (with default fallback)
    info = NOTIFICATION_METADATA.get(termination_reason, NOTIFICATION_METADATA["_default"])

    # Resolve callable description
    desc = info["description"]
    return {
        "emoji": info["emoji"],
        "title": info["title"],
        "state": info["state"],
        "description": desc(job_config) if callable(desc) else desc,
    }


def format_runtime(job_config: JobConfig) -> str:
    """Format runtime string from job config."""
    if not job_config.start_time:
        return ""

    seconds = int(time.time() - job_config.start_time)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


class NotificationManager:
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
        from devops.skypilot.notifications.notifier import send_notifications

        return send_notifications(termination_reason, self.job_config)

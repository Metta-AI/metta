#!/usr/bin/env python3
"""Shared utilities for notification system."""

import re
import time

from devops.skypilot.utils.job_config import JobConfig

# Shared notification metadata
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
        "description": lambda jc: "NCCL tests failed",
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
        "description": lambda jc: "Job ran to completion",
    },
}


def get_exit_code_description(exit_code: int) -> str:
    """Get human-readable description for process exit codes."""
    exit_code_descriptions = {
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
    return exit_code_descriptions.get(exit_code, f"Unknown exit code ({exit_code})")


def get_notification_info(termination_reason: str, job_config: JobConfig) -> dict:
    """Get notification information for a termination reason."""
    # Handle job_failed_<exit_code> pattern
    if termination_reason.startswith("job_failed_"):
        match = re.match(r"job_failed_(-?\d+)", termination_reason)
        if match:
            exit_code = int(match.group(1))
            return {
                "emoji": "🚨",
                "title": "SkyPilot Job Failed",
                "state": "failure",
                "description": f"Job failed with code {exit_code}: {get_exit_code_description(exit_code)}",
            }

    # Look up in metadata
    info = NOTIFICATION_METADATA.get(
        termination_reason,
        {
            "emoji": "📢",
            "title": "SkyPilot Job Status Update",
            "state": "failure",
            "description": lambda jc: "Job status changed",
        },
    )

    # Resolve description if it's a lambda
    description = info["description"]
    if callable(description):
        description = description(job_config)

    return {
        "emoji": info["emoji"],
        "title": info["title"],
        "state": info["state"],
        "description": description,
    }


def format_runtime(job_config: JobConfig) -> str:
    """Format runtime string from job config."""
    if job_config.start_time and job_config.start_time != 0:
        try:
            current_time = int(time.time())
            duration = current_time - job_config.start_time
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            return f"{hours}h {minutes}m"
        except (ValueError, TypeError):
            pass
    return ""

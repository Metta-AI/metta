#!/usr/bin/env python3
"""
Notification utilities for SkyPilot jobs including Discord, GitHub, and W&B alerts.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path

from gitta import post_commit_status
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from metta.common.util.discord import send_to_discord
from metta.common.util.log_config import getRankAwareLogger
from metta.common.util.retry import retry_function
from metta.common.wandb.utils import send_wandb_alert

logger = getRankAwareLogger(__name__)

# Configuration from environment
node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
is_master = node_index == 0
total_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None
heartbeat_timeout = int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None
restart_count = int(os.environ.get("RESTART_COUNT", "0"))
test_nccl = os.environ.get("TEST_NCCL", "false").lower() == "true"
discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
enable_discord_posts = bool(discord_webhook_url)
enable_github_status = os.environ.get("ENABLE_GITHUB_STATUS", "false").lower() == "true"
enable_wandb_alerts = os.environ.get("ENABLE_WANDB_ALERTS", "true").lower() == "true"


def master_only(func):
    """Decorator to ensure function only runs on master node."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_master:
            return func(*args, **kwargs)
        else:
            logger.debug(f"Skipping {func.__name__} on non-master node")
            return None

    return wrapper


def log_config():
    """Log the current configuration."""
    logger.info("Run Configuration:")
    logger.info(f"  - METTA_RUN_ID: {os.environ.get('METTA_RUN_ID', '')}")
    logger.info(f"  - SKYPILOT_TASK_ID: {os.environ.get('SKYPILOT_TASK_ID', '')}")
    logger.info(f"  - NODE_INDEX: {node_index}")
    logger.info(f"  - IS_MASTER: {is_master}")
    logger.info(f"  - TOTAL_NODES: {total_nodes}")
    logger.info(f"  - HEARTBEAT_TIMEOUT: {heartbeat_timeout or 'NOT SET'}")

    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE", "") or None
    logger.info(f"  - HEARTBEAT_FILE: {heartbeat_file_path or 'NOT SET'}")

    accumulated_runtime_file_path = os.environ.get("ACCUMULATED_RUNTIME_FILE", "") or None
    logger.info(f"  - ACCUMULATED_RUNTIME_FILE: {accumulated_runtime_file_path or 'NOT SET'}")

    if accumulated_runtime_file_path:
        accumulated_runtime_file = Path(accumulated_runtime_file_path)
        if accumulated_runtime_file.exists():
            try:
                accumulated_runtime_sec = int(accumulated_runtime_file.read_text())
                logger.info(f"  - ACCUMULATED_RUNTIME_SEC: {accumulated_runtime_sec}")
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to load accumulated runtime: {e}")

    logger.info(f"  - MAX_RUNTIME_HOURS: {max_runtime_hours or 'NOT SET'}")
    logger.info(f"  - RESTART_COUNT: {restart_count}")
    logger.info(f"  - TEST_NCCL: {test_nccl}")
    logger.info(
        f"  - DISCORD_ENABLED: {enable_discord_posts} "
        f"{'(webhook URL provided)' if enable_discord_posts else '(no webhook URL)'}"
    )
    logger.info(f"  - GITHUB_STATUS_ENABLED: {enable_github_status}")
    logger.info(f"  - WANDB_ALERTS_ENABLED: {enable_wandb_alerts}")


def log_final_summary(exit_code: int, termination_reason: str):
    """Log final job summary."""
    logger.info("[SUMMARY] ===== Job Summary =====")
    logger.info(f"[SUMMARY] Metta Run ID: {os.environ.get('METTA_RUN_ID', 'N/A')}")
    logger.info(f"[SUMMARY] Skypilot Task ID: {os.environ.get('SKYPILOT_TASK_ID', 'N/A')}")
    logger.info(f"[SUMMARY] Exit code: {exit_code}")
    logger.info(f"[SUMMARY] Termination reason: {termination_reason or 'unknown'}")
    logger.info("[SUMMARY] ======================")
    logger.info(f"Job complete with exit code: {exit_code} (reason: {termination_reason or 'unknown'})")


@master_only
def set_github_status(state: str, description: str):
    """Update GitHub commit status."""
    if not enable_github_status:
        logger.debug("GitHub status updates disabled")
        return

    # Load environment variables
    commit_sha = os.environ.get("METTA_GIT_REF", "").strip()
    token = os.environ.get("GITHUB_PAT", "").strip()
    context = os.environ.get("GITHUB_STATUS_CONTEXT", "Skypilot/E2E").strip()
    wandb_run_id = os.environ.get("METTA_RUN_ID", "").strip()
    job_id = os.environ.get("SKYPILOT_JOB_ID", "").strip()

    # Validate required fields
    if not all([state, description, commit_sha, token]):
        logger.warning(
            f"Skipping GitHub status - missing params: "
            f"state={bool(state)}, desc={bool(description)}, "
            f"sha={bool(commit_sha)}, token={'set' if token else 'missing'}"
        )
        return

    # Build description
    desc = description
    if job_id:
        desc += f" - [ jl {job_id} ]"

    # Build target URL
    target_url = f"https://wandb.ai/metta-research/metta/runs/{wandb_run_id}" if wandb_run_id else None

    logger.info(f"Setting GitHub status: state={state}, context={context}, sha={commit_sha[:8]}")
    logger.debug(f"GitHub status details: description='{desc}', target_url={target_url}")

    try:
        repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
        retry_function(
            lambda: post_commit_status(
                commit_sha=commit_sha,
                state=state,
                repo=repo,
                context=context,
                description=desc,
                target_url=target_url,
                token=token,
            ),
            max_retries=3,
            initial_delay=2.0,
            max_delay=30.0,
            error_prefix=f"Failed to post GitHub status for {commit_sha[:8]}",
        )
        logger.info(f"Successfully set GitHub status: {repo}@{commit_sha[:8]} → {state}")
    except Exception as e:
        logger.error(f"GitHub status update failed: {e}")


@master_only
def send_discord_notification(emoji: str, title: str, status_msg: str, additional_info: str = ""):
    """Send Discord notification."""
    if not enable_discord_posts:
        logger.debug("Discord notifications disabled")
        return

    logger.info(f"Sending Discord notification: {title}")

    try:
        # Validate required environment variables
        required_env_vars = {
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "METTA_GIT_REF": os.getenv("METTA_GIT_REF"),
            "METTA_RUN_ID": os.getenv("METTA_RUN_ID"),
            "TOTAL_NODES": os.getenv("TOTAL_NODES"),
            "JOB_METADATA_DIR": os.getenv("JOB_METADATA_DIR"),
        }

        missing_vars = [k for k, v in required_env_vars.items() if not v]
        if missing_vars:
            logger.error(f"Cannot send Discord notification - missing env vars: {', '.join(missing_vars)}")
            return

        # Calculate runtime if START_TIME is set
        runtime_msg = ""
        start_time = os.getenv("START_TIME")
        if start_time and start_time != "0":
            try:
                current_time = int(time.time())
                duration = current_time - int(start_time)
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                runtime_msg = f"**Runtime**: {hours}h {minutes}m"
            except (ValueError, TypeError):
                logger.warning(f"Invalid START_TIME: {start_time}")

        # Build Discord message
        message_parts = [
            f"{emoji} **{title}**",
            "",
            f"**Repository**: {required_env_vars['GITHUB_REPOSITORY']}",
            f"**Git Ref**: {required_env_vars['METTA_GIT_REF']}",
            f"**Run ID**: {required_env_vars['METTA_RUN_ID'] or 'N/A'}",
            f"**Status**: {status_msg}",
        ]

        if runtime_msg:
            message_parts.append(runtime_msg)

        message_parts.extend(
            [
                f"**Time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"**Nodes**: {required_env_vars['TOTAL_NODES']}",
            ]
        )

        if additional_info:
            message_parts.extend(["", additional_info])

        discord_content = "\n".join(message_parts)

        assert required_env_vars["JOB_METADATA_DIR"]
        # Save to file for debugging
        discord_message_path = os.path.join(required_env_vars["JOB_METADATA_DIR"], "discord_message.txt")
        with open(discord_message_path, "w") as f:
            f.write(discord_content)

        # Send notification
        success = send_to_discord(webhook_url=discord_webhook_url, content=discord_content, suppress_embeds=True)

        if success:
            logger.info(f"Discord notification sent successfully: {title}")
        else:
            logger.warning(f"Discord notification failed: {title}")

    except Exception as e:
        logger.error(f"Discord notification error: {e}", exc_info=True)


@master_only
def send_wandb_alert_notification(state: str, description: str):
    """Send W&B alert notification."""
    if not enable_wandb_alerts:
        logger.debug("W&B alerts disabled")
        return

    # Map states to emojis and titles
    state_info = {
        "success": ("✅", "Job Completed Successfully"),
        "failure": ("❌", "Job Failed"),
        "error": ("🔧", "Job Configuration Error"),
        "pending": ("🔄", "Job Restarting"),
        "timeout": ("🚨", "Job Timeout"),
    }

    # Check if heartbeat timeout (special case)
    if "heartbeat" in description:
        emoji, title = state_info["timeout"]
    else:
        emoji, title = state_info.get(state, ("❓", "Job Status Unknown"))

    logger.info(f"Sending W&B alert: {title}")

    try:
        required_env_vars = {
            "METTA_RUN_ID": os.getenv("METTA_RUN_ID"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
        }

        missing_vars = [k for k, v in required_env_vars.items() if not v]
        if missing_vars:
            logger.error(f"Cannot send W&B alert - missing env vars: {', '.join(missing_vars)}")
            return

        # Build alert text
        alert_text = description

        # Add runtime info if available
        start_time = os.getenv("START_TIME")
        if start_time and start_time != "0":
            try:
                current_time = int(time.time())
                duration = current_time - int(start_time)
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                alert_text += f"\nRuntime: {hours}h {minutes}m"
            except (ValueError, TypeError):
                pass

        # Add additional context
        alert_text += f"\nNodes: {total_nodes}"
        alert_text += f"\nTask ID: {os.environ.get('SKYPILOT_TASK_ID', 'N/A')}"

        # Send the alert
        send_wandb_alert(
            title=f"{emoji} {title}",
            text=alert_text,
            run_id=required_env_vars["METTA_RUN_ID"] or "",
            project=required_env_vars["WANDB_PROJECT"] or "",
            entity=required_env_vars["WANDB_ENTITY"] or "",
        )

        logger.info(f"W&B alert sent successfully: {title}")

    except Exception as e:
        logger.error(f"W&B alert error: {e}", exc_info=True)


@dataclass
class NotificationConfig:
    """Configuration for a notification."""

    emoji: str
    title: str
    description: str
    discord: bool = True
    wandb: bool = True
    github: bool = True
    github_state: str = "failure"


@master_only
def send_notifications(termination_reason: str):
    """Send notifications based on termination reason."""
    logger.info(f"Processing notifications for termination reason: {termination_reason}")

    heartbeat_timeout = globals()["heartbeat_timeout"]
    max_runtime_hours = globals()["max_runtime_hours"]

    notifications = {
        "heartbeat_timeout": NotificationConfig(
            emoji="🚨",
            title="SkyPilot Job Heartbeat Timeout",
            description=f"Job failed - no heartbeat for {heartbeat_timeout} seconds",
        ),
        "max_runtime_reached": NotificationConfig(
            emoji="✅",
            title="SkyPilot Job Completed",
            description=f"Job ran successfully for {max_runtime_hours} hours",
            github_state="success",
        ),
        "nccl_tests_failed": NotificationConfig(
            emoji="🔧",
            title="SkyPilot Job NCCL Config Error",
            description="NCCL tests failed",
        ),
        "force_restart_test": NotificationConfig(
            emoji="🔧",
            title="SkyPilot Job Restarted",
            description="Job restarted to test automatic recovery.",
        ),
    }

    if termination_reason not in notifications:
        logger.warning(f"No notification configuration found for termination reason: {termination_reason}")
        return

    config = notifications[termination_reason]
    notifications_sent = {}

    if config.discord:
        send_discord_notification(config.emoji, config.title, config.description)
        notifications_sent["discord"] = True

    if config.wandb:
        send_wandb_alert_notification("failure" if config.github_state == "failure" else "success", config.description)
        notifications_sent["wandb"] = True

    if config.github:
        set_github_status(config.github_state, config.description)
        notifications_sent["github"] = True

    logger.info(
        f"Notification summary: reason={termination_reason}, "
        f"discord={notifications_sent.get('discord', False)}, "
        f"wandb={notifications_sent.get('wandb', False)}, "
        f"github={notifications_sent.get('github', False)}"
    )

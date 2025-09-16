#!/usr/bin/env python3

import time
from datetime import datetime

from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.discord import send_to_discord
from metta.common.util.log_config import getRankAwareLogger
from metta.common.util.retry import retry_function

logger = getRankAwareLogger(__name__)


def send_discord_notification(title: str, status_msg: str, job_config: JobConfig) -> bool:
    if not job_config.enable_discord_notification:
        logger.debug("Discord notifications disabled")
        return False

    # Extract and validate required fields
    webhook_url = job_config.discord_webhook_url or ""
    repository = job_config.github_repository
    git_ref = job_config.metta_git_ref
    run_id = job_config.metta_run_id
    total_nodes = job_config.total_nodes
    start_time = job_config.start_time

    if not all([title, status_msg, webhook_url, repository, git_ref]):
        logger.warning(
            f"Skipping Discord notification - missing params: "
            f"title={bool(title)}, status={bool(status_msg)}, "
            f"webhook={'set' if webhook_url else 'missing'}, repo={bool(repository)}, "
            f"git_ref={bool(git_ref)}"
        )
        return False

    # Calculate runtime if available
    runtime_str = ""
    if start_time and start_time != 0:
        try:
            current_time = int(time.time())
            duration = current_time - start_time
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            runtime_str = f"{hours}h {minutes}m"
        except (ValueError, TypeError):
            logger.warning(f"Invalid start_time: {start_time}")

    # Build message content
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    message_parts = [
        f"**{title}**",
        "",
        f"**Repository**: {repository}",
        f"**Git Ref**: {git_ref}",
        f"**Run ID**: {run_id or 'N/A'}",
        f"**Status**: {status_msg}",
    ]

    if runtime_str:
        message_parts.append(f"**Runtime**: {runtime_str}")

    message_parts.extend(
        [
            f"**Time**: {timestamp}",
            f"**Nodes**: {total_nodes}",
        ]
    )

    content = "\n".join(message_parts)

    # Log detailed notification info before sending
    logger.info(
        f"Sending Discord notification:\n"
        f"  title       = {title}\n"
        f"  status      = {status_msg}\n"
        f"  repository  = {repository}\n"
        f"  git_ref     = {git_ref}\n"
        f"  run_id      = {run_id or 'N/A'}\n"
        f"  nodes       = {total_nodes}\n"
        f"  runtime     = {runtime_str or 'N/A'}"
    )

    try:
        retry_function(
            lambda: send_to_discord(webhook_url=webhook_url, content=content, suppress_embeds=True),
            max_retries=3,
            initial_delay=2.0,
            max_delay=30.0,
        )
        logger.info(f"✅ Successfully sent Discord notification: {title}")
        return True

    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        return False

#!/usr/bin/env python3
"""Discord notification platform."""

import os
from datetime import datetime

from devops.skypilot.notifications.utils import format_runtime, get_notification_info
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from metta.common.util.discord import send_to_discord
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


def send_discord_notification(termination_reason: str, job_config: JobConfig) -> bool:
    """Send Discord notification."""
    if not job_config.discord_webhook_url:
        logger.debug("Discord notifications disabled - no webhook URL")
        return False

    info = get_notification_info(termination_reason, job_config)
    runtime = format_runtime(job_config)

    logger.info(f"Sending Discord notification: {info['title']}")

    # Build message
    message_parts = [
        f"{info['emoji']} **{info['title']}**",
        "",
        f"**Repository**: {job_config.github_repository or f'{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}'}",
        f"**Git Ref**: {job_config.metta_git_ref or 'unknown'}",
        f"**Run ID**: {job_config.metta_run_id or 'N/A'}",
        f"**Status**: {info['description']}",
    ]

    if runtime:
        message_parts.append(f"**Runtime**: {runtime}")

    message_parts.extend(
        [
            f"**Time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Nodes**: {job_config.total_nodes}",
        ]
    )

    content = "\n".join(message_parts)

    # Save for debugging
    if job_config.job_metadata_dir:
        try:
            debug_path = os.path.join(job_config.job_metadata_dir, "discord_message.txt")
            with open(debug_path, "w") as f:
                f.write(content)
        except Exception:
            pass

    try:
        return send_to_discord(job_config.discord_webhook_url, content, suppress_embeds=True)
    except Exception as e:
        logger.error(f"Discord notification error: {e}", exc_info=True)
        return False

#!/usr/bin/env python3


import os
import time
from datetime import datetime

from devops.skypilot.notifications.config import JobConfig
from metta.common.util.discord import send_to_discord
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


class DiscordNotifier:
    def send_notification(
        self, emoji: str, title: str, status_msg: str, job_config: JobConfig, additional_info: str = ""
    ) -> bool:
        if not job_config.discord_webhook_url:
            logger.debug("Discord notifications disabled - no webhook URL")
            return False

        logger.info(f"Sending Discord notification: {title}")

        try:
            # Validate required fields
            required_fields = {
                "github_repository": job_config.github_repository,
                "metta_git_ref": job_config.metta_git_ref,
                "metta_run_id": job_config.metta_run_id,
                "total_nodes": job_config.total_nodes,
                "job_metadata_dir": job_config.job_metadata_dir,
            }

            missing_fields = [k for k, v in required_fields.items() if not v]
            if missing_fields:
                logger.error(f"Cannot send Discord notification - missing fields: {', '.join(missing_fields)}")
                return False

            # Calculate runtime if start_time is provided
            runtime_msg = ""
            if job_config.start_time and job_config.start_time != 0:
                try:
                    current_time = int(time.time())
                    duration = current_time - job_config.start_time
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    runtime_msg = f"**Runtime**: {hours}h {minutes}m"
                except (ValueError, TypeError):
                    logger.warning(f"Invalid start_time: {job_config.start_time}")

            # Build Discord message
            message_parts = [
                f"{emoji} **{title}**",
                "",
                f"**Repository**: {job_config.github_repository}",
                f"**Git Ref**: {job_config.metta_git_ref}",
                f"**Run ID**: {job_config.metta_run_id or 'N/A'}",
                f"**Status**: {status_msg}",
            ]

            if runtime_msg:
                message_parts.append(runtime_msg)

            message_parts.extend(
                [
                    f"**Time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    f"**Nodes**: {job_config.total_nodes}",
                ]
            )

            if additional_info:
                message_parts.extend(["", additional_info])

            discord_content = "\n".join(message_parts)

            # Save to file for debugging
            if job_config.job_metadata_dir:
                discord_message_path = os.path.join(job_config.job_metadata_dir, "discord_message.txt")
                with open(discord_message_path, "w") as f:
                    f.write(discord_content)

            # Send notification
            success = send_to_discord(
                webhook_url=job_config.discord_webhook_url, content=discord_content, suppress_embeds=True
            )

            if success:
                logger.info(f"Discord notification sent successfully: {title}")
            else:
                logger.warning(f"Discord notification failed: {title}")

            return success

        except Exception as e:
            logger.error(f"Discord notification error: {e}", exc_info=True)
            return False

#!/usr/bin/env python3

from datetime import datetime
from typing import Any, Dict

from devops.skypilot.notifications.notifier import NotificationBase
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.discord import send_to_discord


class DiscordNotifier(NotificationBase):
    @property
    def name(self) -> str:
        return "Discord"

    def get_required_fields(self, job_config: JobConfig) -> Dict[str, Any]:
        """Extract and validate Discord-specific fields."""
        webhook_url = job_config.discord_webhook_url or ""
        repository = job_config.github_repository
        git_ref = job_config.metta_git_ref

        return {
            "webhook_url": webhook_url,
            "repository": repository,
            "git_ref": git_ref,
            "run_id": job_config.metta_run_id,
            "total_nodes": job_config.total_nodes,
            "start_time": job_config.start_time,
        }

    def format_notification(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Format Discord message."""
        title = fields.get("title", "")
        status_msg = fields.get("status_msg", "")

        if not all([title, status_msg]):
            raise ValueError("Missing title or status_msg")

        # Calculate runtime
        runtime_str = self._calculate_runtime(fields["start_time"])

        # Build message
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        message_parts = [
            f"**{title}**",
            "",
            f"**Repository**: {fields['repository']}",
            f"**Git Ref**: {fields['git_ref']}",
            f"**Run ID**: {fields['run_id'] or 'N/A'}",
            f"**Status**: {status_msg}",
        ]

        if runtime_str:
            message_parts.append(f"**Runtime**: {runtime_str}")

        message_parts.extend(
            [
                f"**Time**: {timestamp}",
                f"**Nodes**: {fields['total_nodes']}",
            ]
        )

        return {
            "webhook_url": fields["webhook_url"],
            "content": "\n".join(message_parts),
            "title": title,
            "status": status_msg,
            "repository": fields["repository"],
            "git_ref": fields["git_ref"],
            "run_id": fields["run_id"] or "N/A",
            "nodes": fields["total_nodes"],
            "runtime": runtime_str or "N/A",
        }

    def send(self, payload: Dict[str, Any]) -> None:
        """Send Discord notification."""
        send_to_discord(webhook_url=payload["webhook_url"], content=payload["content"], suppress_embeds=True)

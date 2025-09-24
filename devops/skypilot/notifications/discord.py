import time
from datetime import datetime, timezone
from typing import Any

from devops.skypilot.notifications.notifier import NotificationBase, NotificationConfig
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.discord import send_to_discord


class DiscordNotifier(NotificationBase):
    def __init__(self):
        super().__init__("Discord")

    def _validate_config(self, job_config: JobConfig) -> str | None:
        if not job_config.discord_webhook_url:
            return "Missing required field: webhook_url"
        return None

    def _make_payload(self, notification: NotificationConfig, job_config: JobConfig) -> dict[str, Any]:
        content_parts = [
            f"**{notification.title}**",
            "",
            f"**Repository**: {job_config.github_repository}",
            f"**Git Ref**: {job_config.metta_git_ref or 'N/A'}",
            f"**Run ID**: {job_config.metta_run_id or 'N/A'}",
            f"**Status**: {notification.description}",
        ]

        if job_config.start_time:
            runtime_sec = int(time.time()) - job_config.start_time
            runtime_str = "N/A" if runtime_sec is None else f"{runtime_sec // 3600}h {runtime_sec % 3600 // 60}m"
            content_parts.append(f"**Runtime**: {runtime_str}")

        content_parts.extend(
            [
                f"**Time**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
                f"**Nodes**: {job_config.total_nodes}",
            ]
        )

        payload = {
            "webhook_url": job_config.discord_webhook_url,
            "content": "\n".join(content_parts),
            "suppress_embeds": True,
        }
        return payload

    def _send(self, payload: dict[str, Any]) -> None:
        send_to_discord(**payload)

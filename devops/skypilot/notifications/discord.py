#!/usr/bin/env python3

from datetime import datetime
from typing import Any, Dict

from devops.skypilot.notifications.notifier import Notification, Notifier
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.discord import send_to_discord


class DiscordNotifier(Notifier):
    def __init__(self, webhook_url: str) -> None:
        super().__init__("Discord")
        self._webhook_url = webhook_url

    def _make_payload(self, notification: Notification, job_config: JobConfig) -> Dict[str, Any]:
        return {
            "content": "\n".join(
                [
                    f"**{notification.title}**",
                    "",
                    f"**Repository**: {job_config.github_repository}",
                    f"**Git Ref**: {job_config.metta_git_ref or 'N/A'}",
                    f"**Run ID**: {job_config.metta_run_id or 'N/A'}",
                    f"**Status**: {notification.description}",
                    f"**Time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    f"**Nodes**: {job_config.total_nodes}",
                    f"**Runtime**: {self._calculate_runtime(job_config.start_time)}",
                ]
            ),
        }

    def _send(self, payload: Dict[str, Any]) -> None:
        send_to_discord(webhook_url=self._webhook_url, content=payload["content"], suppress_embeds=True)

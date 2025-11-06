import datetime
import time
import typing

import devops.skypilot.notifications.notifier
import devops.skypilot.utils.job_config
import metta.common.util.discord


class DiscordNotifier(devops.skypilot.notifications.notifier.NotificationBase):
    def __init__(self):
        super().__init__("Discord")

    def _validate_config(self, job_config: devops.skypilot.utils.job_config.JobConfig) -> str | None:
        if not job_config.discord_webhook_url:
            return "Missing required field: webhook_url"
        return None

    def _make_payload(
        self,
        notification: devops.skypilot.notifications.notifier.NotificationConfig,
        job_config: devops.skypilot.utils.job_config.JobConfig,
    ) -> dict[str, typing.Any]:
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
                f"**Time**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
                f"**Nodes**: {job_config.total_nodes}",
            ]
        )

        payload = {
            "webhook_url": job_config.discord_webhook_url,
            "content": "\n".join(content_parts),
            "suppress_embeds": True,
        }
        return payload

    def _send(self, payload: dict[str, typing.Any]) -> None:
        metta.common.util.discord.send_to_discord(**payload)

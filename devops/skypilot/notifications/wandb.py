import time
from typing import Any

from devops.skypilot.notifications.notifier import NotificationBase, NotificationConfig
from devops.skypilot.utils.job_config import JobConfig
from metta.common.wandb.utils import send_wandb_alert


class WandBNotifier(NotificationBase):
    def __init__(self):
        super().__init__("W&B")

    def _validate_config(self, job_config: JobConfig) -> str | None:
        if not job_config.metta_run_id:
            return "Missing required field: metta_run_id"
        if not job_config.wandb_project:
            return "Missing required field: wandb_project"
        if not job_config.wandb_entity:
            return "Missing required field: wandb_entity"
        return None

    def _make_payload(self, notification: NotificationConfig, job_config: JobConfig) -> dict[str, Any]:
        alert_text_parts = [notification.description]

        if job_config.start_time:
            runtime_sec = int(time.time()) - job_config.start_time
            runtime_str = f"{runtime_sec // 3600}h {runtime_sec % 3600 // 60}m"
            alert_text_parts.append(f"Runtime: {runtime_str}")

        alert_text_parts.append(f"Nodes: {job_config.total_nodes}")

        if job_config.skypilot_task_id:
            alert_text_parts.append(f"Task ID: {job_config.skypilot_task_id}")

        payload = {
            "title": notification.title,
            "text": "\n".join(alert_text_parts),
            "run_id": job_config.metta_run_id,
            "project": job_config.wandb_project,
            "entity": job_config.wandb_entity,
        }
        return payload

    def _send(self, payload: dict[str, Any]) -> None:
        send_wandb_alert(**payload)

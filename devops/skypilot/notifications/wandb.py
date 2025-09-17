#!/usr/bin/env python3

from typing import Any, Dict

from devops.skypilot.notifications.notifier import NotificationBase
from devops.skypilot.utils.job_config import JobConfig
from metta.common.wandb.utils import send_wandb_alert


class WandBNotifier(NotificationBase):
    @property
    def name(self) -> str:
        return "W&B"

    def get_required_fields(self, job_config: JobConfig) -> Dict[str, Any]:
        """Extract W&B-specific fields."""
        return {
            "run_id": job_config.metta_run_id or "",
            "project": job_config.wandb_project or "",
            "entity": job_config.wandb_entity or "",
            "start_time": job_config.start_time,
            "total_nodes": job_config.total_nodes,
            "task_id": job_config.skypilot_task_id,
        }

    def format_notification(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Format W&B alert payload."""
        title = fields.get("title", "")
        description = fields.get("description", "")

        if not all([title, description]):
            raise ValueError("Missing title or description")

        # Calculate runtime
        runtime_str = self._calculate_runtime(fields["start_time"])

        # Build alert text
        alert_text_parts = [description]

        if runtime_str:
            alert_text_parts.append(f"Runtime: {runtime_str}")

        alert_text_parts.extend([f"Nodes: {fields['total_nodes']}", f"Task ID: {fields['task_id'] or 'N/A'}"])

        return {
            "title": title,
            "text": "\n".join(alert_text_parts),
            "run_id": fields["run_id"],
            "project": fields["project"],
            "entity": fields["entity"],
            "description": description,
            "nodes": fields["total_nodes"],
            "runtime": runtime_str or "N/A",
        }

    def send(self, payload: Dict[str, Any]) -> None:
        """Send W&B alert."""
        send_wandb_alert(
            title=payload["title"],
            text=payload["text"],
            run_id=payload["run_id"],
            project=payload["project"],
            entity=payload["entity"],
        )

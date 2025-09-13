#!/usr/bin/env python3

import time

from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.utils import send_wandb_alert

logger = getRankAwareLogger(__name__)


class WandbAlertNotifier:
    def send_alert(self, state: str, description: str, job_config: JobConfig) -> bool:
        if not job_config.enable_wandb_alerts:
            logger.debug("W&B alerts disabled")
            return False

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
            # Validate required fields
            required_fields = {
                "metta_run_id": job_config.metta_run_id,
                "wandb_project": job_config.wandb_project,
                "wandb_entity": job_config.wandb_entity,
            }

            missing_fields = [k for k, v in required_fields.items() if not v]
            if missing_fields:
                logger.error(f"Cannot send W&B alert - missing fields: {', '.join(missing_fields)}")
                return False

            # Build alert text
            alert_text = description

            # Add runtime info if available
            if job_config.start_time and job_config.start_time != 0:
                try:
                    current_time = int(time.time())
                    duration = current_time - job_config.start_time
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    alert_text += f"\nRuntime: {hours}h {minutes}m"
                except (ValueError, TypeError):
                    pass

            # Add additional context
            alert_text += f"\nNodes: {job_config.total_nodes}"
            alert_text += f"\nTask ID: {job_config.skypilot_task_id or 'N/A'}"

            # Send the alert
            send_wandb_alert(
                title=f"{emoji} {title}",
                text=alert_text,
                run_id=job_config.metta_run_id or "",
                project=job_config.wandb_project or "",
                entity=job_config.wandb_entity or "",
            )

            logger.info(f"W&B alert sent successfully: {title}")
            return True

        except Exception as e:
            logger.error(f"W&B alert error: {e}", exc_info=True)
            return False

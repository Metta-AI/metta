#!/usr/bin/env python3

import time

from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger
from metta.common.util.retry import retry_function
from metta.common.wandb.utils import send_wandb_alert

logger = getRankAwareLogger(__name__)


def send_wandb_notification(title: str, description: str, job_config: JobConfig) -> bool:
    if not job_config.enable_wandb_notification:
        logger.debug("W&B alerts disabled")
        return False

    # Extract and validate required fields
    run_id = job_config.metta_run_id or ""
    project = job_config.wandb_project or ""
    entity = job_config.wandb_entity or ""
    start_time = job_config.start_time
    total_nodes = job_config.total_nodes
    task_id = job_config.skypilot_task_id

    if not all([title, description, run_id, project, entity]):
        logger.warning(
            f"Skipping W&B alert - missing params: "
            f"title={bool(title)}, desc={bool(description)}, "
            f"run_id={bool(run_id)}, project={bool(project)}, "
            f"entity={bool(entity)}"
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

    # Build alert content
    alert_text_parts = [description]

    if runtime_str:
        alert_text_parts.append(f"Runtime: {runtime_str}")

    alert_text_parts.extend([f"Nodes: {total_nodes}", f"Task ID: {task_id or 'N/A'}"])

    alert_text = "\n".join(alert_text_parts)

    # Log detailed alert info before sending
    logger.info(
        f"Sending W&B alert:\n"
        f"  title       = {title}\n"
        f"  description = {description}\n"
        f"  run_id      = {run_id}\n"
        f"  project     = {project}\n"
        f"  entity      = {entity}\n"
        f"  nodes       = {total_nodes}\n"
        f"  runtime     = {runtime_str or 'N/A'}"
    )

    try:
        retry_function(
            lambda: send_wandb_alert(
                title=title,
                text=alert_text,
                run_id=run_id,
                project=project,
                entity=entity,
            ),
            max_retries=3,
            initial_delay=2.0,
            max_delay=30.0,
        )
        logger.info(f"✅ Successfully sent W&B alert: {title}")
        return True

    except Exception as e:
        logger.error(f"W&B alert failed: {e}")
        return False

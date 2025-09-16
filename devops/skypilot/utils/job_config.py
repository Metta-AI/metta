#!/usr/bin/env python3


from dataclasses import dataclass
from typing import Optional

from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for a SkyPilot job."""

    # Node configuration
    node_index: int = 0
    total_nodes: int = 1
    is_master: bool = True

    # Job identifiers
    metta_run_id: Optional[str] = None
    skypilot_task_id: Optional[str] = None
    skypilot_job_id: Optional[str] = None

    # Runtime configuration
    max_runtime_hours: Optional[float] = None
    heartbeat_timeout: Optional[int] = None
    restart_count: int = 0
    test_nccl: bool = False
    start_time: Optional[int] = None

    # File paths
    heartbeat_file: Optional[str] = None
    accumulated_runtime_file: Optional[str] = None
    accumulated_runtime_sec: Optional[int] = None
    job_metadata_dir: Optional[str] = None

    # Discord
    discord_webhook_url: Optional[str] = None
    enable_discord_notification: bool = False

    # GitHub
    github_repository: Optional[str] = None
    metta_git_ref: Optional[str] = None
    github_pat: Optional[str] = None
    github_status_context: str = "Skypilot/E2E"
    enable_github_status: bool = False

    # W&B
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    enable_wandb_notification: bool = True


def log_job_config(jc: JobConfig):
    """Log the current configuration."""
    logger.info("Run Configuration:")
    logger.info(f"  - METTA_RUN_ID: {jc.metta_run_id or ''}")
    logger.info(f"  - SKYPILOT_TASK_ID: {jc.skypilot_task_id or ''}")
    logger.info(f"  - NODE_INDEX: {jc.node_index}")
    logger.info(f"  - IS_MASTER: {jc.is_master}")
    logger.info(f"  - TOTAL_NODES: {jc.total_nodes}")
    logger.info(f"  - HEARTBEAT_TIMEOUT: {jc.heartbeat_timeout or 'NOT SET'}")
    logger.info(f"  - HEARTBEAT_FILE: {jc.heartbeat_file or 'NOT SET'}")
    logger.info(f"  - ACCUMULATED_RUNTIME_FILE: {jc.accumulated_runtime_file or 'NOT SET'}")

    if jc.accumulated_runtime_sec is not None:
        logger.info(f"  - ACCUMULATED_RUNTIME_SEC: {jc.accumulated_runtime_sec}")

    logger.info(f"  - MAX_RUNTIME_HOURS: {jc.max_runtime_hours or 'NOT SET'}")
    logger.info(f"  - RESTART_COUNT: {jc.restart_count}")
    logger.info(f"  - TEST_NCCL: {jc.test_nccl}")
    logger.info(f"  - DISCORD_ENABLED: {jc.enable_discord_notification}")
    logger.info(f"  - GITHUB_STATUS_ENABLED: {jc.enable_github_status}")
    logger.info(f"  - WANDB_ALERTS_ENABLED: {jc.enable_wandb_notification}")

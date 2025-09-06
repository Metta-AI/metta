#!/usr/bin/env python3


from dataclasses import dataclass
from typing import Optional


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
    test_job_restart: bool = False
    start_time: Optional[int] = None

    # File paths
    heartbeat_file: Optional[str] = None
    accumulated_runtime_file: Optional[str] = None
    accumulated_runtime_sec: Optional[int] = None
    job_metadata_dir: Optional[str] = None

    # Notification settings
    discord_webhook_url: Optional[str] = None
    enable_github_status: bool = False
    enable_wandb_alerts: bool = True

    # Git/GitHub configuration
    github_repository: Optional[str] = None
    metta_git_ref: Optional[str] = None
    github_pat: Optional[str] = None
    github_status_context: str = "Skypilot/E2E"

    # W&B configuration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

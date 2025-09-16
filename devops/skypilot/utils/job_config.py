#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO


@dataclass
class JobConfig:
    """Configuration for a SkyPilot job."""

    # Job identifiers
    metta_run_id: str
    skypilot_task_id: str
    skypilot_job_id: str
    metta_git_ref: str  # Commit SHA

    # Git/GitHub configuration
    github_repository: str = field(default_factory=lambda: f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}")

    # Node configuration
    node_index: int = 0
    total_nodes: int = 1
    is_master: bool = True

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
    job_metadata_dir: str = "/tmp"

    # Notification settings
    discord_webhook_url: Optional[str] = None
    enable_github_status: bool = False
    enable_wandb_alerts: bool = True

    # GitHub configuration
    github_pat: Optional[str] = None
    github_status_context: str = "Skypilot/E2E"

    # W&B configuration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

#!/usr/bin/env python3


import signal
from dataclasses import dataclass, fields
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

    # GitHub configuration
    github_repository: Optional[str] = None
    metta_git_ref: Optional[str] = None

    # Discord configuration
    def __repr__(self):
        return "\n".join([f"{field.name}: {getattr(self, field.name)}" for field in fields(self)])

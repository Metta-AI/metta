#!/usr/bin/env python3
import dataclasses
import typing

import metta.common.util.log_config

logger = metta.common.util.log_config.getRankAwareLogger(__name__)


@dataclasses.dataclass
class JobConfig:
    """Configuration for a SkyPilot job."""

    # Node configuration
    node_index: int = 0
    total_nodes: int = 1
    is_master: bool = True

    # Job identifiers
    metta_run_id: typing.Optional[str] = None
    skypilot_task_id: typing.Optional[str] = None
    skypilot_job_id: typing.Optional[str] = None

    # Runtime configuration
    max_runtime_hours: typing.Optional[float] = None
    heartbeat_timeout: typing.Optional[int] = None
    restart_count: int = 0
    test_nccl: bool = False
    test_job_restart: bool = False
    start_time: typing.Optional[int] = None

    # File paths
    heartbeat_file: typing.Optional[str] = None
    accumulated_runtime_file: typing.Optional[str] = None
    accumulated_runtime_sec: typing.Optional[int] = None
    job_metadata_dir: typing.Optional[str] = None

    # Discord
    discord_webhook_url: typing.Optional[str] = None
    enable_discord_notification: bool = False

    # GitHub
    github_repository: typing.Optional[str] = None
    metta_git_ref: typing.Optional[str] = None
    github_pat: typing.Optional[str] = None
    github_status_context: str = "Skypilot/E2E"
    enable_github_status: bool = False

    # W&B
    wandb_project: typing.Optional[str] = None
    wandb_entity: typing.Optional[str] = None
    enable_wandb_notification: bool = True

    def to_filtered_dict(self, filtered_field_names: list[str] | None = None) -> dict[str, typing.Any]:
        """Convert to dictionary with sensitive fields redacted."""
        if filtered_field_names is None:
            filtered_field_names = ["discord_webhook_url", "github_pat"]

        result = dataclasses.asdict(self)
        for field_name in result:
            if field_name in filtered_field_names and result[field_name] is not None:
                result[field_name] = "REDACTED"
            elif result[field_name] is None:
                result[field_name] = "<not set>"
        return result


def log_job_config(jc: JobConfig):
    """Log job configuration with sensitive values redacted."""
    logger.info("Run Configuration:")
    for field_name, value in jc.to_filtered_dict().items():  # Need .items() here
        logger.info(f"  - {field_name}: {value}")


def __repr__(self):
    """Return a string representation with sensitive fields redacted."""
    # Use the same redaction logic
    safe_dict = self.to_filtered_dict()
    params = ", ".join(f"{k}={v!r}" for k, v in safe_dict.items())
    return f"JobConfig({params})"

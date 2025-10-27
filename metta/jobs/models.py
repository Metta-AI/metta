"""Core job specification models shared across Metta job systems."""

from typing import Any, Literal

from pydantic import Field

from metta.common.config import Config


class RemoteConfig(Config):
    """Infrastructure requirements for remote execution via SkyPilot.

    If JobConfig.remote is None, the job runs locally.
    If JobConfig.remote is set, the job runs remotely with these settings.
    """

    gpus: int = 1
    nodes: int = 1
    spot: bool = True


class JobConfig(Config):
    """Definition of a job; the input to submit a job.

    The presence of `remote` determines execution location:
    - remote=None: Run locally
    - remote=RemoteConfig(...): Run remotely with specified infrastructure
    """

    # Job identification
    name: str

    # Tool maker specification (module path)
    module: str  # e.g., "experiments.recipes.arena.train"

    # Arguments to tool maker function
    args: dict[str, Any] = Field(default_factory=dict)

    # Config overrides (dotted paths like "trainer.total_timesteps")
    overrides: dict[str, Any] = Field(default_factory=dict)

    # Execution settings
    timeout_s: int = 7200

    # Infrastructure requirements (None = local execution)
    remote: RemoteConfig | None = None

    # Job type hint (for specialized behavior)
    job_type: Literal["train", "eval", "task"] = "train"

    # Group (for organizational grouping and batch operations)
    group: str | None = None

    # Metadata (experiment_id, tags, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

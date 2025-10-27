"""Core job specification models shared across Metta job systems."""

from typing import Any, Literal

from pydantic import Field

from metta.common.config import Config


class RemoteConfig(Config):
    """Infrastructure requirements for remote execution via SkyPilot."""

    gpus: int = 1
    nodes: int = 1
    spot: bool = True


class JobConfig(Config):
    """Job specification combining execution config with task parameters.

    remote=None runs locally, remote=RemoteConfig(...) runs remotely.
    """

    name: str
    module: str
    args: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    timeout_s: int = 7200
    remote: RemoteConfig | None = None
    job_type: Literal["train", "eval", "task"] = "train"
    group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

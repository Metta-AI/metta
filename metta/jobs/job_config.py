"""Core job specification models shared across Metta job systems."""

import shlex
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from mettagrid.base_config import Config


class MetricsSource(str, Enum):
    """Source for metrics collection and parsing strategy."""

    NONE = "none"  # No metrics tracking
    WANDB = "wandb"  # Fetch from WandB API
    COGAMES_LOG = "cogames_log"  # Parse from cogames --log-outputs


class RemoteConfig(Config):
    """Infrastructure requirements for remote execution via SkyPilot."""

    gpus: int = 1
    nodes: int = 1
    spot: bool = True


class JobConfig(Config):
    """Job specification combining execution config with task parameters.

    Use either `tool` (for tools/run.py) or `cmd` (for arbitrary commands), not both.
    remote=None runs locally, remote=RemoteConfig(...) runs remotely.
    metrics_source specifies where/how to collect metrics (wandb, cogames_log, or none).
    metrics_to_track lists which metrics to monitor.
    acceptance_criteria stores metric thresholds for validation (e.g., {"overview/sps": (">=", 40000)}).
    """

    name: str
    tool: str | None = None  # Tool for tools/run.py (e.g., "arena.train")
    cmd: str | None = None  # Arbitrary command string (e.g., "pytest tests/")
    args: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    timeout_s: int = 7200
    remote: RemoteConfig | None = None
    group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics_source: MetricsSource = MetricsSource.NONE  # Where/how to collect metrics
    metrics_to_track: list[str] = Field(default_factory=list)  # Metrics to track
    acceptance_criteria: dict[str, tuple[str, float]] | None = None  # Metric thresholds: {metric: (op, value)}

    @model_validator(mode="before")
    @classmethod
    def migrate_module_to_tool(cls, data: Any) -> Any:
        """Backwards compatibility: migrate old 'module' field to 'tool'."""
        if isinstance(data, dict) and "module" in data:
            data["tool"] = data.pop("module")
        return data

    @model_validator(mode="after")
    def validate_tool_or_cmd(self):
        """Validate that exactly one of tool or cmd is provided."""
        if self.tool and self.cmd:
            raise ValueError("Cannot specify both 'tool' and 'cmd'. Use one or the other.")
        if not self.tool and not self.cmd:
            raise ValueError("Must specify either 'tool' or 'cmd'.")

        # If using cmd, args and overrides should be empty
        if self.cmd and (self.args or self.overrides):
            raise ValueError(
                "Cannot use 'args' or 'overrides' with 'cmd'. Include all arguments directly in the command string."
            )

        return self

    def build_command(self) -> list[str]:
        """Build command list for execution.

        Returns list of command parts for subprocess.run/Popen.
        """
        if self.cmd:
            # Parse arbitrary command string into list
            return shlex.split(self.cmd)
        else:
            # Build tools/run.py command from tool + args + overrides
            cmd = ["uv", "run", "./tools/run.py", self.tool]
            for k, v in self.args.items():
                cmd.append(f"{k}={v}")
            for k, v in self.overrides.items():
                cmd.append(f"{k}={v}")
            return cmd

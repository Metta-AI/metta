"""Core job specification models shared across Metta job systems."""

import shlex
from typing import Any, Literal

from pydantic import Field, model_validator

from mettagrid.base_config import Config


class RemoteConfig(Config):
    """Infrastructure requirements for remote execution via SkyPilot."""

    gpus: int = 1
    nodes: int = 1
    spot: bool = True


class AcceptanceCriterion(Config):
    """Single acceptance criterion for a metric.

    Defines a threshold that a metric must meet for the job to be considered successful.
    Example: AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000)
    """

    metric: str
    operator: Literal[">=", ">", "<=", "<", "=="]
    threshold: float

    def evaluate(self, actual: float) -> bool:
        """Evaluate if the actual value meets this criterion."""
        if self.operator == ">=":
            return actual >= self.threshold
        elif self.operator == ">":
            return actual > self.threshold
        elif self.operator == "<=":
            return actual <= self.threshold
        elif self.operator == "<":
            return actual < self.threshold
        elif self.operator == "==":
            return actual == self.threshold
        return False


class JobConfig(Config):
    """Job specification combining execution config with task parameters.

    Use either `module` (for tools/run.py) or `cmd` (for arbitrary commands), not both.
    remote=None runs locally, remote=RemoteConfig(...) runs remotely.
    is_training_job=True enables WandB tracking and run name generation.
    metrics_to_track tracks which WandB metrics to fetch periodically (training jobs only).
    acceptance_criteria defines thresholds that metrics must meet for job success.
    """

    name: str
    module: str | None = None  # Module for tools/run.py (e.g., "arena.train")
    cmd: str | None = None  # Arbitrary command string (e.g., "pytest tests/")
    args: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    timeout_s: int = 7200
    remote: RemoteConfig | None = None
    group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_training_job: bool = False  # Explicit flag for WandB tracking
    metrics_to_track: list[str] = Field(default_factory=list)  # Metrics to fetch from WandB (training only)
    acceptance_criteria: dict[str, tuple[str, float]] | None = None  # Metric thresholds: {metric: (op, value)}

    @model_validator(mode="after")
    def validate_module_or_cmd(self):
        """Validate that exactly one of module or cmd is provided."""
        if self.module and self.cmd:
            raise ValueError("Cannot specify both 'module' and 'cmd'. Use one or the other.")
        if not self.module and not self.cmd:
            raise ValueError("Must specify either 'module' or 'cmd'.")

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
            # Build tools/run.py command from module + args + overrides
            cmd = ["uv", "run", "./tools/run.py", self.module]
            for k, v in self.args.items():
                cmd.append(f"{k}={v}")
            for k, v in self.overrides.items():
                cmd.append(f"{k}={v}")
            return cmd

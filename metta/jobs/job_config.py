"""Core job specification models shared across Metta job systems."""

import shlex
from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from mettagrid.base_config import Config


class MetricsSource(str, Enum):
    """Source for metrics collection and parsing strategy."""

    NONE = "none"  # No metrics tracking
    WANDB = "wandb"  # Fetch from WandB API
    COGAMES_LOG = "cogames_log"  # Parse from cogames --log-outputs
    ARTIFACTS = "artifacts"  # Parse from S3 artifacts (e.g., eval_results.json)


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

    Use either `tool` (for tools/run.py) or `cmd` (for arbitrary commands), not both.
    remote=None runs locally, remote=RemoteConfig(...) runs remotely.
    metrics_source specifies where/how to collect metrics (wandb, cogames_log, artifacts, or none).
    metrics_to_track lists which metrics to monitor.
    acceptance_criteria defines validation thresholds using AcceptanceCriterion objects
        (e.g., [AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000)]).
    artifacts declares S3 artifacts that jobs may produce (e.g., eval_results.json).
    """

    name: str
    tool_maker: str | None = None  # Tool maker path for tools/run.py (e.g., "recipes.prod.arena_basic_easy_shaped.train")
    cmd: str | None = None  # Arbitrary command string (e.g., "pytest tests/")
    args: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    timeout_s: int = 7200
    remote: RemoteConfig | None = None
    group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Metrics configuration
    metrics_source: MetricsSource = MetricsSource.NONE
    metrics_to_track: list[str] = Field(default_factory=list)
    wandb_entity: str = METTA_WANDB_ENTITY
    wandb_project: str = METTA_WANDB_PROJECT

    # Validation
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    dependency_names: list[str] = Field(default_factory=list)

    # Artifacts (for ARTIFACTS metrics source)
    artifacts: list[str] = Field(default_factory=list)  # Expected artifact names (e.g., ["eval_results.json"])

    @model_validator(mode="after")
    def validate_tool_or_cmd(self):
        """Validate that exactly one of tool_maker or cmd is provided."""
        if self.tool_maker and self.cmd:
            raise ValueError("Cannot specify both 'tool_maker' and 'cmd'. Use one or the other.")
        if not self.tool_maker and not self.cmd:
            raise ValueError("Must specify either 'tool_maker' or 'cmd'.")

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
            # Build tools/run.py command from tool_maker + args + overrides
            cmd = ["uv", "run", "./tools/run.py", self.tool_maker]
            for k, v in self.args.items():
                cmd.append(f"{k}={v}")
            for k, v in self.overrides.items():
                cmd.append(f"{k}={v}")
            return cmd

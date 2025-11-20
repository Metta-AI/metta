"""Core job specification models shared across Metta job systems."""

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from mettagrid.base_config import Config


class MetricsSource(str, Enum):
    """Source for metrics collection and parsing strategy."""

    NONE = "none"  # No metrics tracking
    WANDB = "wandb"  # Fetch from WandB API


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

    remote=None runs locally, remote=RemoteConfig(...) runs remotely.
    metrics_source specifies where/how to collect metrics (wandb or none).
    metrics_to_track lists which metrics to monitor.
    acceptance_criteria defines validation thresholds using AcceptanceCriterion objects.
    """

    name: str
    recipe: str  # Recipe path for tools/run.py (e.g., "recipes.prod.arena_basic_easy_shaped.train")
    args: dict[str, Any] = Field(default_factory=dict)
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

    def build_command(self) -> list[str]:
        """Build command list for execution.

        Returns list of command parts for subprocess.run/Popen.
        """
        cmd = ["uv", "run", "./tools/run.py", self.recipe]
        for k, v in self.args.items():
            cmd.append(f"{k}={v}")
        return cmd

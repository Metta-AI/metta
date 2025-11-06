"""Core job specification models shared across Metta job systems."""

import typing

import pydantic

import metta.common.util.constants
import mettagrid.base_config


class RemoteConfig(mettagrid.base_config.Config):
    """Infrastructure requirements for remote execution via SkyPilot."""

    gpus: int = 1
    nodes: int = 1
    spot: bool = True


class AcceptanceCriterion(mettagrid.base_config.Config):
    """Single acceptance criterion for a metric.

    Defines a threshold that a metric must meet for the job to be considered successful.
    Example: AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000)
    """

    metric: str
    operator: typing.Literal[">=", ">", "<=", "<", "=="]
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


class JobConfig(mettagrid.base_config.Config):
    """Job specification combining execution config with task parameters.

    remote=None runs locally, remote=RemoteConfig(...) runs remotely.
    is_training_job=True enables WandB tracking and run name generation.
    metrics_to_track tracks which WandB metrics to fetch periodically (training jobs only).
    acceptance_criteria defines thresholds that metrics must meet for job success.
    """

    name: str
    module: str
    args: list[str] = pydantic.Field(default_factory=list)
    timeout_s: int = 7200
    remote: RemoteConfig | None = None
    group: str | None = None
    metadata: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    is_training_job: bool = False  # Explicit flag for WandB tracking
    metrics_to_track: list[str] = pydantic.Field(default_factory=list)  # Metrics to fetch from WandB (training only)
    wandb_entity: str = (
        metta.common.util.constants.METTA_WANDB_ENTITY
    )  # WandB entity for metrics (defaults to metta project)
    wandb_project: str = (
        metta.common.util.constants.METTA_WANDB_PROJECT
    )  # WandB project for metrics (defaults to metta project)
    acceptance_criteria: list[AcceptanceCriterion] = pydantic.Field(default_factory=list)  # Thresholds for job success
    dependency_names: list[str] = pydantic.Field(default_factory=list)  # Job names this job depends on

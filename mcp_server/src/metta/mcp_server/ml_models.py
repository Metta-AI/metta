"""
Machine Learning and Training Pydantic models for MCP server.

This module contains models for training runs, checkpoints, scorecards,
leaderboards, and other ML-specific functionality.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from metta.app_backend.routes.scorecard_routes import ScorecardData


class TrainingRunList(BaseModel):
    """List of training runs."""

    training_runs: list[dict[str, str | int | float | bool | None]] = Field(description="List of training run objects")
    total_count: int = Field(description="Total number of training runs")


class TrainingRunDetail(BaseModel):
    """Detailed training run information."""

    id: str = Field(description="Training run identifier")
    name: str = Field(description="Training run name")
    description: str | None = Field(description="Training run description")
    tags: list[str] = Field(description="List of tags associated with the run")
    url: str | None = Field(description="URL to the training run")
    created_at: str = Field(description="ISO timestamp when run was created")
    attributes: dict[str, str | int | float | bool | None] = Field(description="Additional run attributes")


class LocalTrainingRunList(BaseModel):
    """List of local training runs."""

    training_runs: list[dict[str, str | int | float | bool | None]] = Field(
        description="List of local training run objects"
    )
    total_runs: int = Field(description="Total number of local training runs")


class CheckpointInfo(BaseModel):
    """Training checkpoint information."""

    checkpoint_path: str = Field(description="Path to the checkpoint file")
    file_size: int = Field(description="Size of checkpoint file in bytes")
    modified_time: str = Field(description="ISO timestamp when checkpoint was last modified")
    model_metadata: dict[str, str | int | float | bool | None] = Field(description="Model configuration and metadata")


class CheckpointInfoError(BaseModel):
    """Error in getting checkpoint info."""

    error: str = Field(description="Error message")
    path: str = Field(description="Path to the checkpoint file")


class PolicyIdMapping(BaseModel):
    """Policy name to ID mapping."""

    policy_mapping: dict[str, str] = Field(description="Dictionary mapping policy names to their IDs")


class MetricsList(BaseModel):
    """List of available metrics."""

    metrics: list[str] = Field(description="List of metric names available for the specified training runs")
    training_run_count: int = Field(description="Number of training runs included in the analysis")
    eval_count: int = Field(description="Number of evaluations considered")


class EvalNamesResult(BaseModel):
    """Result of eval names lookup."""

    eval_names_mapping: dict[str, list[str]] = Field(
        description="Mapping of run/policy IDs to their associated eval names"
    )


class ScorecardDataWithMetadata(BaseModel):
    """Scorecard data for training runs and policies with metadata."""

    primary_metric: str = Field(description="Primary metric used for scoring")
    valid_metrics: list[str] = Field(description="List of all valid metrics available")
    scorecard_data: ScorecardData = Field(description="Backend scorecard data structure")


class Leaderboard(BaseModel):
    """Single leaderboard object."""

    id: str = Field(description="Leaderboard identifier")
    name: str = Field(description="Leaderboard name")
    evals: list[str] = Field(description="List of evaluation names included")
    metric: str = Field(description="Primary metric for ranking")
    start_date: str = Field(description="ISO date when leaderboard period starts")
    created_at: str = Field(description="ISO timestamp when leaderboard was created")


class LeaderboardList(BaseModel):
    """List of leaderboards."""

    leaderboards: list[dict[str, str | int | float | bool | None]] = Field(description="List of leaderboard objects")

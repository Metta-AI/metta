"""Pydantic models for Observatory MCP Server."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from metta.app_backend.routes.stats_routes import (
    EvalsRequest,
    MetricsRequest,
    PoliciesSearchRequest,
    ScorecardRequest,
)
from metta.app_backend.routes.sql_routes import AIQueryRequest, SQLQueryRequest


class SuccessResponse(BaseModel):
    """Success response model."""

    status: Literal["success"] = "success"
    data: Any


class ErrorResponse(BaseModel):
    """Error response model."""

    status: Literal["error"] = "error"
    tool: str | None = None
    message: str
    error_type: str | None = None
    context: str | None = None


class GenerateScorecardInput(BaseModel):
    """Input model for generate_scorecard tool."""

    training_run_ids: list[str] = Field(description="List of training run IDs")
    run_free_policy_ids: list[str] = Field(description="List of run-free policy IDs")
    eval_names: list[str] = Field(description="List of evaluation names (format: 'eval_category/env_name')")
    metric: str = Field(description="Metric to use for scorecard (e.g., 'reward', 'score', 'episode_length')")
    policy_selector: Literal["best", "latest"] = Field(
        default="best", description="Policy selection strategy: 'best' (best performing) or 'latest' (most recent)"
    )

    def to_backend_request(self) -> ScorecardRequest:
        """Convert to backend ScorecardRequest."""
        return ScorecardRequest(
            training_run_ids=self.training_run_ids,
            run_free_policy_ids=self.run_free_policy_ids,
            eval_names=self.eval_names,
            metric=self.metric,
            training_run_policy_selector=self.policy_selector,
        )


class RunSqlQueryInput(BaseModel):
    """Input model for run_sql_query tool."""

    sql: str = Field(description="SQL query string to execute")

    def to_backend_request(self) -> SQLQueryRequest:
        """Convert to backend SQLQueryRequest."""
        return SQLQueryRequest(query=self.sql)


class ListWandbRunsInput(BaseModel):
    """Input model for list_wandb_runs tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    tags: list[str] | None = Field(default=None, description="Filter by tags")
    state: Literal["running", "finished", "crashed", "killed"] | None = Field(default=None, description="Filter by state")
    limit: int = Field(default=50, description="Maximum number of runs to return")


class GetWandbRunInput(BaseModel):
    """Input model for get_wandb_run tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str | None = Field(default=None, description="WandB run ID (preferred if available)")
    run_name: str | None = Field(default=None, description="WandB run name (used if run_id not provided)")


class GetWandbRunMetricsInput(BaseModel):
    """Input model for get_wandb_run_metrics tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    metric_keys: list[str] = Field(description="List of metric names to fetch")
    samples: int | None = Field(default=None, description="Optional limit on number of samples")


class DiscoverWandbRunMetricsInput(BaseModel):
    """Input model for discover_wandb_run_metrics tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")


class GetWandbRunArtifactsInput(BaseModel):
    """Input model for get_wandb_run_artifacts tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")


class GetWandbRunLogsInput(BaseModel):
    """Input model for get_wandb_run_logs tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")


class AnalyzeWandbTrainingProgressionInput(BaseModel):
    """Input model for analyze_wandb_training_progression tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    metric_keys: list[str] = Field(description="List of metric keys to analyze")
    context_window_steps: int = Field(default=1000, description="Number of steps to analyze around center")
    center_step: int | None = Field(default=None, description="Optional center step (defaults to middle of data)")


class CompareWandbRunsInput(BaseModel):
    """Input model for compare_wandb_runs tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_ids: list[str] = Field(description="List of WandB run IDs to compare")
    metric_keys: list[str] = Field(description="List of metric keys to compare")


class AnalyzeWandbLearningCurvesInput(BaseModel):
    """Input model for analyze_wandb_learning_curves tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    metric_keys: list[str] = Field(description="List of metric keys to analyze")
    smoothing_window: int = Field(default=10, description="Window size for smoothing")


class IdentifyWandbCriticalMomentsInput(BaseModel):
    """Input model for identify_wandb_critical_moments tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    metric_keys: list[str] = Field(description="List of metric keys to analyze")
    threshold: float = Field(default=0.1, description="Threshold for detecting significant changes")


class CorrelateWandbMetricsInput(BaseModel):
    """Input model for correlate_wandb_metrics tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    metric_pairs: list[list[str]] = Field(description="List of [metric1, metric2] pairs to correlate")


class AnalyzeWandbBehavioralPatternsInput(BaseModel):
    """Input model for analyze_wandb_behavioral_patterns tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    behavior_categories: list[str] | None = Field(
        default=None, description="Optional list of behavior categories to analyze"
    )


class GenerateWandbTrainingInsightsInput(BaseModel):
    """Input model for generate_wandb_training_insights tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")


class PredictWandbTrainingOutcomeInput(BaseModel):
    """Input model for predict_wandb_training_outcome tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")
    target_metric: str = Field(description="Metric to predict (e.g., 'overview/reward')")
    projection_steps: int = Field(default=1000, description="Number of steps to project forward")


class ListS3CheckpointsInput(BaseModel):
    """Input model for list_s3_checkpoints tool."""

    run_name: str | None = Field(default=None, description="Optional training run name to filter by")
    prefix: str | None = Field(default=None, description="Optional S3 prefix (overrides run_name if both provided)")
    max_keys: int = Field(default=1000, description="Maximum number of objects to return")


class GetS3CheckpointMetadataInput(BaseModel):
    """Input model for get_s3_checkpoint_metadata tool."""

    key: str = Field(description="S3 object key (full path)")


class GetS3CheckpointUrlInput(BaseModel):
    """Input model for get_s3_checkpoint_url tool."""

    key: str = Field(description="S3 object key (full path)")
    expires_in: int = Field(default=3600, description="URL expiration time in seconds")


class ListS3ReplaysInput(BaseModel):
    """Input model for list_s3_replays tool."""

    run_name: str | None = Field(default=None, description="Optional training run name to filter by")
    prefix: str | None = Field(default=None, description="Optional S3 prefix (overrides run_name if both provided)")
    max_keys: int = Field(default=1000, description="Maximum number of objects to return")


class CheckS3ObjectExistsInput(BaseModel):
    """Input model for check_s3_object_exists tool."""

    key: str = Field(description="S3 object key (full path)")


class ListSkypilotJobsInput(BaseModel):
    """Input model for list_skypilot_jobs tool."""

    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"] | None = Field(
        default=None, description="Optional status filter"
    )
    limit: int = Field(default=100, description="Maximum number of jobs to return")


class GetSkypilotJobStatusInput(BaseModel):
    """Input model for get_skypilot_job_status tool."""

    job_id: str = Field(description="Skypilot job ID")


class GetSkypilotJobLogsInput(BaseModel):
    """Input model for get_skypilot_job_logs tool."""

    job_id: str = Field(description="Skypilot job ID")
    tail_lines: int = Field(default=100, description="Number of lines to return")


class AnalyzeS3CheckpointProgressionInput(BaseModel):
    """Input model for analyze_s3_checkpoint_progression tool."""

    run_name: str = Field(description="Training run name")
    prefix: str | None = Field(default=None, description="Optional S3 prefix (overrides run_name if provided)")


class FindBestS3CheckpointInput(BaseModel):
    """Input model for find_best_s3_checkpoint tool."""

    run_name: str = Field(description="Training run name")
    criteria: Literal["latest", "largest", "smallest", "earliest"] = Field(
        default="latest", description="Criteria to use"
    )
    prefix: str | None = Field(default=None, description="Optional S3 prefix (overrides run_name if provided)")


class AnalyzeS3CheckpointUsageInput(BaseModel):
    """Input model for analyze_s3_checkpoint_usage tool."""

    run_name: str | None = Field(default=None, description="Optional training run name to filter by")
    prefix: str | None = Field(default=None, description="Optional S3 prefix (overrides run_name if both provided)")
    time_window_days: int = Field(default=30, description="Time window in days to analyze")


class GetS3CheckpointStatisticsInput(BaseModel):
    """Input model for get_s3_checkpoint_statistics tool."""

    run_name: str | None = Field(default=None, description="Optional training run name to filter by")
    prefix: str | None = Field(default=None, description="Optional S3 prefix (overrides run_name if both provided)")


class CompareS3CheckpointsAcrossRunsInput(BaseModel):
    """Input model for compare_s3_checkpoints_across_runs tool."""

    run_names: list[str] = Field(description="List of training run names to compare")


class AnalyzeSkypilotJobPerformanceInput(BaseModel):
    """Input model for analyze_skypilot_job_performance tool."""

    limit: int = Field(default=100, description="Maximum number of jobs to analyze")


class GetSkypilotResourceUtilizationInput(BaseModel):
    """Input model for get_skypilot_resource_utilization tool."""

    limit: int = Field(default=100, description="Maximum number of jobs to analyze")


class CompareSkypilotJobConfigsInput(BaseModel):
    """Input model for compare_skypilot_job_configs tool."""

    job_ids: list[str] = Field(description="List of job IDs to compare")


class AnalyzeSkypilotJobFailuresInput(BaseModel):
    """Input model for analyze_skypilot_job_failures tool."""

    limit: int = Field(default=100, description="Maximum number of jobs to analyze")


class GetSkypilotJobCostEstimatesInput(BaseModel):
    """Input model for get_skypilot_job_cost_estimates tool."""

    limit: int = Field(default=100, description="Maximum number of jobs to analyze")


class LinkWandbRunToS3CheckpointsInput(BaseModel):
    """Input model for link_wandb_run_to_s3_checkpoints tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")


class LinkWandbRunToSkypilotJobInput(BaseModel):
    """Input model for link_wandb_run_to_skypilot_job tool."""

    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")
    run_id: str = Field(description="WandB run ID")


class LinkS3CheckpointToWandbRunInput(BaseModel):
    """Input model for link_s3_checkpoint_to_wandb_run tool."""

    key: str = Field(description="S3 checkpoint key")
    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")


class LinkS3CheckpointToSkypilotJobInput(BaseModel):
    """Input model for link_s3_checkpoint_to_skypilot_job tool."""

    key: str = Field(description="S3 checkpoint key")


class LinkSkypilotJobToWandbRunsInput(BaseModel):
    """Input model for link_skypilot_job_to_wandb_runs tool."""

    job_id: str = Field(description="Skypilot job ID")
    entity: str = Field(description="WandB entity (user/team)")
    project: str = Field(description="WandB project name")


class LinkSkypilotJobToS3CheckpointsInput(BaseModel):
    """Input model for link_skypilot_job_to_s3_checkpoints tool."""

    job_id: str = Field(description="Skypilot job ID")


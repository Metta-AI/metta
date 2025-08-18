from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import boto3

# from mcp.server import FastMCP
from fastmcp import Context, FastMCP
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, Field
from pydantic.types import Json
from wandb.apis.public.api import Api as WandbApi

from metta.app_backend.clients.scorecard_client import ScorecardClient
from metta.app_backend.routes.dashboard_routes import (
    SavedDashboardDeleteResponse,
    SavedDashboardListResponse,
    SavedDashboardResponse,
)
from metta.app_backend.routes.entity_routes import (
    TrainingRunListResponse,
    TrainingRunPolicyListResponse,
    TrainingRunResponse,
)
from metta.app_backend.routes.eval_task_routes import (
    GitHashesResponse,
    TaskClaimResponse,
    TaskResponse,
    TasksResponse,
    TaskUpdateResponse,
)
from metta.app_backend.routes.leaderboard_routes import (
    LeaderboardDeleteResponse,
    LeaderboardListResponse,
    LeaderboardResponse,
)
from metta.app_backend.routes.score_routes import PolicyScoresData
from metta.app_backend.routes.scorecard_routes import PoliciesResponse
from metta.app_backend.routes.scorecard_routes import ScorecardData as BackendScorecardData
from metta.app_backend.routes.sql_routes import (
    AIQueryResponse,
    TableInfo,
)
from metta.app_backend.routes.sql_routes import (
    SQLQueryResponse as BackendSQLQueryResponse,
)
from metta.app_backend.routes.sql_routes import (
    TableSchema as BackendTableSchema,
)
from metta.app_backend.routes.stats_routes import (
    EpisodeResponse,
    EpochResponse,
    PolicyResponse,
)
from metta.app_backend.routes.stats_routes import (
    TrainingRunResponse as StatsTrainingRunResponse,
)
from metta.app_backend.routes.sweep_routes import (
    RunIdResponse,
    SweepCreateResponse,
    SweepInfo,
)
from metta.app_backend.routes.token_routes import (
    TokenListResponse as BackendTokenListResponse,
)
from metta.app_backend.routes.token_routes import (
    TokenResponse,
)
from metta.mcp_server.config_utils import (
    get_available_config_types,
    list_configs_for_type,
)
from metta.mcp_server.config_utils import (
    get_config_schema as get_config_schema_func,
)
from metta.mcp_server.config_utils import (
    validate_config as validate_config_func,
)
from metta.mcp_server.training_utils import (
    generate_replay_summary_with_llm,
    list_training_runs,
)
from metta.mcp_server.training_utils import (
    get_checkpoint_info as get_checkpoint_info_func,
)
from metta.mcp_server.training_utils import (
    get_training_status as get_training_status_func,
)

backend_url = os.environ.get("METTA_MCP_BACKEND_URL", "http://localhost:8000")
if backend_url != "http://localhost:8000":
    # For production backend, set debug user email to bypass authentication
    os.environ["DEBUG_USER_EMAIL"] = "zachary@stem.ai"
    print(f"Set DEBUG_USER_EMAIL for MCP stdio mode with backend: {backend_url}")

    # Also set production database URI for SQL routes if provided
    production_db_uri = os.environ.get("PRODUCTION_STATS_DB_URI")
    if production_db_uri:
        os.environ["STATS_DB_URI"] = production_db_uri
        print("Set STATS_DB_URI to production database for SQL routes")
    else:
        print("Warning: PRODUCTION_STATS_DB_URI not set, SQL routes will use local database")

CONFIG_PATH = Path(__file__).resolve().parent / "metta.mcp.json"
try:
    CONFIG = json.loads(Path(CONFIG_PATH).read_text())
except Exception:
    CONFIG = {}

# mcp = FastMCP("metta")
mcp = FastMCP("metta")


# Pydantic models for type safety
class ReplaySummarySuccess(BaseModel):
    """Successful replay summary response."""

    replay_path: str = Field(description="Path to the replay file that was analyzed")
    file_size: int = Field(description="Size of the replay file in bytes")
    summary: str = Field(description="AI-generated summary of the replay analysis")
    llm_used: bool = Field(description="Whether LLM was successfully used for summary generation")


class ReplaySummaryError(BaseModel):
    """Error response for replay summary."""

    error: str = Field(description="Error message describing what went wrong")
    path: str = Field(description="Path to the replay file that failed to analyze")


ReplaySummaryResponse = ReplaySummarySuccess | ReplaySummaryError


# Common response models for MCP tools
class GenericSuccess(BaseModel):
    """Generic successful response."""

    success: bool = Field(default=True, description="Operation completed successfully")
    data: dict[str, str | int | float | bool] = Field(description="Response data")


class GenericError(BaseModel):
    """Generic error response."""

    success: bool = Field(default=False, description="Operation failed")
    error: str = Field(description="Error message describing what went wrong")


class SearchResult(BaseModel):
    """Search result with pagination."""

    results: list[dict[str, str | int | float | bool | None]] = Field(description="List of search results")
    total: int = Field(description="Total number of results available")
    limit: int = Field(description="Number of results per page")
    offset: int = Field(description="Offset into the result set")


class TrainingRunList(BaseModel):
    """List of training runs."""

    training_runs: list[dict[str, str | int | float | bool | None]] = Field(description="List of training run objects")
    total_count: int = Field(description="Total number of training runs")


class TableMetadata(BaseModel):
    """Database table metadata."""

    name: str = Field(description="Table name")
    column_count: int = Field(description="Number of columns in the table")
    row_count: int = Field(description="Number of rows in the table")


class TableList(BaseModel):
    """List of database tables."""

    tables: list[TableMetadata] = Field(description="List of table metadata objects")


class SQLQueryResult(BaseModel):
    """SQL query execution result."""

    rows: list[dict[str, str | int | float | bool | None]] = Field(description="Query result rows")
    columns: list[str] = Field(description="Column names in the result")
    row_count: int = Field(description="Number of rows returned")
    execution_time_ms: float = Field(default=0.0, description="Query execution time in milliseconds")


class ConfigList(BaseModel):
    """List of configuration options."""

    configs: list[str] = Field(description="List of available configuration names")
    config_type: str = Field(description="Type of configurations listed")


class CheckpointInfo(BaseModel):
    """Training checkpoint information."""

    checkpoint_path: str = Field(description="Path to the checkpoint file")
    file_size: int = Field(description="Size of checkpoint file in bytes")
    modified_time: str = Field(description="ISO timestamp when checkpoint was last modified")
    model_metadata: dict[str, str | int | float | bool | None] = Field(description="Model configuration and metadata")


class TrainingStatus(BaseModel):
    """Training run status information."""

    run_name: str = Field(description="Name of the training run")
    status: str = Field(description="Current status (running, completed, failed, etc.)")
    progress: dict[str, str | int | float | bool | None] = Field(description="Training progress metrics")
    logs: list[str] = Field(description="Recent log entries")
    checkpoints: list[str] = Field(description="Available checkpoint files")


class EvalNamesResult(BaseModel):
    """Result of eval names lookup."""

    eval_names_mapping: dict[str, list[str]] = Field(
        description="Mapping of run/policy IDs to their associated eval names"
    )


class SQLQueryError(BaseModel):
    """SQL query execution error."""

    error: str = Field(description="Error message describing why the query failed")
    sql_statement: str = Field(description="The SQL statement that caused the error")


class AIQueryResult(BaseModel):
    """AI-generated SQL query result."""

    generated_query: str = Field(description="AI-generated SQL query")
    explanation: str = Field(description="Explanation of what the query does")
    confidence: float = Field(description="AI confidence in the generated query (0.0-1.0)")


class MetricsList(BaseModel):
    """List of available metrics."""

    metrics: list[str] = Field(description="List of metric names available for the specified training runs")
    training_run_count: int = Field(description="Number of training runs included in the analysis")
    eval_count: int = Field(description="Number of evaluations considered")


class ScorecardDataWithMetadata(BaseModel):
    """Scorecard data for training runs and policies with metadata."""

    primary_metric: str = Field(description="Primary metric used for scoring")
    valid_metrics: list[str] = Field(description="List of all valid metrics available")
    scorecard_data: BackendScorecardData = Field(description="Backend scorecard data structure")


class TrainingRunDetail(BaseModel):
    """Detailed training run information."""

    id: str = Field(description="Training run identifier")
    name: str = Field(description="Training run name")
    description: str | None = Field(description="Training run description")
    tags: list[str] = Field(description="List of tags associated with the run")
    url: str | None = Field(description="URL to the training run")
    created_at: str = Field(description="ISO timestamp when run was created")
    attributes: dict[str, str | int | float | bool | None] = Field(description="Additional run attributes")


class PolicyIdMapping(BaseModel):
    """Policy name to ID mapping."""

    policy_mapping: dict[str, str] = Field(description="Dictionary mapping policy names to their IDs")


class WandBRunList(BaseModel):
    """List of Weights & Biases runs."""

    runs: list[dict[str, str | int | float | bool | None]] = Field(description="List of W&B run objects")
    project: str = Field(description="W&B project name")
    entity: str = Field(description="W&B entity/organization name")


class TaskList(BaseModel):
    """List of tasks."""

    tasks: list[dict[str, str | int | float | bool | None]] = Field(description="List of task objects")
    total_count: int = Field(description="Total number of tasks")


class LeaderboardList(BaseModel):
    """List of leaderboards."""

    leaderboards: list[dict[str, str | int | float | bool | None]] = Field(description="List of leaderboard objects")


class Leaderboard(BaseModel):
    """Single leaderboard object."""

    id: str = Field(description="Leaderboard identifier")
    name: str = Field(description="Leaderboard name")
    evals: list[str] = Field(description="List of evaluation names included")
    metric: str = Field(description="Primary metric for ranking")
    start_date: str = Field(description="ISO date when leaderboard period starts")
    created_at: str = Field(description="ISO timestamp when leaderboard was created")


class WandBRun(BaseModel):
    """Single Weights & Biases run object."""

    id: str = Field(description="W&B run ID")
    name: str = Field(description="W&B run name")
    state: str = Field(description="Run state (running, finished, failed, etc.)")
    url: str = Field(description="URL to the W&B run page")


class S3ObjectList(BaseModel):
    """List of S3 objects."""

    objects: list[str] = Field(description="List of S3 object keys")
    bucket: str = Field(description="S3 bucket name")
    prefix: str | None = Field(description="Prefix used to filter objects")


class S3PrefixList(BaseModel):
    """List of S3 prefixes (directories)."""

    prefixes: list[str] = Field(description="List of S3 common prefixes ending with '/'")
    bucket: str = Field(description="S3 bucket name")
    parent_prefix: str | None = Field(description="Parent prefix searched under")


class S3ObjectMetadata(BaseModel):
    """S3 object metadata."""

    content_length: int = Field(description="Size of the object in bytes")
    content_type: str = Field(description="MIME type of the object")
    etag: str = Field(description="Entity tag of the object")
    last_modified: str = Field(description="ISO timestamp when object was last modified")
    storage_class: str | None = Field(description="S3 storage class")
    metadata: dict[str, str] = Field(description="Custom metadata key-value pairs")


class SkypilotStatus(BaseModel):
    """Skypilot job status information."""

    status_output: str = Field(description="Raw output from 'sky status --verbose' command")
    success: bool = Field(description="Whether the command executed successfully")


class UserInfo(BaseModel):
    """Current user information."""

    email: str = Field(description="User's email address")
    authenticated: bool = Field(description="Whether user is properly authenticated")


class GenericOperationResult(BaseModel):
    """Generic operation result."""

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Result message or confirmation")
    data: dict[str, str | int | float | bool | None] = Field(description="Additional result data")


class TokenCreationResult(BaseModel):
    """Result of token creation."""

    token_id: str = Field(description="Unique identifier for the created token")
    token_value: str = Field(description="The actual token string (only shown once)")
    name: str = Field(description="Human-readable name for the token")
    created_at: str = Field(description="ISO timestamp when token was created")


class CliTokenResult(BaseModel):
    """CLI token creation with redirect."""

    redirect_url: str = Field(description="URL to redirect to with token parameter")
    token_id: str = Field(description="Unique identifier for the created token")


class HydraConfigTypes(BaseModel):
    """Available Hydra configuration types."""

    config_types: list[str] = Field(description="List of available configuration types")
    total_types: int = Field(description="Total number of configuration types")


class HydraConfigList(BaseModel):
    """List of configurations for a specific type."""

    config_type: str = Field(description="Configuration type (agent, sim, trainer, etc.)")
    configs: list[str] = Field(description="List of configuration names")
    total_configs: int = Field(description="Total number of configurations")


class HydraConfigSchema(BaseModel):
    """Schema information for a configuration type."""

    config_type: str = Field(description="Configuration type analyzed")
    schema_info: dict[str, str | int | float | bool | list | dict | None] = Field(
        description="Schema structure with field types and descriptions"
    )


class LocalTrainingRunList(BaseModel):
    """List of local training runs."""

    training_runs: list[dict[str, str | int | float | bool | None]] = Field(
        description="List of local training run objects"
    )
    total_runs: int = Field(description="Total number of local training runs")


class ValidationResult(BaseModel):
    """Configuration validation result with detailed info."""

    valid: bool = Field(description="Whether the configuration is valid")
    config_data: dict[str, Any] | list[Any] | None = Field(description="Parsed configuration data if valid")
    errors: list[str] = Field(description="List of validation errors if invalid")
    config_path: str = Field(description="Path to the configuration file validated")


class TaskCreationResult(BaseModel):
    """Result of task creation."""

    task_id: str = Field(description="Unique identifier for the created task")
    policy_id: str = Field(description="Policy ID associated with the task")
    sim_suite: str = Field(description="Simulation suite for the task")
    status: str = Field(description="Initial task status")
    created_at: str = Field(description="ISO timestamp when task was created")


class TaskAssignment(BaseModel):
    """Task assignment information."""

    task_id: str = Field(description="Unique task identifier")
    assignee: str = Field(description="Worker/assignee identifier")
    assigned_at: str = Field(description="ISO timestamp when task was assigned")
    policy_id: str = Field(description="Policy ID for the assigned task")
    status: str = Field(description="Current task status")


class TaskClaimResult(BaseModel):
    """Result of claiming tasks."""

    claimed_tasks: list[str] = Field(description="List of successfully claimed task IDs")
    assignee: str = Field(description="Worker who claimed the tasks")
    total_claimed: int = Field(description="Number of tasks successfully claimed")


class WorkerGitInfo(BaseModel):
    """Git hash information for workers."""

    worker_git_hashes: dict[str, str] = Field(description="Mapping of worker names to their git commit hashes")


class TaskUpdateResult(BaseModel):
    """Result of task status updates."""

    updated_tasks: list[str] = Field(description="List of task IDs that were successfully updated")
    total_updated: int = Field(description="Number of tasks successfully updated")
    errors: list[str] = Field(description="List of any errors encountered during updates")


def _get_backend_url(dev_mode: bool) -> str:
    if backend_url:
        return backend_url
    if dev_mode == "true":
        return CONFIG["resources"]["app_backend_scorecard"]["default_url"]
    else:
        return CONFIG["resources"]["app_backend_scorecard"]["production_url"]


def _parse_json_maybe(value: str) -> Json | None:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            import json

            parsed = json.loads(value)
            return parsed
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, fall through to CSV parsing
            pass
    return None


def _parse_str_or_list(value: str | list[str] | None) -> list[str]:
    """Normalize tool args that may arrive as CSV strings or lists.

    - "a,b,c" -> ["a", "b", "c"]
    - ["a", "b"] -> ["a", "b"]
    - '["a", "b"]' -> ["a", "b"]  (JSON array)
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    maybe_json = _parse_json_maybe(value)
    if maybe_json is not None:
        return [str(item) for item in maybe_json]

    # Parse as CSV
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_optional_str_or_list(value: str | list[str] | None) -> list[str] | None:
    """Normalize tool args that may arrive as CSV strings or lists.

    - "a,b,c" -> ["a", "b", "c"]
    - ["a", "b"] -> ["a", "b"]
    - '["a", "b"]' -> ["a", "b"]  (JSON array)
    - None -> None // []
    """

    if value is None:
        return None

    if isinstance(value, list):
        return value

    # Check if it's a JSON array string
    maybe_json = _parse_json_maybe(value)
    if maybe_json is not None:
        return [str(item) for item in maybe_json]

    return [p.strip() for p in value.split(",") if p.strip()]


@mcp.tool()
async def search_training_runs(
    search: str | None = None,
    policy_type: str | None = None,
    tags: list[str] | str | None = None,
    user_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    dev_mode: bool = False,
) -> PoliciesResponse:
    """Search training runs for policies with pagination support.

    Args:
        search (str | None): Free-text search over names, descriptions, and tags.
        policy_type (str | None): Filter by policy type when provided.
        tags (list[str] | str | None): Filter to items that contain any of these tags.
        user_id (str | None): Filter to items owned by a specific user id.
        limit (int): Maximum number of results to return. Defaults to 100.
        offset (int): Offset into the result set for pagination. Defaults to 0.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        PoliciesResponse: Backend response containing paginated search results
            with matching policy objects, total count, and pagination metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.search_policies(
            search=search,
            policy_type=policy_type,
            tags=_parse_str_or_list(tags) if isinstance(tags, (list, str)) else None,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )


@mcp.tool()
async def get_eval_names_for_training_runs(
    training_run_ids: list[str] | str | None = None,
    run_free_policy_ids: list[str] | str | None = None,
    dev_mode: bool = False,
) -> list[str]:
    """Lookup eval names referenced by the given training runs.

    Args:
        training_run_ids (list[str] | str | None): Training run identifiers to query, or CSV string.
        run_free_policy_ids (list[str] | str | None): Run-free policy identifiers to query, or CSV string.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        list[str]: List of evaluation names available for the specified training runs
            and policies as returned by the backend.
    """
    training_run_ids_list = _parse_str_or_list(training_run_ids)
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_eval_names(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
        )


@mcp.tool()
async def run_sql_query(sql: str, dev_mode: bool = False) -> BackendSQLQueryResponse:
    """Execute a SQL query against the scorecard database API.

    Args:
        sql (str): SQL statement to execute.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        BackendSQLQueryResponse: Backend response containing query execution results
            with rows, columns, row count, and execution time information.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.sql_query(sql)


@mcp.tool()
async def list_sql_tables(dev_mode: bool = False) -> list[TableInfo]:
    """List all available tables in the database (excluding migrations).

    Args:
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        list[TableInfo]: Backend response containing list of table metadata objects
            with names, column counts, and row counts.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.list_tables()


@mcp.tool()
async def get_sql_table_schema(table_name: str, dev_mode: bool = False) -> BackendTableSchema:
    """Get the schema for a specific table.

    Args:
        table_name (str): Name of the table to inspect.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        BackendTableSchema: Backend response containing table schema information
            with table name, column definitions with types and constraints, and table-level constraints.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_table_schema(table_name)


@mcp.tool()
async def artificial_intelligence_sql_query_generation(sql: str, dev_mode: bool = False) -> AIQueryResponse:
    """Generate an SQL query using artificial intelligence according to the
    schema of the database. Very useful for when you get errors or don't know
    the correct syntax.

    Args:
        sql (str): Natural language description or partial SQL that needs to be completed.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        AIQueryResponse: Backend response containing AI-generated query information
            with the complete SQL query, human-readable explanation, and confidence score.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.generate_ai_query(sql)


@mcp.tool()
async def get_available_metrics_for_training_runs(
    training_run_ids: list[str] | str,
    run_free_policy_ids: list[str] | str,
    eval_names: list[str] | str,
    dev_mode: bool = False,
) -> list[str]:
    """Enumerate available metrics for the provided training runs and policies.

    Args:
        training_run_ids (list[str] | str): Training run identifiers to include, or CSV string.
        run_free_policy_ids (list[str] | str): Run-free policy identifiers to include, or CSV string.
        eval_names (list[str] | str): Eval names to consider when computing availability, or CSV string.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        list[str]: List of metric names available across all specified training runs,
            policies, and evaluations as returned by the backend.
    """
    training_run_ids_list = _parse_str_or_list(training_run_ids)
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids)
    eval_names_list = _parse_str_or_list(eval_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_available_metrics(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
            eval_names=eval_names_list,
        )


@mcp.tool()
async def get_scorecard_data(
    search_term: str | None = None,
    restrict_to_policy_ids: list[str] | str | None = None,
    restrict_to_metrics: list[str] | str | None = None,
    restrict_to_policy_names: list[str] | str | None = None,
    restrict_to_eval_names: list[str] | str | None = None,
    policy_selector: Literal["best", "latest"] = "best",
    max_policies: int = 20,
    include_run_free_policies: bool = False,
    dev_mode: bool = False,
) -> ScorecardDataWithMetadata | None:
    """Retrieve a scorecard for the specified training runs and policies

    Args:
        search_term (str | None): Search term to filter policies by name
        restrict_to_policy_ids (list[str] | None): List of policy IDs to include (e.g., ["123", "456"])
        restrict_to_metrics (list[str] | None): List of metrics to include (e.g., ["reward", "heart.get"])
        restrict_to_policy_names (list[str] | None): List of policy name filters (e.g., ["relh.skypilot"])
        restrict_to_eval_names (list[str] | None): List of specific evaluation names to include (["memory/easy"])
        policy_selector (Literal["best", "latest"]): "best" or "latest" policy selection strategy.
        max_policies (int): Maximum number of policies to display
        include_run_free_policies (bool): Whether to include standalone policies

    Returns:
        ScorecardDataWithMetadata | None: Scorecard information containing:
            - primary_metric: The primary metric used for scoring and ranking
            - valid_metrics: List of all available metrics that can be displayed
            - scorecard_data: Detailed scorecard structure with policy names, eval names,
              average scores, and individual cell data including replay URLs and values
            Returns None when no scorecard data is found for the specified criteria.
    """
    restrict_to_policy_ids_list = _parse_optional_str_or_list(restrict_to_policy_ids)
    restrict_to_metrics_list = _parse_optional_str_or_list(restrict_to_metrics)
    restrict_to_policy_names_list = _parse_optional_str_or_list(restrict_to_policy_names)
    restrict_to_eval_names_list = _parse_optional_str_or_list(restrict_to_eval_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_scorecard_data(
            search_term=search_term,
            restrict_to_policy_ids=restrict_to_policy_ids_list,
            restrict_to_policy_names=restrict_to_policy_names_list,
            restrict_to_metrics=restrict_to_metrics_list,
            restrict_to_eval_names=restrict_to_eval_names_list,
            policy_selector=policy_selector,
            max_policies=max_policies,
            include_run_free_policies=include_run_free_policies,
        )
        if result is None:
            return None
        _, scorecard_data, valid_metrics, primary_metric = result
        return ScorecardDataWithMetadata(
            primary_metric=primary_metric,
            valid_metrics=valid_metrics,
            scorecard_data=scorecard_data,
        )


@mcp.tool()
async def list_wandb_runs(
    project: str | None = None,
    entity: str | None = None,
    limit: int = 50,
) -> WandBRunList:
    """List recent Weights & Biases runs for the configured project.

    Args:
        project (str | None): W&B project name. Uses configured default when None.
        entity (str | None): W&B entity (organization or user). Uses configured default when None.
        limit (int): Maximum number of runs to return. Defaults to 50.

    Returns:
        WandBRunList: List of W&B run objects containing run metadata with project and entity info.
            Each run includes ID, name, state, and other identifying metadata.
    """
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    wandb_api = WandbApi()
    runs = wandb_api.runs(f"{entity}/{project}")[:limit]
    run_data = [{"id": run.id, "name": run.name, "state": run.state} for run in runs]
    return WandBRunList(runs=run_data, project=project, entity=entity)


@mcp.tool()
async def get_wandb_run(
    run_name: str,
    project: str | None = None,
    entity: str | None = None,
) -> WandBRun:
    """Get a specific Weights & Biases run by name.

    Args:
        run_name (str): The run name or id to fetch.
        project (str | None): W&B project name. Uses configured default when None.
        entity (str | None): W&B entity (organization or user). Uses configured default when None.

    Returns:
        WandBRun: Single W&B run object containing ID, name, state, and URL.
            Provides direct access to run metadata and link to W&B dashboard.
    """
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    wandb_api = WandbApi()
    run = wandb_api.runs(f"{entity}/{project}/runs/${run_name}")
    return WandBRun(id=run.id, name=run.name, state=run.state, url=run.url)


@mcp.tool()
async def list_s3_objects(
    bucket: str | None = None,
    prefix: str | None = None,
    limit: int = 100,
) -> S3ObjectList:
    """List objects from a configured AWS S3 bucket.

    Args:
        bucket (str | None): Bucket name. Uses the first configured bucket when None.
        prefix (str | None): Key prefix to filter objects.
        limit (int): Maximum number of objects to return. Defaults to 100.

    Returns:
        S3ObjectList: List of S3 object keys with bucket and prefix information.
            Contains object keys, bucket name, and the prefix used for filtering.
    """
    cfg = CONFIG["resources"]["aws_s3"]
    bucket_name = bucket or cfg["buckets"][0]
    s3 = boto3.client("s3", region_name=cfg.get("region"))
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix or "",
        PaginationConfig={"MaxItems": limit},
    )
    keys: list[str] = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return S3ObjectList(objects=keys, bucket=bucket_name, prefix=prefix)


@mcp.tool()
async def list_s3_prefixes(
    bucket: str | None = None,
    prefix: str | None = None,
    limit: int = 100,
) -> S3PrefixList:
    """List common prefixes ("directories") in an S3 bucket under a prefix.

    Args:
        bucket (str | None): Bucket name. Uses first configured bucket when None.
        prefix (str | None): Key prefix to look under.
        limit (int): Maximum number of results to return. Defaults to 100.

    Returns:
        S3PrefixList: List of S3 common prefixes (directory-like structures) with metadata.
            Contains prefixes ending with '/', bucket name, and parent prefix information.
    """
    cfg = CONFIG["resources"]["aws_s3"]
    bucket_name = bucket or cfg["buckets"][0]
    s3 = boto3.client("s3", region_name=cfg.get("region"))
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix or "",
        Delimiter="/",
        PaginationConfig={"MaxItems": limit},
    )
    prefixes: list[str] = []
    for page in page_iterator:
        for cp in page.get("CommonPrefixes", []):
            p = cp.get("Prefix")
            if p:
                prefixes.append(p)
    return S3PrefixList(prefixes=prefixes, bucket=bucket_name, parent_prefix=prefix)


@mcp.tool()
async def get_s3_object_head(
    bucket: str | None,
    key: str,
) -> S3ObjectMetadata:
    """Fetch S3 object metadata (HEAD) without downloading the body.

    Args:
        bucket (str | None): Bucket name. Uses first configured bucket when None.
        key (str): Object key to inspect.

    Returns:
        S3ObjectMetadata: Complete object metadata including size, content type, ETag,
            last modified timestamp, storage class, and custom metadata key-value pairs.
    """
    cfg = CONFIG["resources"]["aws_s3"]
    bucket_name = bucket or cfg["buckets"][0]
    s3 = boto3.client("s3", region_name=cfg.get("region"))
    resp = s3.head_object(Bucket=bucket_name, Key=key)

    def _serialize(value: Any) -> Any:
        try:
            # Serialize datetimes and other objects that expose isoformat
            if hasattr(value, "isoformat"):
                return value.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass
        return value

    return S3ObjectMetadata(
        content_length=resp.get("ContentLength", 0),
        content_type=resp.get("ContentType", ""),
        etag=resp.get("ETag", ""),
        last_modified=_serialize(resp.get("LastModified", "")),
        storage_class=resp.get("StorageClass"),
        metadata=resp.get("Metadata", {}),
    )


@mcp.tool()
async def list_skypilot_jobs() -> SkypilotStatus:
    """List active Skypilot jobs for the logged in user.

    Returns:
        SkypilotStatus: Skypilot job status information containing the raw output
            from 'sky status --verbose' and success indicator. Includes details
            about running jobs, resource usage, and cluster states.
    """
    result = subprocess.run(["sky", "status", "--verbose"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return SkypilotStatus(status_output=result.stdout.strip(), success=True)
    else:
        return SkypilotStatus(
            status_output=f"return code: {result.returncode} - {result.stderr.strip()}", success=False
        )


@mcp.tool()
async def get_wandb_run_url(run_name: str, project: str | None = None, entity: str | None = None) -> str:
    """Get the URL for a specific Weights & Biases run by name.

    Args:
        run_name (str): The run name or id to fetch.
        project (str | None): W&B project name. Uses configured default when None.
        entity (str | None): W&B entity (organization or user). Uses configured default when None.

    Returns:
        str: Direct URL to the W&B run dashboard page where you can view
            training metrics, logs, artifacts, and other run details.
    """
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    wandb_api = WandbApi()
    run = wandb_api.runs(f"{entity}/{project}/runs/${run_name}")
    return run.url


@mcp.tool()
async def whoami(dev_mode: bool = False) -> UserInfo:
    """Get the current user's email and authentication status.

    Returns:
        UserInfo: Current user information including email address and authentication status.
            Shows whether the user is properly authenticated with the backend service.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        try:
            result = await client.validate_authenticated()
            return UserInfo(email=result, authenticated=True)
        except ConnectionError as e:
            return UserInfo(email=f"Error: {e}", authenticated=False)


# Dashboard Tools
@mcp.tool()
async def list_saved_dashboards(dev_mode: bool = False) -> SavedDashboardListResponse:
    """List all saved dashboards.

    Returns:
        SavedDashboardListResponse: Backend response containing list of dashboard objects
            with dashboard metadata, names, types, and configurations.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.list_saved_dashboards()


@mcp.tool()
async def create_saved_dashboard(
    name: str,
    type: str,
    dashboard_state: dict[str, Any],
    description: str | None = None,
    dev_mode: bool = False,
) -> SavedDashboardResponse:
    """Create a new saved dashboard (always creates a new row, even if name is duplicate).

    Returns:
        SavedDashboardResponse: Backend response containing the newly created dashboard object
            with ID, name, type, description, dashboard state configuration, and creation timestamp.
    """
    from metta.app_backend.routes.dashboard_routes import SavedDashboardCreate

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        dashboard_data = SavedDashboardCreate(
            name=name,
            type=type,
            dashboard_state=dashboard_state,
            description=description,
        )
        return await client.create_saved_dashboard(dashboard_data)


@mcp.tool()
async def get_saved_dashboard(dashboard_id: str, dev_mode: bool = False) -> SavedDashboardResponse:
    """Get a specific saved dashboard by ID.

    Returns:
        SavedDashboardResponse: Backend response containing complete dashboard object
            with ID, name, type, description, dashboard state configuration, and creation timestamp.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_saved_dashboard(dashboard_id)


@mcp.tool()
async def update_saved_dashboard(
    dashboard_id: str,
    dashboard_state: dict[str, Any],
    dev_mode: bool = False,
) -> SavedDashboardResponse:
    """Update an existing saved dashboard.

    Returns:
        SavedDashboardResponse: Backend response containing the updated dashboard object
            with new state configuration and other dashboard metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.update_saved_dashboard(dashboard_id, dashboard_state)


@mcp.tool()
async def delete_saved_dashboard(dashboard_id: str, dev_mode: bool = False) -> SavedDashboardDeleteResponse:
    """Delete a saved dashboard.

    Returns:
        SavedDashboardDeleteResponse: Backend response confirming dashboard deletion
            with operation status and any relevant metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.delete_saved_dashboard(dashboard_id)


# Entity Routes (Training Runs)
@mcp.tool()
async def get_training_runs(dev_mode: bool = False) -> TrainingRunListResponse:
    """Get all training runs.

    Returns:
        TrainingRunListResponse: Backend response containing list of training run objects
            with metadata, names, descriptions, tags, and total count.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_training_runs()


@mcp.tool()
async def get_training_run(run_id: str, dev_mode: bool = False) -> TrainingRunResponse:
    """Get a specific training run by ID.

    Returns:
        TrainingRunResponse: Backend response containing detailed training run information
            including ID, name, description, tags, URL, creation timestamp, and attributes.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_training_run(run_id)


@mcp.tool()
async def update_training_run_description(run_id: str, description: str, dev_mode: bool = False) -> TrainingRunResponse:
    """Update the description of a training run.

    Returns:
        TrainingRunResponse: Backend response containing the updated training run
            with new description and all other run metadata preserved.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.update_training_run_description(run_id, description)


@mcp.tool()
async def update_training_run_tags(run_id: str, tags: list[str] | str, dev_mode: bool = False) -> TrainingRunResponse:
    """Update the tags of a training run.

    Returns:
        TrainingRunResponse: Backend response containing the updated training run
            with new tags and all other run metadata preserved.
    """
    tags_list = _parse_str_or_list(tags)
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.update_training_run_tags(run_id, tags_list)


@mcp.tool()
async def get_training_run_policies(run_id: str, dev_mode: bool = False) -> TrainingRunPolicyListResponse:
    """Get policies for a training run with epoch information.

    Returns:
        TrainingRunPolicyListResponse: Backend response containing list of policy objects
            with epoch information, policy IDs, names, epoch numbers, training metrics, and metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_training_run_policies(run_id)


# Task Management Tools
@mcp.tool()
async def create_task(
    policy_id: str,
    sim_suite: str,
    attributes: dict[str, Any] | None = None,
    git_hash: str | None = None,
    env_overrides: dict[str, Any] | None = None,
    dev_mode: bool = False,
) -> TaskResponse:
    """Create Task

    Returns:
        TaskResponse: Backend response containing the newly created task information
            including task ID, policy ID, simulation suite, initial status, and creation timestamp.
    """
    import uuid

    from metta.app_backend.routes.eval_task_routes import TaskCreateRequest

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        task_data = TaskCreateRequest(
            policy_id=uuid.UUID(policy_id),
            sim_suite=sim_suite,
            attributes=attributes or {},
            git_hash=git_hash,
            env_overrides=env_overrides or {},
        )
        return await client.create_task(task_data)


@mcp.tool()
async def get_latest_assigned_task_for_worker(assignee: str, dev_mode: bool = False) -> TaskResponse | None:
    """Get Latest Assigned Task For Worker

    Returns:
        TaskResponse | None: Backend response containing the latest task assigned to the worker,
            or None if no tasks are assigned. Includes task ID, assignee, assignment time,
            policy ID, and current status.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_latest_assigned_task_for_worker(assignee)


@mcp.tool()
async def get_available_tasks(limit: int = 200, dev_mode: bool = False) -> TasksResponse:
    """Get Available Tasks

    Returns:
        TasksResponse: Backend response containing list of available task objects
            with metadata and total count. Tasks represent evaluation jobs that can be claimed by workers.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_available_tasks(limit=limit)


@mcp.tool()
async def claim_tasks(
    tasks: list[str] | str,
    assignee: str,
    dev_mode: bool = False,
) -> TaskClaimResponse:
    """Claim Tasks

    Returns:
        TaskClaimResponse: Backend response containing information about the claimed tasks
            including list of successfully claimed task IDs, assignee name, and total count.
    """
    import uuid

    task_ids = _parse_str_or_list(tasks)
    task_uuids = [uuid.UUID(task_id) for task_id in task_ids]

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.claim_tasks(task_uuids, assignee)


@mcp.tool()
async def get_claimed_tasks(assignee: str | None = None, dev_mode: bool = False) -> TasksResponse:
    """Get Claimed Tasks

    Returns:
        TasksResponse: Backend response containing list of claimed task objects
            with task metadata, assignee information, and status details.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_claimed_tasks(assignee)


@mcp.tool()
async def get_git_hashes_for_workers(assignees: list[str] | str, dev_mode: bool = False) -> GitHashesResponse:
    """Get Git Hashes For Workers

    Returns:
        GitHashesResponse: Backend response containing git hash information
            for the specified workers, mapping assignee names to git commit hashes.
    """
    assignee_list = _parse_str_or_list(assignees)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_git_hashes_for_workers(assignee_list)


@mcp.tool()
async def get_all_tasks(
    limit: int = 500,
    git_hash: str | None = None,
    policy_ids: list[str] | str | None = None,
    sim_suites: list[str] | str | None = None,
    statuses: list[str] | str | None = None,
    dev_mode: bool = False,
) -> TasksResponse:
    """Get All Tasks

    Returns:
        TasksResponse: Backend response containing list of all task objects
            with optional filtering by git hash, policy IDs, simulation suites, and status.
    """
    import uuid

    policy_ids_list = None
    if policy_ids:
        policy_ids_list = [uuid.UUID(pid) for pid in _parse_str_or_list(policy_ids)]

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_all_tasks(
            limit=limit,
            statuses=_parse_optional_str_or_list(statuses),
            git_hash=git_hash,
            policy_ids=policy_ids_list,
            sim_suites=_parse_optional_str_or_list(sim_suites),
        )


@mcp.tool()
async def update_task_statuses(
    updates: dict[str, dict[str, Any]],
    require_assignee: str | None = None,
    dev_mode: bool = False,
) -> TaskUpdateResponse:
    """Update Task Statuses

    Returns:
        TaskUpdateResponse: Backend response containing information about the task updates
            including number of tasks updated, validation results, and any error details.
    """
    import uuid

    # Convert string keys to UUIDs and leave update dicts as-is for the client to handle
    converted_updates = {}
    for task_id_str, update_dict in updates.items():
        task_id = uuid.UUID(task_id_str)
        converted_updates[task_id] = update_dict

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.update_task_statuses(converted_updates, require_assignee)


# Leaderboard Tools
@mcp.tool()
async def list_leaderboards(dev_mode: bool = False) -> LeaderboardListResponse:
    """List all leaderboards for the current user.

    Returns:
        LeaderboardListResponse: Backend response containing list of leaderboard objects
            with metadata, names, evaluation lists, metrics, and configuration details.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.list_leaderboards()


@mcp.tool()
async def create_leaderboard(
    name: str,
    evals: list[str] | str,
    metric: str,
    start_date: str,
    dev_mode: bool = False,
) -> LeaderboardResponse:
    """Create a new leaderboard.

    Returns:
        LeaderboardResponse: Backend response containing the created leaderboard
            with ID, name, evaluations, metric, start date, and timestamps.
    """
    from metta.app_backend.routes.leaderboard_routes import LeaderboardCreateOrUpdate

    eval_list = _parse_str_or_list(evals)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        leaderboard_data = LeaderboardCreateOrUpdate(
            name=name,
            evals=eval_list,
            metric=metric,
            start_date=start_date,
        )
        return await client.create_leaderboard(leaderboard_data)


@mcp.tool()
async def get_leaderboard(leaderboard_id: str, dev_mode: bool = False) -> LeaderboardResponse:
    """Get a specific leaderboard by ID.

    Returns:
        LeaderboardResponse: Backend response containing complete leaderboard object
            with ID, name, evaluation list, primary metric, start date, and creation timestamp.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_leaderboard(leaderboard_id)


@mcp.tool()
async def update_leaderboard(
    leaderboard_id: str,
    name: str,
    evals: list[str] | str,
    metric: str,
    start_date: str,
    dev_mode: bool = False,
) -> LeaderboardResponse:
    """Update a leaderboard.

    Returns:
        LeaderboardResponse: Backend response containing the updated leaderboard
            with ID, name, evaluations, metric, start date, and timestamps.
    """
    from metta.app_backend.routes.leaderboard_routes import LeaderboardCreateOrUpdate

    eval_list = _parse_str_or_list(evals)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        leaderboard_data = LeaderboardCreateOrUpdate(
            name=name,
            evals=eval_list,
            metric=metric,
            start_date=start_date,
        )
        return await client.update_leaderboard(leaderboard_id, leaderboard_data)


@mcp.tool()
async def delete_leaderboard(leaderboard_id: str, dev_mode: bool = False) -> LeaderboardDeleteResponse:
    """Delete a leaderboard.

    Returns:
        LeaderboardDeleteResponse: Backend response confirming leaderboard deletion
            with operation status and any relevant metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.delete_leaderboard(leaderboard_id)


# Stats and Metrics Tools
@mcp.tool()
async def get_policy_ids(policy_names: list[str] | str, dev_mode: bool = False) -> PolicyIdMapping:
    """Get policy IDs for given policy names.

    Returns:
        PolicyIdMapping: Dictionary mapping policy names to their corresponding IDs.
            This allows you to look up the UUID for a policy given its human-readable name.
    """
    policy_names_list = _parse_str_or_list(policy_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_policy_ids(policy_names_list)
        raw_data = result.model_dump()
        return PolicyIdMapping(policy_mapping=raw_data.get("policy_mapping", {}))


@mcp.tool()
async def create_training_run(
    name: str,
    attributes: dict[str, str] | None = None,
    url: str | None = None,
    description: str | None = None,
    tags: list[str] | str | None = None,
    dev_mode: bool = False,
) -> StatsTrainingRunResponse:
    """Create a new training run.

    Returns:
        StatsTrainingRunResponse: Backend response containing the created training run
            with ID, name, description, tags, URL, and creation metadata.
    """
    from metta.app_backend.routes.stats_routes import TrainingRunCreate

    tags_list = _parse_optional_str_or_list(tags) if tags else None

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        training_run_data = TrainingRunCreate(
            name=name,
            attributes=attributes or {},
            url=url,
            description=description,
            tags=tags_list,
        )
        return await client.create_training_run(training_run_data)


@mcp.tool()
async def create_epoch(
    run_id: str,
    start_training_epoch: int,
    end_training_epoch: int,
    attributes: dict[str, str] | None = None,
    dev_mode: bool = False,
) -> EpochResponse:
    """Create a new policy epoch.

    Returns:
        EpochResponse: Backend response containing the created epoch
            with ID, training epoch range, run ID, and creation metadata.
    """
    from metta.app_backend.routes.stats_routes import EpochCreate

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        epoch_data = EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        return await client.create_epoch(run_id, epoch_data)


@mcp.tool()
async def create_policy(
    name: str,
    description: str | None = None,
    url: str | None = None,
    epoch_id: str | None = None,
    dev_mode: bool = False,
) -> PolicyResponse:
    """Create a new policy."""
    import uuid

    from metta.app_backend.routes.stats_routes import PolicyCreate

    epoch_uuid = uuid.UUID(epoch_id) if epoch_id else None

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        policy_data = PolicyCreate(
            name=name,
            description=description,
            url=url,
            epoch_id=epoch_uuid,
        )
        return await client.create_policy(policy_data)


@mcp.tool()
async def record_episode(
    agent_policies: dict[int, str],
    agent_metrics: dict[int, dict[str, float]],
    primary_policy_id: str,
    stats_epoch: str | None = None,
    eval_name: str | None = None,
    simulation_suite: str | None = None,
    replay_url: str | None = None,
    attributes: dict[str, Any] | None = None,
    eval_task_id: str | None = None,
    tags: list[str] | str | None = None,
    dev_mode: bool = False,
) -> EpisodeResponse:
    """Record a new episode with agent policies and metrics."""
    import uuid

    from metta.app_backend.routes.stats_routes import EpisodeCreate

    # Convert string policy IDs to UUIDs in agent_policies dict
    agent_policies_uuids = {agent_id: uuid.UUID(policy_id) for agent_id, policy_id in agent_policies.items()}

    tags_list = _parse_optional_str_or_list(tags) if tags else None

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        episode_data = EpisodeCreate(
            agent_policies=agent_policies_uuids,
            agent_metrics=agent_metrics,
            primary_policy_id=uuid.UUID(primary_policy_id),
            stats_epoch=uuid.UUID(stats_epoch) if stats_epoch else None,
            eval_name=eval_name,
            simulation_suite=simulation_suite,
            replay_url=replay_url,
            attributes=attributes or {},
            eval_task_id=uuid.UUID(eval_task_id) if eval_task_id else None,
            tags=tags_list,
        )
        return await client.record_episode(episode_data)


# Sweep Tools
@mcp.tool()
async def create_sweep(
    sweep_name: str,
    project: str,
    entity: str,
    wandb_sweep_id: str,
    dev_mode: bool = False,
) -> SweepCreateResponse:
    """Initialize a new sweep or return existing sweep info (idempotent)."""
    from metta.app_backend.routes.sweep_routes import SweepCreateRequest

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        request_data = SweepCreateRequest(
            project=project,
            entity=entity,
            wandb_sweep_id=wandb_sweep_id,
        )
        return await client.create_sweep(sweep_name, request_data)


@mcp.tool()
async def get_sweep(sweep_name: str, dev_mode: bool = False) -> SweepInfo:
    """Get sweep information by name."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_sweep(sweep_name)


@mcp.tool()
async def get_next_run_id(sweep_name: str, dev_mode: bool = False) -> RunIdResponse:
    """Get the next run ID for a sweep (atomic operation)."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_next_run_id(sweep_name)


# Token Management Tools
@mcp.tool()
async def list_tokens(dev_mode: bool = False) -> BackendTokenListResponse:
    """List all machine tokens for the authenticated user."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.list_tokens()


@mcp.tool()
async def create_token(name: str, dev_mode: bool = False) -> TokenResponse:
    """Create a new machine token for the authenticated user."""
    from metta.app_backend.routes.token_routes import TokenCreate

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        token_data = TokenCreate(name=name)
        return await client.create_token(token_data)


@mcp.tool()
async def delete_token(token_id: str, dev_mode: bool = False) -> dict[str, str]:
    """Delete a machine token for the authenticated user."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.delete_token(token_id)
        return result


@mcp.tool()
async def create_cli_token(callback: str, dev_mode: bool = False) -> dict[str, str | int | float | bool | None]:
    """Create a machine token and redirect to callback URL with token parameter.

    Args:
        callback (str): Callback URL to redirect to with token parameter.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        dict[str, str | int | float | bool | None]: Token creation response with redirect information.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.create_cli_token(callback)
        return result


# Score Routes
@mcp.tool()
async def get_policy_scores(
    policy_ids: list[str] | str,
    eval_names: list[str] | str,
    metrics: list[str] | str,
    dev_mode: bool = False,
) -> PolicyScoresData:
    """Get Policy Scores"""
    import uuid

    policy_ids_list = [uuid.UUID(pid) for pid in _parse_str_or_list(policy_ids)]
    eval_names_list = _parse_str_or_list(eval_names)
    metrics_list = _parse_str_or_list(metrics)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_policy_scores(policy_ids_list, eval_names_list, metrics_list)


# Enhanced Scorecard Generation
@mcp.tool()
async def generate_policy_scorecard(
    training_run_ids: list[str] | str,
    run_free_policy_ids: list[str] | str,
    eval_names: list[str] | str,
    metric: str,
    training_run_policy_selector: Literal["best", "latest"] = "latest",
    dev_mode: bool = False,
) -> BackendScorecardData:
    """Generate scorecard data based on training run and policy selection."""
    training_run_ids_list = _parse_str_or_list(training_run_ids)
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids)
    eval_names_list = _parse_str_or_list(eval_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.generate_scorecard(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
            eval_names=eval_names_list,
            metric=metric,
            policy_selector=training_run_policy_selector,
        )
        return result


@mcp.tool()
async def generate_heatmap_scorecard(
    training_run_ids: list[str] | str,
    run_free_policy_ids: list[str] | str,
    eval_names: list[str] | str,
    metric: str,
    training_run_policy_selector: Literal["best", "latest"] = "latest",
    dev_mode: bool = False,
) -> BackendScorecardData:
    """Generate heatmap scorecard data based on training run and policy selection."""
    training_run_ids_list = _parse_str_or_list(training_run_ids)
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids)
    eval_names_list = _parse_str_or_list(eval_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.generate_heatmap_scorecard(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
            eval_names=eval_names_list,
            metric=metric,
            training_run_policy_selector=training_run_policy_selector,
        )
        return result


@mcp.tool()
async def generate_training_run_scorecard(
    run_id: str,
    eval_names: list[str] | str,
    metric: str,
    dev_mode: bool = False,
) -> BackendScorecardData:
    """Generate scorecard data for a specific training run showing ALL policies."""
    eval_names_list = _parse_str_or_list(eval_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.generate_training_run_scorecard(
            run_id=run_id,
            eval_names=eval_names_list,
            metric=metric,
        )
        return result


@mcp.tool()
async def generate_leaderboard_scorecard(
    leaderboard_id: str,
    selector: Literal["latest", "best"] = "latest",
    num_policies: int = 10,
    dev_mode: bool = False,
) -> BackendScorecardData:
    """Generate scorecard data for a leaderboard."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.generate_leaderboard_scorecard(
            leaderboard_id=leaderboard_id,
            selector=selector,
            num_policies=num_policies,
        )
        return result


# Configuration Management Tools
@mcp.tool()
async def list_hydra_configs(config_type: str | None = None) -> HydraConfigList:
    """Browse available Hydra configurations.

    Args:
        config_type (str | None): Configuration type to list (agent, sim, trainer, user, wandb, sweep).
                                 If None, returns all available config types.

    Returns:
        HydraConfigList: Available configurations or config types.
    """
    try:
        if config_type is None:
            # Return all available config types as configs
            config_types = get_available_config_types()
            return HydraConfigList(
                config_type="all_types",
                configs=config_types,
                total_configs=len(config_types),
            )
        else:
            # Return configs for specific type
            configs_data = list_configs_for_type(config_type)
            config_names = [config["name"] for config in configs_data]
            return HydraConfigList(
                config_type=config_type,
                configs=config_names,
                total_configs=len(config_names),
            )
    except Exception as e:
        return HydraConfigList(
            config_type=config_type or "error",
            configs=[f"Error: {str(e)}"],
            total_configs=0,
        )


@mcp.tool()
async def validate_config(
    config_path: str,
    overrides: list[str] | str | None = None,
) -> ValidationResult:
    """Validate a Hydra configuration with optional overrides.

    Args:
        config_path (str): Path to configuration file (relative to configs/ or absolute).
        overrides (list[str] | str | None): List of Hydra override strings (key=value, +key=value, ++key=value).

    Returns:
        ValidationResult: Validation result with config data or error information.
    """
    try:
        # Parse overrides if provided as string
        override_list = []
        if overrides:
            if isinstance(overrides, str):
                override_list = [overrides]
            else:
                override_list = overrides

        result = validate_config_func(config_path, override_list)

        # Convert OmegaConf objects to regular Python objects for Pydantic serialization
        config_data = None
        if result.config_data is not None:
            if isinstance(result.config_data, (DictConfig, ListConfig)):
                config_data = OmegaConf.to_container(result.config_data, resolve=True)
            else:
                config_data = result.config_data

        return ValidationResult(
            valid=result.valid,
            config_data=config_data,
            errors=result.errors,
            config_path=result.config_path,
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            config_data=None,
            errors=[str(e)],
            config_path=config_path,
        )


@mcp.tool()
async def get_config_schema(config_type: str) -> HydraConfigSchema:
    """Get schema information for a configuration type.

    Args:
        config_type (str): Configuration type to analyze (agent, sim, trainer, etc.).

    Returns:
        HydraConfigSchema: Schema information including field types and structure.
    """
    try:
        result = get_config_schema_func(config_type)
        return HydraConfigSchema(
            config_type=result.get("config_type", config_type),
            schema_info=result.get("schema", {}),
        )
    except Exception:
        return HydraConfigSchema(
            config_type=config_type,
            schema_info={},
        )


# Training and Evaluation Management Tools
@mcp.tool()
async def list_training_runs_local() -> LocalTrainingRunList:
    """List local training runs in train_dir with metadata.

    Returns:
        LocalTrainingRunList: List of training runs with their status, checkpoints, and metadata.
    """
    try:
        runs = list_training_runs()
        return LocalTrainingRunList(
            training_runs=runs,
            total_runs=len(runs),
        )
    except Exception:
        return LocalTrainingRunList(
            training_runs=[],
            total_runs=0,
        )


@mcp.tool()
async def get_checkpoint_info(checkpoint_path: str) -> CheckpointInfo:
    """Inspect checkpoint metadata without loading the full model.

    Args:
        checkpoint_path (str): Path to checkpoint file or directory. Supports policy URIs (file://).

    Returns:
        CheckpointInfo: Checkpoint information including file size, modification time, and model metadata.
    """
    try:
        return get_checkpoint_info_func(checkpoint_path)
    except Exception:
        return CheckpointInfo(
            checkpoint_path=checkpoint_path,
            file_size=0,
            modified_time="",
            model_metadata={},
        )


@mcp.tool()
async def get_training_status(run_name: str) -> TrainingStatus:
    """Check if training is running/completed with detailed status information.

    Args:
        run_name (str): Name of the training run to check.

    Returns:
        TrainingStatus: Detailed training status including process info, logs, and checkpoints.
    """
    try:
        return get_training_status_func(run_name)
    except Exception as e:
        return TrainingStatus(
            run_name=run_name,
            status="error",
            progress={},
            logs=[str(e)],
            checkpoints=[],
        )


@mcp.tool()
async def generate_replay_summary(
    replay_path: str, policy_uri: str | None = None, ctx: Context = None
) -> ReplaySummaryResponse:
    """Generate AI-powered summary of replay contents using statistical analysis.

    This function performs comprehensive analysis of a replay file including:
    - Agent behavioral patterns and efficiency rankings
    - Resource flow analysis and scarcity assessment
    - Combat interaction analysis and cooperation metrics
    - Strategic phase detection and timeline analysis
    - Building efficiency scoring and optimization recommendations
    - AI-powered narrative summary using Claude

    Args:
        replay_path (str): Path to replay file (supports .json, .json.z compressed format).
                          Can be local file path or S3 URL.
        policy_uri (str, optional): URI of the policy used to generate this replay
                    (e.g., "file://./checkpoints", "wandb://run/my-run:v42").
                    Used to determine if agents are trained or untrained for context.

    Returns:
        ReplaySummaryResponse: Either ReplaySummarySuccess with analysis data and AI summary,
                              or ReplaySummaryError with error details.

        Success Response:
            - replay_path: Path to analyzed file
            - file_size: File size in bytes
            - summary: Comprehensive AI-generated analysis including statistical insights
            - llm_used: Whether Claude LLM was used for summary generation

        Error Response:
            - error: Detailed error message
            - path: File path that failed to analyze

    Raises:
        No exceptions - all errors are captured and returned in ReplaySummaryError format.
    """
    try:
        result = await generate_replay_summary_with_llm(replay_path, policy_uri, ctx)

        # Check if result contains error
        if "error" in result:
            return ReplaySummaryError(error=result["error"], path=result.get("path", replay_path))

        # Return successful result
        return ReplaySummarySuccess(
            replay_path=result["replay_path"],
            file_size=result["file_size"],
            summary=result["summary"],
            llm_used=result["llm_used"],
        )
    except Exception as e:
        if ctx:
            await ctx.error(f"Replay summary generation failed: {str(e)}")
        return ReplaySummaryError(error=str(e), path=replay_path)


if __name__ == "__main__":
    override = None
    if "--config" in sys.argv:
        override = sys.argv[sys.argv.index("--config") + 1]
    override = override or os.getenv("METTA_MCP_CONFIG")
    if override:
        try:
            CONFIG = json.loads(Path(override).read_text())
        except Exception as e:
            print(f"Failed to load config: {e}", file=sys.stderr)
    mcp.run()

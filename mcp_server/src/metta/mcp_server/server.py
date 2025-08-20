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
from pydantic import Field
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
from metta.app_backend.routes.scorecard_routes import PoliciesResponse, ScorecardData
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
from metta.mcp_server.training_utils import (
    ReplaySummaryError,
    ReplaySummarySuccess,
    TrainingStatus,
    TrainingUtilsError,
    generate_replay_summary_with_llm,
    list_training_runs,
)
from metta.mcp_server.training_utils import (
    get_checkpoint_info as get_checkpoint_info_func,
)
from metta.mcp_server.training_utils import (
    get_training_status as get_training_status_func,
)

# Import models from organized modules
from .base_models import (
    UserInfo,
)
from .cloud_models import (
    S3ObjectList,
    S3ObjectMetadata,
    S3PrefixList,
    SkypilotStatus,
    WandbError,
    WandbRun,
    WandbRunList,
)
from .ml_models import (
    CheckpointInfo,
    CheckpointInfoError,
    LocalTrainingRunList,
    PolicyIdMapping,
    ScorecardDataWithMetadata,
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


def _parse_dev_mode(dev_mode_str: str) -> bool:
    """Convert string dev_mode parameter to boolean.

    Args:
        dev_mode_str: String representation of dev_mode ("true", "1", "yes" -> True)

    Returns:
        Boolean value for dev_mode
    """
    return dev_mode_str.lower() in ("true", "1", "yes")


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


@mcp.resource(
    "metta://training-runs/{dev_mode}",
    name="Training Runs",
    description="Search and list training runs and policies with filtering and pagination",
    tags={"training", "search", "machine-learning"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "ml-platform"},
)
async def search_training_runs(
    dev_mode: str = "false",
    search: str | None = None,
    policy_type: str | None = None,
    tags: list[str] | str | None = None,
    user_id: str | None = None,
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results to return"),
    offset: int = Field(default=0, ge=0, description="Offset into the result set for pagination"),
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
    async with ScorecardClient(backend_url=_get_backend_url(_parse_dev_mode(dev_mode))) as client:
        return await client.search_policies(
            search=search,
            policy_type=policy_type,
            tags=_parse_str_or_list(tags) if isinstance(tags, (list, str)) else None,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )


@mcp.resource(
    "metta://evaluations/names/{dev_mode}",
    name="Evaluation Names",
    description="Available evaluation names for training runs and policies",
    tags={"training", "evaluation", "metadata"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_eval_names_for_training_runs(
    dev_mode: str = "false",
    training_run_ids: list[str] | str | None = None,
    run_free_policy_ids: list[str] | str | None = None,
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

    async with ScorecardClient(backend_url=_get_backend_url(_parse_dev_mode(dev_mode))) as client:
        return await client.get_eval_names(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
        )


@mcp.tool(
    name="Execute SQL Query",
    description="Execute a SQL query against the scorecard database API",
    tags={"database", "sql", "query"},
    annotations={
        "readOnlyHint": False,  # SQL can modify data
        "destructiveHint": True,  # Could be destructive
        "idempotentHint": True,
    },
)
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


@mcp.resource(
    "metta://database/tables/{dev_mode}",
    name="Database Tables",
    description="List all available database tables with metadata",
    tags={"database", "schema", "metadata"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def list_sql_tables(dev_mode: str = "false") -> list[TableInfo]:
    """List all available tables in the database (excluding migrations).

    Args:
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        list[TableInfo]: Backend response containing list of table metadata objects
            with names, column counts, and row counts.
    """
    async with ScorecardClient(backend_url=_get_backend_url(_parse_dev_mode(dev_mode))) as client:
        return await client.list_tables()


@mcp.resource(
    "metta://database/schema/{table_name}/{dev_mode}",
    name="Table Schema",
    description="Get detailed schema information for a specific database table",
    tags={"database", "schema", "metadata"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,  # Table schema is stable
    },
)
async def get_sql_table_schema(table_name: str, dev_mode: str = "false") -> BackendTableSchema:
    """Get the schema for a specific table.

    Args:
        table_name (str): Name of the table to inspect.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        BackendTableSchema: Backend response containing table schema information
            with table name, column definitions with types and constraints, and table-level constraints.
    """
    async with ScorecardClient(backend_url=_get_backend_url(_parse_dev_mode(dev_mode))) as client:
        return await client.get_table_schema(table_name)


@mcp.tool(
    name="AI SQL Query Generator",
    description="Generate SQL queries using artificial intelligence based on natural language descriptions",
    tags={"database", "sql", "ai", "generation"},
    annotations={
        "readOnlyHint": True,  # Just generates
        "idempotentHint": True,  # Generate as many as you like
    },
)
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


@mcp.resource(
    "metta://metrics/available/{dev_mode}",
    name="Available Training Metrics",
    description="Enumerate metrics available for specified training runs, policies, and evaluations",
    tags={"training", "metrics", "evaluation"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_available_metrics_for_training_runs(
    dev_mode: str = "false",
    training_run_ids: list[str] | str | None = None,
    run_free_policy_ids: list[str] | str | None = None,
    eval_names: list[str] | str | None = None,
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
    training_run_ids_list = _parse_str_or_list(training_run_ids) if training_run_ids else []
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids) if run_free_policy_ids else []
    eval_names_list = _parse_str_or_list(eval_names) if eval_names else []

    async with ScorecardClient(backend_url=_get_backend_url(_parse_dev_mode(dev_mode))) as client:
        return await client.get_available_metrics(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
            eval_names=eval_names_list,
        )


@mcp.resource(
    "metta://scorecard/data/{dev_mode}",
    name="Training Scorecard Data",
    description="Retrieve comprehensive scorecard data for training runs and policies with filtering",
    tags={"training", "scorecard", "evaluation", "metrics"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_scorecard_data(
    dev_mode: str = "false",
    search_term: str | None = None,
    restrict_to_policy_ids: list[str] | str | None = None,
    restrict_to_metrics: list[str] | str | None = None,
    restrict_to_policy_names: list[str] | str | None = None,
    restrict_to_eval_names: list[str] | str | None = None,
    policy_selector: Literal["best", "latest"] = "best",
    max_policies: int = 20,
    include_run_free_policies: bool = False,
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

    async with ScorecardClient(backend_url=_get_backend_url(_parse_dev_mode(dev_mode))) as client:
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


@mcp.tool(
    name="Weights & Biases Runs",
    description="List recent training runs from Weights & Biases for a project",
    tags={"wandb", "training", "monitoring"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,  # Connects to external W&B service
    },
)
async def list_wandb_runs(
    project: str | None = None,
    entity: str | None = None,
    limit: int = Field(default=50, ge=1, le=100, description="Maximum number of runs to return"),
) -> WandbRunList | WandbError:
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
    if not entity or not project:
        return WandbError(error="Entity and project must be provided", status="error")
    wandb_api = WandbApi()
    runs = wandb_api.runs(f"{entity}/{project}")[:limit]
    run_data = [{"id": run.id, "name": run.name, "state": run.state} for run in runs]
    return WandbRunList(runs=run_data, project=project, entity=entity)


@mcp.resource(
    "metta://wandb/runs/{run_name}",
    name="Weights & Biases Run Details",
    description="Get detailed information about a specific W&B training run",
    tags={"wandb", "training", "monitoring"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,  # Connects to external W&B service
    },
)
async def get_wandb_run(
    run_name: str = Field(min_length=1, max_length=255, description="The run name or id to fetch"),
    project: str | None = None,
    entity: str | None = None,
) -> WandbRun:
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
    return WandbRun(id=run.id, name=run.name, state=run.state, url=run.url)


@mcp.resource(
    "metta://s3/objects/{bucket_name}",
    name="S3 Bucket Objects",
    description="List objects in an AWS S3 bucket with optional prefix filtering",
    tags={"aws", "s3", "storage", "files"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,  # Connects to external AWS service
    },
)
async def list_s3_objects(
    bucket_name: str = "default",
    prefix: str | None = None,
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of objects to return"),
) -> S3ObjectList:
    """List objects from a configured AWS S3 bucket.

    Args:
        bucket_name (str): Bucket name. Uses the first configured bucket when "default".
        prefix (str | None): Key prefix to filter objects.
        limit (int): Maximum number of objects to return. Defaults to 100.

    Returns:
        S3ObjectList: List of S3 object keys with bucket and prefix information.
            Contains object keys, bucket name, and the prefix used for filtering.
    """
    cfg = CONFIG["resources"]["aws_s3"]
    bucket_name = bucket_name if bucket_name != "default" else cfg["buckets"][0]
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


@mcp.resource(
    "metta://s3/prefixes/{bucket}",
    name="S3 Bucket Prefixes",
    description="List directory-like prefixes in an AWS S3 bucket",
    tags={"aws", "s3", "storage", "directories"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,  # Connects to external AWS service
    },
)
async def list_s3_prefixes(
    bucket: str | None = None,
    prefix: str | None = None,
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results to return"),
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


@mcp.resource(
    "metta://s3/objects/{key}/metadata",
    name="S3 Object Metadata",
    description="Get metadata for a specific S3 object without downloading the content",
    tags={"aws", "s3", "storage", "metadata"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,  # Object metadata is static
        "openWorldHint": True,  # Connects to external AWS service
    },
)
async def get_s3_object_head(
    key: str,
    bucket: str | None = None,
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


@mcp.resource(
    "metta://skypilot/jobs/status",
    name="Skypilot Job Status",
    description="List active Skypilot jobs and cluster information for the logged in user",
    tags={"skypilot", "cloud", "jobs", "clusters"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,  # Connects to external cloud services
    },
)
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


@mcp.resource(
    "metta://wandb/runs/{run_name}/url",
    name="W&B Run URL",
    description="Get the direct dashboard URL for a specific Weights & Biases run",
    tags={"wandb", "training", "url"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,  # Connects to external W&B service
    },
)
async def get_wandb_run_url(
    run_name: str = Field(min_length=1, max_length=255, description="The run name or id to fetch"),
    project: str | None = None,
    entity: str | None = None,
) -> str:
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


@mcp.resource(
    "metta://user/profile/{dev_mode}",
    name="Current User Info",
    description="Get the current user's authentication status and profile information",
    tags={"authentication", "user", "profile"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
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
@mcp.resource(
    "metta://dashboards/{dev_mode}",
    name="Saved Dashboards",
    description="List all saved dashboards with metadata and configurations",
    tags={"dashboard", "visualization", "saved-state"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def list_saved_dashboards(dev_mode: bool = False) -> SavedDashboardListResponse:
    """List all saved dashboards.

    Returns:
        SavedDashboardListResponse: Backend response containing list of dashboard objects
            with dashboard metadata, names, types, and configurations.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.list_saved_dashboards()


@mcp.tool(
    name="Create Dashboard",
    description="Create a new saved dashboard with specified configuration",
    tags={"dashboard", "create", "visualization"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
    },
)
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


@mcp.resource(
    "metta://dashboards/{dashboard_id}",
    name="Saved Dashboard Details",
    description="Get complete configuration and metadata for a specific saved dashboard",
    tags={"dashboard", "visualization", "configuration"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_saved_dashboard(
    dashboard_id: str = Field(min_length=1, max_length=100, description="Dashboard ID to retrieve"),
    dev_mode: bool = False,
) -> SavedDashboardResponse:
    """Get a specific saved dashboard by ID.

    Returns:
        SavedDashboardResponse: Backend response containing complete dashboard object
            with ID, name, type, description, dashboard state configuration, and creation timestamp.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_saved_dashboard(dashboard_id)


@mcp.tool(
    name="Update Dashboard",
    description="Update the configuration and state of an existing saved dashboard",
    tags={"dashboard", "update", "configuration"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,  # Same update produces same result
    },
)
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


@mcp.tool(
    name="Delete Dashboard",
    description="Permanently delete a saved dashboard",
    tags={"dashboard", "delete", "remove"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,  # Permanently deletes data
        "idempotentHint": True,  # Multiple deletes of same ID have same effect
    },
)
async def delete_saved_dashboard(dashboard_id: str, dev_mode: bool = False) -> SavedDashboardDeleteResponse:
    """Delete a saved dashboard.

    Returns:
        SavedDashboardDeleteResponse: Backend response confirming dashboard deletion
            with operation status and any relevant metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.delete_saved_dashboard(dashboard_id)


# Entity Routes (Training Runs)
@mcp.tool(
    name="Training Runs List",
    description="Complete list of all training runs with metadata, descriptions, and tags",
    tags={"training", "machine-learning", "runs"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "ml-platform"},
)
async def get_training_runs(dev_mode: bool = False) -> TrainingRunListResponse:
    """Get all training runs.

    Returns:
        TrainingRunListResponse: Backend response containing list of training run objects
            with metadata, names, descriptions, tags, and total count.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_training_runs()


@mcp.resource(
    "metta://training-runs/{run_id}",
    name="Training Run Details",
    description="Detailed information about a specific training run including metadata and attributes",
    tags={"training", "machine-learning", "details"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "ml-platform"},
)
async def get_training_run(
    run_id: str = Field(min_length=1, max_length=100, description="Training run ID to retrieve"), dev_mode: bool = False
) -> TrainingRunResponse:
    """Get a specific training run by ID.

    Returns:
        TrainingRunResponse: Backend response containing detailed training run information
            including ID, name, description, tags, URL, creation timestamp, and attributes.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_training_run(run_id)


@mcp.tool(
    name="Update Training Run Description",
    description="Update the description text for a training run",
    tags={"training", "update", "description", "metadata"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,  # Only updates metadata
        "idempotentHint": True,
    },
)
async def update_training_run_description(run_id: str, description: str, dev_mode: bool = False) -> TrainingRunResponse:
    """Update the description of a training run.

    Returns:
        TrainingRunResponse: Backend response containing the updated training run
            with new description and all other run metadata preserved.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.update_training_run_description(run_id, description)


@mcp.tool(
    name="Update Training Run Tags",
    description="Update the tags associated with a training run for organization and filtering",
    tags={"training", "update", "tags", "organization"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,  # Only updates metadata
        "idempotentHint": True,
    },
)
async def update_training_run_tags(run_id: str, tags: list[str] | str, dev_mode: bool = False) -> TrainingRunResponse:
    """Update the tags of a training run.

    Returns:
        TrainingRunResponse: Backend response containing the updated training run
            with new tags and all other run metadata preserved.
    """
    tags_list = _parse_str_or_list(tags)
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.update_training_run_tags(run_id, tags_list)


@mcp.resource(
    "metta://training-runs/{run_id}/policies",
    name="Training Run Policies",
    description="List all policies associated with a training run including epoch and performance data",
    tags={"training", "policies", "epochs", "performance"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "ml-platform"},
)
async def get_training_run_policies(run_id: str, dev_mode: bool = False) -> TrainingRunPolicyListResponse:
    """Get policies for a training run with epoch information.

    Returns:
        TrainingRunPolicyListResponse: Backend response containing list of policy objects
            with epoch information, policy IDs, names, epoch numbers, training metrics, and metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_training_run_policies(run_id)


# Task Management Tools
@mcp.tool(
    name="Create Evaluation Task",
    description="Create a new evaluation task for a policy with specified simulation suite and parameters",
    tags={"tasks", "evaluation", "create", "scheduling"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,  # Creates new resources
    },
)
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


@mcp.tool(
    name="Get Latest Assigned Task",
    description="Get the most recent task assigned to a specific worker",
    tags={"tasks", "workers", "assignment", "latest"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
async def get_latest_assigned_task_for_worker(assignee: str, dev_mode: bool = False) -> TaskResponse | None:
    """Get Latest Assigned Task For Worker

    Returns:
        TaskResponse | None: Backend response containing the latest task assigned to the worker,
            or None if no tasks are assigned. Includes task ID, assignee, assignment time,
            policy ID, and current status.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_latest_assigned_task_for_worker(assignee)


@mcp.tool(
    name="Available Tasks",
    description="List of evaluation tasks available for worker assignment",
    tags={"tasks", "evaluation", "workers", "queue"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "evaluation"},
)
async def get_available_tasks(
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum number of tasks to return"),
    dev_mode: bool = False,
) -> TasksResponse:
    """Get Available Tasks

    Returns:
        TasksResponse: Backend response containing list of available task objects
            with metadata and total count. Tasks represent evaluation jobs that can be claimed by workers.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_available_tasks(limit=limit)


@mcp.tool(
    name="Claim Evaluation Tasks",
    description="Assign evaluation tasks to a worker for execution",
    tags={"tasks", "workers", "claim", "assignment"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,  # Changes task ownership
    },
)
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


@mcp.tool(
    name="Claimed Tasks",
    description="List of evaluation tasks currently claimed by workers",
    tags={"tasks", "evaluation", "workers", "assigned"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "evaluation"},
)
async def get_claimed_tasks(assignee: str | None = None, dev_mode: bool = False) -> TasksResponse:
    """Get Claimed Tasks

    Returns:
        TasksResponse: Backend response containing list of claimed task objects
            with task metadata, assignee information, and status details.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_claimed_tasks(assignee)


@mcp.tool(
    name="Get Worker Git Hashes",
    description="Get git commit hashes for specified workers to track code version alignment",
    tags={"tasks", "workers", "git", "versioning"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
async def get_git_hashes_for_workers(assignees: list[str] | str, dev_mode: bool = False) -> GitHashesResponse:
    """Get Git Hashes For Workers

    Returns:
        GitHashesResponse: Backend response containing git hash information
            for the specified workers, mapping assignee names to git commit hashes.
    """
    assignee_list = _parse_str_or_list(assignees)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_git_hashes_for_workers(assignee_list)


@mcp.tool(
    name="All Tasks",
    description="Complete list of evaluation tasks with optional filtering by git hash, policies, and status",
    tags={"tasks", "evaluation", "filtering", "status"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "evaluation"},
)
async def get_all_tasks(
    limit: int = Field(default=500, ge=1, le=2000, description="Maximum number of tasks to return"),
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


@mcp.tool(
    name="Update Task Statuses",
    description="Update the execution status and results for multiple evaluation tasks",
    tags={"tasks", "status", "update", "results"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    },
)
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
@mcp.tool(
    name="Leaderboards List",
    description="List all evaluation leaderboards with their configurations and metrics",
    tags={"leaderboards", "evaluation", "competition", "metrics"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,  # Leaderboards can be added or modified
    },
    meta={"version": "1.0", "team": "evaluation"},
)
async def list_leaderboards(dev_mode: bool = False) -> LeaderboardListResponse:
    """List all leaderboards for the current user.

    Returns:
        LeaderboardListResponse: Backend response containing list of leaderboard objects
            with metadata, names, evaluation lists, metrics, and configuration details.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.list_leaderboards()


@mcp.tool(
    name="Create Leaderboard",
    description="Create a new evaluation leaderboard with specified metrics and evaluations",
    tags={"leaderboards", "evaluation", "create", "competition"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,  # Creates new resources
    },
)
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


@mcp.resource(
    "metta://leaderboards/{leaderboard_id}",
    name="Leaderboard Details",
    description="Complete configuration and metadata for a specific evaluation leaderboard",
    tags={"leaderboards", "evaluation", "configuration"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "evaluation"},
)
async def get_leaderboard(leaderboard_id: str, dev_mode: bool = False) -> LeaderboardResponse:
    """Get a specific leaderboard by ID.

    Returns:
        LeaderboardResponse: Backend response containing complete leaderboard object
            with ID, name, evaluation list, primary metric, start date, and creation timestamp.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.get_leaderboard(leaderboard_id)


@mcp.tool(
    name="Update Leaderboard",
    description="Update the configuration of an existing evaluation leaderboard",
    tags={"leaderboards", "evaluation", "update", "configuration"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,  # Same update produces same result
    },
)
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


@mcp.tool(
    name="Delete Leaderboard",
    description="Permanently delete an evaluation leaderboard",
    tags={"leaderboards", "evaluation", "delete", "remove"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,  # Permanently deletes data
        "idempotentHint": True,  # Multiple deletes have same effect
    },
)
async def delete_leaderboard(leaderboard_id: str, dev_mode: bool = False) -> LeaderboardDeleteResponse:
    """Delete a leaderboard.

    Returns:
        LeaderboardDeleteResponse: Backend response confirming leaderboard deletion
            with operation status and any relevant metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        return await client.delete_leaderboard(leaderboard_id)


# Stats and Metrics Tools
@mcp.resource(
    "metta://policies/mapping/{dev_mode}/{policy_names}",
    name="Policy Name to ID Mapping",
    description="Look up policy UUIDs by their human-readable names",
    tags={"policies", "mapping", "lookup", "uuid"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "ml-platform"},
)
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


@mcp.tool(
    name="Create Training Run",
    description="Create a new training run record with metadata for tracking ML experiments",
    tags={"training", "create", "experiments", "tracking"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,  # Creates new resources
    },
)
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


@mcp.tool(
    name="Create Training Epoch",
    description="Create a new epoch record to track training progress within a run",
    tags={"training", "epochs", "create", "progress"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,  # Creates new resources
    },
)
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


@mcp.tool(
    name="Create Policy",
    description="Create a new policy record for tracking trained models and their metadata",
    tags={"policies", "models", "create", "tracking"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,  # Creates new resources
    },
)
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


@mcp.resource(
    "resource://sweeps/{sweep_name}",
    name="Sweep Information",
    description="Get configuration and status information for a specific hyperparameter sweep",
    tags={"sweeps", "hyperparameter", "optimization", "wandb"},
    annotations={"readOnlyHint": True, "idempotentHint": True},
    meta={"version": "1.0", "team": "ml-platform"},
)
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
@mcp.tool(
    name="User API Tokens",
    description="List all machine tokens associated with the authenticated user",
    tags={"authentication", "tokens", "api", "security"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "auth"},
)
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
) -> ScorecardData:
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
) -> ScorecardData:
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
) -> ScorecardData:
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
    num_policies: int = Field(default=10, ge=1, le=100, description="Number of policies to include in leaderboard"),
    dev_mode: bool = False,
) -> ScorecardData:
    """Generate scorecard data for a leaderboard."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.generate_leaderboard_scorecard(
            leaderboard_id=leaderboard_id,
            selector=selector,
            num_policies=num_policies,
        )
        return result


# Training and Evaluation Management Tools
@mcp.resource(
    "metta://training-runs/local/list",
    name="Local Training Runs",
    description="List training runs stored locally in train_dir with status and checkpoint information",
    tags={"training", "local", "filesystem", "checkpoints"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "training"},
)
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


@mcp.resource(
    "metta://checkpoints/{checkpoint_path*}/info",
    name="Checkpoint Information",
    description="Get metadata about model checkpoints without loading the full model",
    tags={"checkpoints", "models", "metadata", "inspection"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
    meta={"version": "1.0", "team": "training"},
)
async def get_checkpoint_info(
    checkpoint_path: str = Field(
        min_length=1, description="Path to checkpoint file or directory. Supports policy URIs (file://)"
    ),
) -> CheckpointInfo | CheckpointInfoError:
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


@mcp.resource(
    "metta://training/{run_name}/status",
    name="Training Status",
    description="Get detailed status information for a training run including process state and logs",
    tags={"training", "status", "monitoring", "processes"},
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,  # Status changes over time
    },
    meta={"version": "1.0", "team": "training"},
)
async def get_training_status(run_name: str) -> TrainingStatus | TrainingUtilsError:
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
    replay_path: str = Field(
        min_length=1, description="Path to replay file (supports .json, .json.z compressed format)"
    ),
    policy_uri: str | None = None,
    ctx: Context | None = None,
) -> ReplaySummarySuccess | ReplaySummaryError:
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

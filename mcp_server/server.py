from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import boto3

# from mcp.server import FastMCP
from fastmcp import FastMCP
from pydantic.types import Json
from wandb.apis.public.api import Api as WandbApi

from metta.app_backend.clients.scorecard_client import ScorecardClient

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
) -> dict[str, Any]:
    """Search training runs for policies.

    Args:
        search (str | None): Free-text search over names, descriptions, and tags.
        policy_type (str | None): Filter by policy type when provided.
        tags (list[str] | str | None): Filter to items that contain any of these tags.
        user_id (str | None): Filter to items owned by a specific user id.
        limit (int): Maximum number of results to return. Defaults to 100.
        offset (int): Offset into the result set for pagination. Defaults to 0.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
            dict[str, Any]: A dictionary containing matched policies and their paging metadata.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.search_policies(
            search=search,
            policy_type=policy_type,
            tags=_parse_str_or_list(tags) if isinstance(tags, (list, str)) else None,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
    return result.model_dump()


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
            dict[str, Any]: Mapping of ids to lists of eval names.
    """
    training_run_ids_list = _parse_str_or_list(training_run_ids)
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_eval_names(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
        )
    return result


@mcp.tool()
async def run_sql_query(sql: str, dev_mode: bool = False) -> dict[str, Any]:
    """Execute a SQL query against the scorecard database API.

    Args:
        sql (str): SQL statement to execute.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        dict[str, Any]: Result payload returned by the backend for the query.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.sql_query(sql)
        return result.model_dump()


@mcp.tool()
async def artificial_intelligence_sql_query_generation(sql: str, dev_mode: bool = False) -> dict[str, Any]:
    """Generate an SQL query using artificial intelligence according to the
    schema of the database. Very useful for when you get errors or don't know

    Args:
        sql (str): SQL statement to execute.
        dev_mode (bool): When True, use the default/local backend URL; otherwise use production.

    Returns:
        dict[str, Any]: Result payload returned by the backend for the query.
    """
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.generate_ai_query(sql)
        return result.model_dump()


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
        list[str]: Available metric names for the training runs and policies you specified.
    """
    training_run_ids_list = _parse_str_or_list(training_run_ids)
    run_free_policy_ids_list = _parse_str_or_list(run_free_policy_ids)
    eval_names_list = _parse_str_or_list(eval_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_available_metrics(
            training_run_ids=training_run_ids_list,
            run_free_policy_ids=run_free_policy_ids_list,
            eval_names=eval_names_list,
        )
        return result


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
) -> dict[str, Any]:
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

    Returns data that looks like:
        primary_metric:
            string
        valid_metrics:
            string[]
        scorecard_data:
            evalNames:
                string[]
            policyNames:
                string[]
            policyAverageScores:
                [string]: number
            evalAverageScores:
                [string]: number
            evalMaxScores:
                [string]: number
            cells:
                "example_cell":
                    "arena/tag":
                        "evalName": "sample_eval",
                        "replayUrl": "https://softmax-public.s3.amazonaws.com/replays/sample_policy/sample_eval.json.z",
                        "value": 1
                // ... more cells
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
            return {}
        _, scorecard_data, valid_metrics, primary_metric = result
        return {
            "primaryMetric": primary_metric,
            "availableMetrics": valid_metrics,
            "scorecard_data": scorecard_data,
        }


@mcp.tool()
async def list_wandb_runs(
    project: str | None = None,
    entity: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List recent Weights & Biases runs for the configured project.

    Args:
        project (str | None): W&B project name. Uses configured default when None.
        entity (str | None): W&B entity (organization or user). Uses configured default when None.
        limit (int): Maximum number of runs to return. Defaults to 50.

    Returns:
        list[dict[str, Any]]: A list of runs with minimal identifying metadata.
    """
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    wandb_api = WandbApi()
    runs = wandb_api.runs(f"{entity}/{project}")[:limit]
    return [{"id": run.id, "name": run.name, "state": run.state} for run in runs]


@mcp.tool()
async def get_wandb_run(
    run_name: str,
    project: str | None = None,
    entity: str | None = None,
) -> list[dict[str, Any]]:
    """Get a specific Weights & Biases run by name.

    Args:
        run_name (str): The run name or id to fetch.
        project (str | None): W&B project name. Uses configured default when None.
        entity (str | None): W&B entity (organization or user). Uses configured default when None.

    Returns:
        list[dict[str, Any]]: A list result from the W&B API matching the provided run name.
    """
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    wandb_api = WandbApi()
    return wandb_api.runs(f"{entity}/{project}/runs/${run_name}")


@mcp.tool()
async def list_s3_objects(
    bucket: str | None = None,
    prefix: str | None = None,
    limit: int = 100,
) -> list[str]:
    """List objects from a configured AWS S3 bucket.

    Args:
        bucket (str | None): Bucket name. Uses the first configured bucket when None.
        prefix (str | None): Key prefix to filter objects.
        limit (int): Maximum number of objects to return. Defaults to 100.

    Returns:
        list[str]: Object keys returned from S3.
    """
    cfg = CONFIG["resources"]["aws_s3"]
    bucket = bucket or cfg["buckets"][0]
    s3 = boto3.client("s3", region_name=cfg.get("region"))
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix or "",
        PaginationConfig={"MaxItems": limit},
    )
    keys: list[str] = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


@mcp.tool()
async def list_s3_prefixes(
    bucket: str | None = None,
    prefix: str | None = None,
    limit: int = 100,
) -> list[str]:
    """List common prefixes ("directories") in an S3 bucket under a prefix.

    Args:
        bucket (str | None): Bucket name. Uses first configured bucket when None.
        prefix (str | None): Key prefix to look under.
        limit (int): Maximum number of results to return. Defaults to 100.

    Returns:
        list[str]: Discovered common prefixes ending with '/'.
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
    return prefixes


@mcp.tool()
async def get_s3_object_head(
    bucket: str | None,
    key: str,
) -> dict[str, Any]:
    """Fetch S3 object metadata (HEAD) without downloading the body.

    Args:
        bucket (str | None): Bucket name. Uses first configured bucket when None.
        key (str): Object key to inspect.

    Returns:
        dict[str, Any]: Object metadata such as size, content type, ETag, and custom metadata.
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

    # Select a subset plus pass-through Metadata
    fields = [
        "ContentLength",
        "ContentType",
        "ETag",
        "LastModified",
        "StorageClass",
        "ContentEncoding",
        "ContentLanguage",
        "CacheControl",
        "Expires",
        "VersionId",
        "Metadata",
    ]
    result: dict[str, Any] = {}
    for f in fields:
        if f in resp:
            result[f] = _serialize(resp[f])
    return result


@mcp.tool()
async def list_skypilot_jobs() -> str | None:
    """List active Skypilot jobs for the logged in user.

    Returns:
        str: Text about jobs statuses returned by `sky status --verbose`.
    """
    result = subprocess.run(["sky", "status", "--verbose"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"return code: {result.returncode} - {result.stderr.strip()}"


@mcp.tool()
async def get_wandb_run_url(run_name: str, project: str | None = None, entity: str | None = None) -> str:
    """Get the URL for a specific Weights & Biases run by name.

    Args:
        run_name (str): The run name or id to fetch.
        project (str | None): W&B project name. Uses configured default when None.
        entity (str | None): W&B entity (organization or user). Uses configured default when None.

    Returns:
        str: The URL for the run.
    """
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    wandb_api = WandbApi()
    run = wandb_api.runs(f"{entity}/{project}/runs/${run_name}")
    return run.url


@mcp.tool()
async def whoami(dev_mode: bool = False) -> str:
    """Get the current user's email."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        try:
            result = await client.validate_authenticated()
            return result
        except ConnectionError as e:
            return f"Error: {e}"


# Dashboard Tools
@mcp.tool()
async def list_saved_dashboards(dev_mode: bool = False) -> dict[str, Any]:
    """List all saved dashboards."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.list_saved_dashboards()
        return result.model_dump()


@mcp.tool()
async def create_saved_dashboard(
    name: str,
    type: str,
    dashboard_state: dict[str, Any],
    description: str | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Create a new saved dashboard (always creates a new row, even if name is duplicate)."""
    from metta.app_backend.routes.dashboard_routes import SavedDashboardCreate

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        dashboard_data = SavedDashboardCreate(
            name=name,
            type=type,
            dashboard_state=dashboard_state,
            description=description,
        )
        result = await client.create_saved_dashboard(dashboard_data)
        return result.model_dump()


@mcp.tool()
async def get_saved_dashboard(dashboard_id: str, dev_mode: bool = False) -> dict[str, Any]:
    """Get a specific saved dashboard by ID."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_saved_dashboard(dashboard_id)
        return result.model_dump()


@mcp.tool()
async def update_saved_dashboard(
    dashboard_id: str,
    dashboard_state: dict[str, Any],
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Update an existing saved dashboard."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.update_saved_dashboard(dashboard_id, dashboard_state)
        return result.model_dump()


@mcp.tool()
async def delete_saved_dashboard(dashboard_id: str, dev_mode: bool = False) -> dict[str, str]:
    """Delete a saved dashboard."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.delete_saved_dashboard(dashboard_id)
        return result.model_dump()


# Entity Routes (Training Runs)
@mcp.tool()
async def get_training_runs(dev_mode: bool = False) -> dict[str, Any]:
    """Get all training runs."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_training_runs()
        return result.model_dump()


@mcp.tool()
async def get_training_run(run_id: str, dev_mode: bool = False) -> dict[str, Any]:
    """Get a specific training run by ID."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_training_run(run_id)
        return result.model_dump()


@mcp.tool()
async def update_training_run_description(run_id: str, description: str, dev_mode: bool = False) -> dict[str, Any]:
    """Update the description of a training run."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.update_training_run_description(run_id, description)
        return result.model_dump()


@mcp.tool()
async def update_training_run_tags(run_id: str, tags: list[str] | str, dev_mode: bool = False) -> dict[str, Any]:
    """Update the tags of a training run."""
    tags_list = _parse_str_or_list(tags)
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.update_training_run_tags(run_id, tags_list)
        return result.model_dump()


@mcp.tool()
async def get_training_run_policies(run_id: str, dev_mode: bool = False) -> list[dict[str, Any]]:
    """Get policies for a training run with epoch information."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_training_run_policies(run_id)
        return [policy.model_dump() for policy in result.policies]


# Task Management Tools
@mcp.tool()
async def create_task(
    policy_id: str,
    sim_suite: str,
    attributes: dict[str, Any] | None = None,
    git_hash: str | None = None,
    env_overrides: dict[str, Any] | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Create Task"""
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
        result = await client.create_task(task_data)
        return result.model_dump()


@mcp.tool()
async def get_latest_assigned_task_for_worker(assignee: str, dev_mode: bool = False) -> dict[str, Any] | None:
    """Get Latest Assigned Task For Worker"""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_latest_assigned_task_for_worker(assignee)
        return result.model_dump() if result else None


@mcp.tool()
async def get_available_tasks(limit: int = 200, dev_mode: bool = False) -> dict[str, Any]:
    """Get Available Tasks"""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_available_tasks(limit=limit)
        return result.model_dump()


@mcp.tool()
async def claim_tasks(
    tasks: list[str] | str,
    assignee: str,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Claim Tasks"""
    import uuid

    task_uuids = [uuid.UUID(task_id) for task_id in _parse_str_or_list(tasks)]

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.claim_tasks(task_uuids, assignee)
        return result.model_dump()


@mcp.tool()
async def get_claimed_tasks(assignee: str | None = None, dev_mode: bool = False) -> dict[str, Any]:
    """Get Claimed Tasks"""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_claimed_tasks(assignee)
        return result.model_dump()


@mcp.tool()
async def get_git_hashes_for_workers(assignees: list[str] | str, dev_mode: bool = False) -> dict[str, Any]:
    """Get Git Hashes For Workers"""
    assignee_list = _parse_str_or_list(assignees)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_git_hashes_for_workers(assignee_list)
        return result.model_dump()


@mcp.tool()
async def get_all_tasks(
    limit: int = 500,
    git_hash: str | None = None,
    policy_ids: list[str] | str | None = None,
    sim_suites: list[str] | str | None = None,
    statuses: list[str] | str | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Get All Tasks"""
    import uuid

    policy_ids_list = None
    if policy_ids:
        policy_ids_list = [uuid.UUID(pid) for pid in _parse_str_or_list(policy_ids)]

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_all_tasks(
            limit=limit,
            statuses=_parse_optional_str_or_list(statuses),
            git_hash=git_hash,
            policy_ids=policy_ids_list,
            sim_suites=_parse_optional_str_or_list(sim_suites),
        )
        return result.model_dump()


@mcp.tool()
async def update_task_statuses(
    updates: dict[str, dict[str, Any]],
    require_assignee: str | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Update Task Statuses"""
    import uuid

    # Convert string keys to UUIDs and leave update dicts as-is for the client to handle
    converted_updates = {}
    for task_id_str, update_dict in updates.items():
        task_id = uuid.UUID(task_id_str)
        converted_updates[task_id] = update_dict

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.update_task_statuses(converted_updates, require_assignee)
        return result.model_dump()


# Leaderboard Tools
@mcp.tool()
async def list_leaderboards(dev_mode: bool = False) -> dict[str, Any]:
    """List all leaderboards for the current user."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.list_leaderboards()
        return result.model_dump()


@mcp.tool()
async def create_leaderboard(
    name: str,
    evals: list[str] | str,
    metric: str,
    start_date: str,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Create a new leaderboard."""
    from metta.app_backend.routes.leaderboard_routes import LeaderboardCreateOrUpdate

    eval_list = _parse_str_or_list(evals)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        leaderboard_data = LeaderboardCreateOrUpdate(
            name=name,
            evals=eval_list,
            metric=metric,
            start_date=start_date,
        )
        result = await client.create_leaderboard(leaderboard_data)
        return result.model_dump()


@mcp.tool()
async def get_leaderboard(leaderboard_id: str, dev_mode: bool = False) -> dict[str, Any]:
    """Get a specific leaderboard by ID."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_leaderboard(leaderboard_id)
        return result.model_dump()


@mcp.tool()
async def update_leaderboard(
    leaderboard_id: str,
    name: str,
    evals: list[str] | str,
    metric: str,
    start_date: str,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Update a leaderboard."""
    from metta.app_backend.routes.leaderboard_routes import LeaderboardCreateOrUpdate

    eval_list = _parse_str_or_list(evals)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        leaderboard_data = LeaderboardCreateOrUpdate(
            name=name,
            evals=eval_list,
            metric=metric,
            start_date=start_date,
        )
        result = await client.update_leaderboard(leaderboard_id, leaderboard_data)
        return result.model_dump()


@mcp.tool()
async def delete_leaderboard(leaderboard_id: str, dev_mode: bool = False) -> dict[str, str]:
    """Delete a leaderboard."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.delete_leaderboard(leaderboard_id)
        return result.model_dump()


# Stats and Metrics Tools
@mcp.tool()
async def get_policy_ids(policy_names: list[str] | str, dev_mode: bool = False) -> dict[str, Any]:
    """Get policy IDs for given policy names."""
    policy_names_list = _parse_str_or_list(policy_names)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_policy_ids(policy_names_list)
        return result.model_dump()


@mcp.tool()
async def create_training_run(
    name: str,
    attributes: dict[str, str] | None = None,
    url: str | None = None,
    description: str | None = None,
    tags: list[str] | str | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Create a new training run."""
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
        result = await client.create_training_run(training_run_data)
        return result.model_dump()


@mcp.tool()
async def create_epoch(
    run_id: str,
    start_training_epoch: int,
    end_training_epoch: int,
    attributes: dict[str, str] | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Create a new policy epoch."""
    from metta.app_backend.routes.stats_routes import EpochCreate

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        epoch_data = EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        result = await client.create_epoch(run_id, epoch_data)
        return result.model_dump()


@mcp.tool()
async def create_policy(
    name: str,
    description: str | None = None,
    url: str | None = None,
    epoch_id: str | None = None,
    dev_mode: bool = False,
) -> dict[str, Any]:
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
        result = await client.create_policy(policy_data)
        return result.model_dump()


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
) -> dict[str, Any]:
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
        result = await client.record_episode(episode_data)
        return result.model_dump()


# Sweep Tools
@mcp.tool()
async def create_sweep(
    sweep_name: str,
    project: str,
    entity: str,
    wandb_sweep_id: str,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Initialize a new sweep or return existing sweep info (idempotent)."""
    from metta.app_backend.routes.sweep_routes import SweepCreateRequest

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        request_data = SweepCreateRequest(
            project=project,
            entity=entity,
            wandb_sweep_id=wandb_sweep_id,
        )
        result = await client.create_sweep(sweep_name, request_data)
        return result.model_dump()


@mcp.tool()
async def get_sweep(sweep_name: str, dev_mode: bool = False) -> dict[str, Any]:
    """Get sweep information by name."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_sweep(sweep_name)
        return result.model_dump()


@mcp.tool()
async def get_next_run_id(sweep_name: str, dev_mode: bool = False) -> dict[str, Any]:
    """Get the next run ID for a sweep (atomic operation)."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_next_run_id(sweep_name)
        return result.model_dump()


# Token Management Tools
@mcp.tool()
async def list_tokens(dev_mode: bool = False) -> dict[str, Any]:
    """List all machine tokens for the authenticated user."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.list_tokens()
        return result.model_dump()


@mcp.tool()
async def create_token(name: str, dev_mode: bool = False) -> dict[str, Any]:
    """Create a new machine token for the authenticated user."""
    from metta.app_backend.routes.token_routes import TokenCreate

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        token_data = TokenCreate(name=name)
        result = await client.create_token(token_data)
        return result.model_dump()


@mcp.tool()
async def delete_token(token_id: str, dev_mode: bool = False) -> dict[str, str]:
    """Delete a machine token for the authenticated user."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.delete_token(token_id)
        return result


# Score Routes
@mcp.tool()
async def get_policy_scores(
    policy_ids: list[str] | str,
    eval_names: list[str] | str,
    metrics: list[str] | str,
    dev_mode: bool = False,
) -> dict[str, Any]:
    """Get Policy Scores"""
    import uuid

    policy_ids_list = [uuid.UUID(pid) for pid in _parse_str_or_list(policy_ids)]
    eval_names_list = _parse_str_or_list(eval_names)
    metrics_list = _parse_str_or_list(metrics)

    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        result = await client.get_policy_scores(policy_ids_list, eval_names_list, metrics_list)
        return result.model_dump()


# Enhanced Scorecard Generation
@mcp.tool()
async def generate_policy_scorecard(
    training_run_ids: list[str] | str,
    run_free_policy_ids: list[str] | str,
    eval_names: list[str] | str,
    metric: str,
    training_run_policy_selector: Literal["best", "latest"] = "latest",
    dev_mode: bool = False,
) -> dict[str, Any]:
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
        return result.model_dump()


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

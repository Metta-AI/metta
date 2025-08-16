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

CONFIG_PATH = Path(__file__).resolve().parent / "metta.mcp.json"
try:
    CONFIG = json.loads(Path(CONFIG_PATH).read_text())
except Exception:
    CONFIG = {}

# mcp = FastMCP("metta")
mcp = FastMCP("metta")


def _get_backend_url(dev_mode: bool) -> str:
    if dev_mode:
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
async def list_skypilot_jobs() -> list[dict[str, Any]]:
    """List active Skypilot jobs for the logged in user.

    Returns:
        list[dict[str, Any]]: Job entries returned by `sky status --json`.
    """
    result = subprocess.run(["sky", "status", "--json"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return data


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
    run = wandb_api.runs(f"{entity}/{project}/runs/{run_name}")
    return f"https://wandb.ai/{entity}/{project}/runs/{run.id}"


@mcp.tool()
async def whoami(dev_mode: bool = False) -> str:
    """Get the current user's email."""
    async with ScorecardClient(backend_url=_get_backend_url(dev_mode)) as client:
        try:
            result = await client.validate_authenticated()
            return result
        except ConnectionError as e:
            return f"Error: {e}"


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

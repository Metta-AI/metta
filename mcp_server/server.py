from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import boto3
import wandb
from mcp.server import FastMCP

from metta.app_backend.clients.scorecard_client import ScorecardClient

CONFIG_PATH = Path(__file__).resolve().parents[1] / "metta.mcp.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

mcp = FastMCP("metta")


def _get_scorecard_client() -> ScorecardClient:
    return ScorecardClient()


@mcp.tool()
async def search_policies(
    search: str | None = None,
    policy_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Search policies in the app backend scorecard."""
    async with _get_scorecard_client() as client:
        result = await client.search_policies(
            search=search,
            policy_type=policy_type,
            limit=limit,
            offset=offset,
        )
    return result.model_dump()


@mcp.tool()
async def list_wandb_runs(
    project: str | None = None,
    entity: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List recent runs from the configured Weights & Biases project."""
    cfg = CONFIG["resources"]["wandb"]
    entity = entity or cfg["entity"]
    project = project or cfg["project"]
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")[:limit]
    return [{"id": run.id, "name": run.name, "state": run.state} for run in runs]


@mcp.tool()
async def list_s3_objects(
    bucket: str | None = None,
    prefix: str | None = None,
    limit: int = 100,
) -> list[str]:
    """List objects from the configured AWS S3 bucket."""
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
async def list_skypilot_jobs() -> list[dict[str, Any]]:
    """List active Skypilot jobs."""
    result = subprocess.run(["sky", "status", "--json"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return data


if __name__ == "__main__":
    mcp.run()

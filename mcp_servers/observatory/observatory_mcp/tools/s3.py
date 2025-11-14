"""S3 tool handlers."""

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from botocore.exceptions import ClientError

from observatory_mcp.analyzers import s3_analyzer
from observatory_mcp.utils import format_error_response, format_success_response

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client as BotoS3Client
    from wandb import Api

    from metta.utils.s3 import S3Store

logger = logging.getLogger(__name__)


async def list_s3_checkpoints(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
    max_keys: int = 1000,
) -> str:
    """List checkpoints in S3 bucket/prefix."""
    if prefix:
        s3_prefix = prefix
    elif run_name:
        s3_prefix = f"checkpoints/{run_name}/"
    else:
        s3_prefix = "checkpoints/"

    logger.info(f"Listing S3 checkpoints: s3://{s3_store.bucket}/{s3_prefix}")

    checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=max_keys)
    checkpoints.sort(key=lambda x: x["last_modified"], reverse=True)

    data = {
        "bucket": s3_store.bucket,
        "prefix": s3_prefix,
        "checkpoints": checkpoints,
        "count": len(checkpoints),
    }

    logger.info(f"list_s3_checkpoints completed ({len(checkpoints)} checkpoints)")
    return format_success_response(data)


async def get_s3_checkpoint_metadata(
    s3_store: "S3Store",
    key: str,
) -> str:
    """Get metadata for a specific S3 checkpoint."""
    logger.info(f"Getting S3 checkpoint metadata: s3://{s3_store.bucket}/{key}")

    metadata = s3_store.get_object_metadata(key)
    if not metadata or not metadata.get("exists"):
        return format_error_response(
            FileNotFoundError(f"Checkpoint not found: s3://{s3_store.bucket}/{key}"),
            "get_s3_checkpoint_metadata",
            f"Checkpoint s3://{s3_store.bucket}/{key} does not exist",
        )

    filename = key.split("/")[-1]
    checkpoint_metadata = metadata.copy()

    parsed_metadata = s3_store.parse_checkpoint_filename(filename)
    if parsed_metadata:
        checkpoint_metadata.update(parsed_metadata)

    logger.info("get_s3_checkpoint_metadata completed")
    return format_success_response(checkpoint_metadata)


async def get_s3_checkpoint_url(
    s3_store: "S3Store",
    key: str,
    expires_in: int = 3600,
) -> str:
    """Generate presigned URL for downloading a checkpoint."""
    logger.info(f"Generating presigned URL for: s3://{s3_store.bucket}/{key}")

    url = s3_store.generate_presigned_url(key=key, expires_in=expires_in)

    data = {
        "bucket": s3_store.bucket,
        "key": key,
        "url": url,
        "expires_in": expires_in,
    }

    logger.info("get_s3_checkpoint_url completed")
    return format_success_response(data)


async def list_s3_replays(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
    max_keys: int = 1000,
) -> str:
    """List replay files in S3 bucket/prefix."""
    if prefix:
        s3_prefix = prefix
    elif run_name:
        s3_prefix = f"replays/{run_name}/"
    else:
        s3_prefix = "replays/"

    logger.info(f"Listing S3 replays: s3://{s3_store.bucket}/{s3_prefix}")

    replays = s3_store.list_replays(prefix=s3_prefix, max_keys=max_keys)
    replays.sort(key=lambda x: x["last_modified"], reverse=True)

    data = {
        "bucket": s3_store.bucket,
        "prefix": s3_prefix,
        "replays": replays,
        "count": len(replays),
    }

    logger.info(f"list_s3_replays completed ({len(replays)} replays)")
    return format_success_response(data)


async def check_s3_object_exists(
    s3_store: "S3Store",
    key: str,
) -> str:
    """Check if an S3 object exists and return metadata if it does."""
    logger.info(f"Checking S3 object existence: s3://{s3_store.bucket}/{key}")

    exists = s3_store.object_exists(key)
    if exists:
        metadata = s3_store.get_object_metadata(key)
        if metadata:
            metadata["exists"] = True
        else:
            metadata = {
                "exists": True,
                "key": key,
                "uri": f"s3://{s3_store.bucket}/{key}",
            }
    else:
        metadata = {
            "exists": False,
            "key": key,
            "uri": f"s3://{s3_store.bucket}/{key}",
        }

    logger.info(f"check_s3_object_exists completed (exists={exists})")
    return format_success_response(metadata)


async def analyze_s3_checkpoint_progression(
    s3_store: "S3Store",
    run_name: str,
    prefix: Optional[str] = None,
) -> str:
    """Analyze checkpoint progression over time for a training run."""
    if prefix:
        s3_prefix = prefix
    else:
        s3_prefix = f"checkpoints/{run_name}/"

    logger.info(f"Analyzing S3 checkpoint progression: s3://{s3_store.bucket}/{s3_prefix}")

    checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)
    analysis = s3_analyzer.analyze_checkpoint_progression(checkpoints)
    analysis["run_name"] = run_name
    analysis["prefix"] = s3_prefix

    logger.info("analyze_s3_checkpoint_progression completed")
    return format_success_response(analysis)


async def find_best_s3_checkpoint(
    s3_store: "S3Store",
    run_name: str,
    criteria: str = "latest",
    prefix: Optional[str] = None,
) -> str:
    """Find best checkpoint by criteria."""
    if prefix:
        s3_prefix = prefix
    else:
        s3_prefix = f"checkpoints/{run_name}/"

    logger.info(f"Finding best S3 checkpoint: s3://{s3_store.bucket}/{s3_prefix}, criteria={criteria}")

    checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)
    best_checkpoint = s3_analyzer.find_best_checkpoint(checkpoints, criteria)

    if not best_checkpoint:
        return format_error_response(
            ValueError("No checkpoints found"),
            "find_best_s3_checkpoint",
            f"No checkpoints found in s3://{s3_store.bucket}/{s3_prefix}",
        )

    data = {
        "run_name": run_name,
        "criteria": criteria,
        "checkpoint": best_checkpoint,
    }

    logger.info("find_best_s3_checkpoint completed")
    return format_success_response(data)


async def analyze_s3_checkpoint_usage(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
    time_window_days: int = 30,
) -> str:
    """Analyze checkpoint usage patterns."""
    if prefix:
        s3_prefix = prefix
    elif run_name:
        s3_prefix = f"checkpoints/{run_name}/"
    else:
        s3_prefix = "checkpoints/"

    logger.info(f"Analyzing S3 checkpoint usage: s3://{s3_store.bucket}/{s3_prefix}")

    checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)
    analysis = s3_analyzer.analyze_checkpoint_usage(checkpoints, time_window_days)
    analysis["prefix"] = s3_prefix

    logger.info("analyze_s3_checkpoint_usage completed")
    return format_success_response(analysis)


async def get_s3_checkpoint_statistics(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    """Get statistics about checkpoints."""
    if prefix:
        s3_prefix = prefix
    elif run_name:
        s3_prefix = f"checkpoints/{run_name}/"
    else:
        s3_prefix = "checkpoints/"

    logger.info(f"Getting S3 checkpoint statistics: s3://{s3_store.bucket}/{s3_prefix}")

    checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)
    stats = s3_analyzer.get_checkpoint_statistics(checkpoints)
    stats["prefix"] = s3_prefix

    logger.info("get_s3_checkpoint_statistics completed")
    return format_success_response(stats)


async def compare_s3_checkpoints_across_runs(
    s3_store: "S3Store",
    run_names: list[str],
) -> str:
    """Compare checkpoints across multiple runs."""
    logger.info(f"Comparing S3 checkpoints across {len(run_names)} runs")

    runs_data: dict[str, list[dict[str, Any]]] = {}

    for run_name in run_names:
        s3_prefix = f"checkpoints/{run_name}/"
        checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)
        runs_data[run_name] = checkpoints

    comparison = s3_analyzer.compare_checkpoints_across_runs(runs_data)

    logger.info("compare_s3_checkpoints_across_runs completed")
    return format_success_response(comparison)


async def link_s3_checkpoint_to_wandb_run(
    s3_client: "BotoS3Client",
    bucket: str,
    wandb_api: "Api",
    key: str,
    entity: str,
    project: str,
) -> str:
    """Link an S3 checkpoint to its WandB run."""
    logger.info(f"Linking S3 checkpoint to WandB run: {key}")

    run_name = None
    if "/checkpoints/" in key:
        parts = key.split("/checkpoints/")
        if len(parts) > 1:
            run_name = parts[1].split("/")[0]

    if not run_name:
        return format_error_response(
            ValueError("Could not extract run name from checkpoint key"),
            "link_s3_checkpoint_to_wandb_run",
            f"Could not extract run name from key: {key}",
        )

    runs = wandb_api.runs(f"{entity}/{project}", filters={"name": run_name})
    run = next(iter(runs), None)

    if not run:
        return format_error_response(
            ValueError(f"WandB run not found for run name: {run_name}"),
            "link_s3_checkpoint_to_wandb_run",
            f"No WandB run found matching: {run_name}",
        )

    data = {
        "checkpoint": {
            "key": key,
            "uri": f"s3://{bucket}/{key}",
        },
        "wandb_run": {
            "id": run.id,
            "name": run.name,
            "url": run.url,
            "state": run.state,
        },
    }

    logger.info("link_s3_checkpoint_to_wandb_run completed")
    return format_success_response(data)

async def link_s3_checkpoint_to_skypilot_job(
    s3_client: "BotoS3Client",
    bucket: str,
    key: str,
) -> str:
    """Link an S3 checkpoint to its Skypilot job."""
    logger.info(f"Linking S3 checkpoint to Skypilot job: {key}")

    run_name = None
    if "/checkpoints/" in key:
        parts = key.split("/checkpoints/")
        if len(parts) > 1:
            run_name = parts[1].split("/")[0]

    if not run_name:
        return format_error_response(
            ValueError("Could not extract run name from checkpoint key"),
            "link_s3_checkpoint_to_skypilot_job",
            f"Could not extract run name from key: {key}",
        )

    import subprocess

    cmd = ["sky", "jobs", "queue", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

    if returncode != 0:
        raise RuntimeError(f"sky jobs queue failed: {stderr}")

    try:
        jobs_data = json.loads(stdout)
    except json.JSONDecodeError:
        from observatory_mcp.tools.skypilot import _parse_sky_jobs_text_output

        jobs_data = _parse_sky_jobs_text_output(stdout)

    matching_jobs = []
    for job in jobs_data:
        job_name = job.get("name", "")
        if run_name in job_name or job_name in run_name:
            matching_jobs.append(job)

    data = {
        "checkpoint": {
            "key": key,
            "uri": f"s3://{bucket}/{key}",
        },
        "jobs": matching_jobs,
        "count": len(matching_jobs),
    }

    logger.info(f"link_s3_checkpoint_to_skypilot_job completed ({len(matching_jobs)} jobs)")
    return format_success_response(data)

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
    """List checkpoints in S3 bucket/prefix.

    Args:
        s3_store: S3Store instance
        run_name: Optional training run name to filter by
        prefix: Optional S3 prefix (overrides run_name if both provided)
        max_keys: Maximum number of objects to return (default: 1000)

    Returns:
        JSON string with list of checkpoints and metadata
    """
    try:
        if prefix:
            s3_prefix = prefix
        elif run_name:
            s3_prefix = f"checkpoints/{run_name}/"
        else:
            s3_prefix = "checkpoints/"

        logger.info(f"Listing S3 checkpoints: s3://{s3_store.bucket}/{s3_prefix}")

        # Use store's list_checkpoints method
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

    except Exception as e:
        logger.warning(f"list_s3_checkpoints failed: {e}")
        return format_error_response(e, "list_s3_checkpoints")


async def get_s3_checkpoint_metadata(
    s3_store: "S3Store",
    key: str,
) -> str:
    """Get metadata for a specific S3 checkpoint.

    Args:
        s3_store: S3Store instance
        key: S3 object key (full path)

    Returns:
        JSON string with checkpoint metadata
    """
    try:
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

        # Parse checkpoint metadata
        parsed_metadata = s3_store.parse_checkpoint_filename(filename)
        if parsed_metadata:
            checkpoint_metadata.update(parsed_metadata)

        logger.info("get_s3_checkpoint_metadata completed")
        return format_success_response(checkpoint_metadata)

    except Exception as e:
        logger.warning(f"get_s3_checkpoint_metadata failed: {e}")
        return format_error_response(e, "get_s3_checkpoint_metadata")


async def get_s3_checkpoint_url(
    s3_store: "S3Store",
    key: str,
    expires_in: int = 3600,
) -> str:
    """Generate presigned URL for downloading a checkpoint.

    Args:
        s3_store: S3Store instance
        key: S3 object key (full path)
        expires_in: URL expiration time in seconds (default: 3600 = 1 hour)

    Returns:
        JSON string with presigned URL
    """
    try:
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

    except Exception as e:
        logger.warning(f"get_s3_checkpoint_url failed: {e}")
        return format_error_response(e, "get_s3_checkpoint_url")


async def list_s3_replays(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
    max_keys: int = 1000,
) -> str:
    """List replay files in S3 bucket/prefix.

    Args:
        s3_store: S3Store instance
        run_name: Optional training run name to filter by
        prefix: Optional S3 prefix (overrides run_name if both provided)
        max_keys: Maximum number of objects to return (default: 1000)

    Returns:
        JSON string with list of replay files
    """
    try:
        if prefix:
            s3_prefix = prefix
        elif run_name:
            s3_prefix = f"replays/{run_name}/"
        else:
            s3_prefix = "replays/"

        logger.info(f"Listing S3 replays: s3://{s3_store.bucket}/{s3_prefix}")

        # Use store's list_replays method
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

    except Exception as e:
        logger.warning(f"list_s3_replays failed: {e}")
        return format_error_response(e, "list_s3_replays")


async def check_s3_object_exists(
    s3_client: "BotoS3Client",
    bucket: str,
    key: str,
) -> str:
    """Check if an S3 object exists and return metadata if it does.

    Args:
        s3_client: S3 client instance
        key: S3 object key (full path)

    Returns:
        JSON string with existence status and metadata if exists
    """
    try:
        logger.info(f"Checking S3 object existence: s3://{bucket}/{key}")

        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
            exists = True
            metadata = {
                "exists": True,
                "key": key,
                "uri": f"s3://{bucket}/{key}",
                "size": response["ContentLength"],
                "last_modified": response["LastModified"].isoformat(),
                "etag": response["ETag"].strip('"'),
                "content_type": response.get("ContentType", "application/octet-stream"),
            }
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                exists = False
                metadata = {
                    "exists": False,
                    "key": key,
                    "uri": f"s3://{bucket}/{key}",
                }
            else:
                raise

        logger.info(f"check_s3_object_exists completed (exists={exists})")
        return format_success_response(metadata)

    except Exception as e:
        logger.warning(f"check_s3_object_exists failed: {e}")
        return format_error_response(e, "check_s3_object_exists")


async def analyze_s3_checkpoint_progression(
    s3_store: "S3Store",
    run_name: str,
    prefix: Optional[str] = None,
) -> str:
    """Analyze checkpoint progression over time for a training run.

    Args:
        s3_store: S3Store instance
        run_name: Training run name
        prefix: Optional S3 prefix (overrides run_name if provided)

    Returns:
        JSON string with progression analysis
    """
    try:
        if prefix:
            s3_prefix = prefix
        else:
            s3_prefix = f"checkpoints/{run_name}/"

        logger.info(f"Analyzing S3 checkpoint progression: s3://{s3_store.bucket}/{s3_prefix}")

        # Use store's list_checkpoints method
        checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)

        analysis = s3_analyzer.analyze_checkpoint_progression(checkpoints)
        analysis["run_name"] = run_name
        analysis["prefix"] = s3_prefix

        logger.info("analyze_s3_checkpoint_progression completed")
        return format_success_response(analysis)

    except Exception as e:
        logger.warning(f"analyze_s3_checkpoint_progression failed: {e}")
        return format_error_response(e, "analyze_s3_checkpoint_progression")


async def find_best_s3_checkpoint(
    s3_store: "S3Store",
    run_name: str,
    criteria: str = "latest",
    prefix: Optional[str] = None,
) -> str:
    """Find best checkpoint by criteria.

    Args:
        s3_store: S3Store instance
        run_name: Training run name
        criteria: Criteria to use ("latest", "largest", "smallest", "earliest")
        prefix: Optional S3 prefix (overrides run_name if provided)

    Returns:
        JSON string with best checkpoint
    """
    try:
        if prefix:
            s3_prefix = prefix
        else:
            s3_prefix = f"checkpoints/{run_name}/"

        logger.info(f"Finding best S3 checkpoint: s3://{s3_store.bucket}/{s3_prefix}, criteria={criteria}")

        # Use store's list_checkpoints method
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

    except Exception as e:
        logger.warning(f"find_best_s3_checkpoint failed: {e}")
        return format_error_response(e, "find_best_s3_checkpoint")


async def analyze_s3_checkpoint_usage(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
    time_window_days: int = 30,
) -> str:
    """Analyze checkpoint usage patterns.

    Args:
        s3_store: S3Store instance
        run_name: Optional training run name to filter by
        prefix: Optional S3 prefix (overrides run_name if both provided)
        time_window_days: Time window in days to analyze (default: 30)

    Returns:
        JSON string with usage analysis
    """
    try:
        if prefix:
            s3_prefix = prefix
        elif run_name:
            s3_prefix = f"checkpoints/{run_name}/"
        else:
            s3_prefix = "checkpoints/"

        logger.info(f"Analyzing S3 checkpoint usage: s3://{s3_store.bucket}/{s3_prefix}")

        # Use store's list_checkpoints method
        checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)

        analysis = s3_analyzer.analyze_checkpoint_usage(checkpoints, time_window_days)
        analysis["prefix"] = s3_prefix

        logger.info("analyze_s3_checkpoint_usage completed")
        return format_success_response(analysis)

    except Exception as e:
        logger.warning(f"analyze_s3_checkpoint_usage failed: {e}")
        return format_error_response(e, "analyze_s3_checkpoint_usage")


async def get_s3_checkpoint_statistics(
    s3_store: "S3Store",
    run_name: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    """Get statistics about checkpoints.

    Args:
        s3_store: S3Store instance
        run_name: Optional training run name to filter by
        prefix: Optional S3 prefix (overrides run_name if both provided)

    Returns:
        JSON string with checkpoint statistics
    """
    try:
        if prefix:
            s3_prefix = prefix
        elif run_name:
            s3_prefix = f"checkpoints/{run_name}/"
        else:
            s3_prefix = "checkpoints/"

        logger.info(f"Getting S3 checkpoint statistics: s3://{s3_store.bucket}/{s3_prefix}")

        # Use store's list_checkpoints method
        checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)

        stats = s3_analyzer.get_checkpoint_statistics(checkpoints)
        stats["prefix"] = s3_prefix

        logger.info("get_s3_checkpoint_statistics completed")
        return format_success_response(stats)

    except Exception as e:
        logger.warning(f"get_s3_checkpoint_statistics failed: {e}")
        return format_error_response(e, "get_s3_checkpoint_statistics")


async def compare_s3_checkpoints_across_runs(
    s3_store: "S3Store",
    run_names: list[str],
) -> str:
    """Compare checkpoints across multiple runs.

    Args:
        s3_store: S3Store instance
        run_names: List of training run names to compare

    Returns:
        JSON string with comparison analysis
    """
    try:
        logger.info(f"Comparing S3 checkpoints across {len(run_names)} runs")

        runs_data: dict[str, list[dict[str, Any]]] = {}

        for run_name in run_names:
            s3_prefix = f"checkpoints/{run_name}/"
            # Use store's list_checkpoints method
            checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)
            runs_data[run_name] = checkpoints

        comparison = s3_analyzer.compare_checkpoints_across_runs(runs_data)

        logger.info("compare_s3_checkpoints_across_runs completed")
        return format_success_response(comparison)

    except Exception as e:
        logger.warning(f"compare_s3_checkpoints_across_runs failed: {e}")
        return format_error_response(e, "compare_s3_checkpoints_across_runs")


async def link_s3_checkpoint_to_wandb_run(
    s3_client: "BotoS3Client",
    bucket: str,
    wandb_api: "Api",
    key: str,
    entity: str,
    project: str,
) -> str:
    """Link an S3 checkpoint to its WandB run.

    Args:
        s3_client: S3 client instance
        wandb_api: WandB API instance
        key: S3 checkpoint key
        entity: WandB entity (user/team)
        project: WandB project name

    Returns:
        JSON string with linked WandB run information
    """
    try:
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

    except Exception as e:
        logger.warning(f"link_s3_checkpoint_to_wandb_run failed: {e}")
        return format_error_response(e, "link_s3_checkpoint_to_wandb_run")


async def link_s3_checkpoint_to_skypilot_job(
    s3_client: "BotoS3Client",
    bucket: str,
    key: str,
) -> str:
    """Link an S3 checkpoint to its Skypilot job.

    Args:
        s3_client: S3 client instance
        bucket: S3 bucket name
        key: S3 checkpoint key

    Returns:
        JSON string with linked job information
    """
    try:
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

    except Exception as e:
        logger.warning(f"link_s3_checkpoint_to_skypilot_job failed: {e}")
        return format_error_response(e, "link_s3_checkpoint_to_skypilot_job")

"""Skypilot tool handlers."""

import json
import logging
from typing import Optional

from observatory_mcp.analyzers import skypilot_analyzer
from observatory_mcp.clients.s3_client import S3Client
from observatory_mcp.clients.skypilot_client import SkypilotClient
from observatory_mcp.clients.wandb_client import WandBClient
from observatory_mcp.utils import format_error_response, format_success_response

logger = logging.getLogger(__name__)


async def list_skypilot_jobs(
    skypilot_client: SkypilotClient,
    status: Optional[str] = None,
    limit: int = 100,
) -> str:
    """List Skypilot jobs with optional status filter.

    Args:
        skypilot_client: Skypilot client instance
        status: Optional status filter ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")
        limit: Maximum number of jobs to return (default: 100)

    Returns:
        JSON string with list of jobs and their status
    """
    try:
        logger.info("Listing Skypilot jobs")

        cmd = ["sky", "jobs", "queue", "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs queue failed: {stderr}")

        try:
            jobs_data = json.loads(stdout)
        except json.JSONDecodeError:
            jobs_data = _parse_sky_jobs_text_output(stdout)

        if status:
            jobs_data = [job for job in jobs_data if job.get("status", "").upper() == status.upper()]

        jobs_data = jobs_data[:limit]

        data = {
            "jobs": jobs_data,
            "count": len(jobs_data),
        }

        logger.info(f"list_skypilot_jobs completed ({len(jobs_data)} jobs)")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"list_skypilot_jobs failed: {e}")
        return format_error_response(e, "list_skypilot_jobs")


def _parse_sky_jobs_text_output(text: str) -> list[dict]:
    """Parse sky jobs queue text output (fallback if JSON fails)."""
    jobs = []
    lines = text.strip().split("\n")
    for line in lines:
        if line.strip() and not line.startswith("JOB_ID"):
            parts = line.split()
            if len(parts) >= 3:
                jobs.append(
                    {
                        "job_id": parts[0],
                        "status": parts[1],
                        "name": " ".join(parts[2:]) if len(parts) > 2 else "",
                    }
                )
    return jobs


async def get_skypilot_job_status(
    skypilot_client: SkypilotClient,
    job_id: str,
) -> str:
    """Get detailed status for a specific Skypilot job.

    Args:
        skypilot_client: Skypilot client instance
        job_id: Skypilot job ID

    Returns:
        JSON string with detailed job status
    """
    try:
        logger.info(f"Getting Skypilot job status: {job_id}")

        cmd = ["sky", "jobs", "status", job_id, "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs status failed: {stderr}")

        try:
            job_data = json.loads(stdout)
        except json.JSONDecodeError:
            job_data = {"job_id": job_id, "status": "UNKNOWN", "raw_output": stdout}

        logger.info("get_skypilot_job_status completed")
        return format_success_response(job_data)

    except Exception as e:
        logger.warning(f"get_skypilot_job_status failed: {e}")
        return format_error_response(e, "get_skypilot_job_status")


async def get_skypilot_job_logs(
    skypilot_client: SkypilotClient,
    job_id: str,
    tail_lines: int = 100,
) -> str:
    """Get logs for a Skypilot job.

    Args:
        skypilot_client: Skypilot client instance
        job_id: Skypilot job ID
        tail_lines: Number of lines to return (default: 100)

    Returns:
        JSON string with log content
    """
    try:
        logger.info(f"Getting Skypilot job logs: {job_id}")

        cmd = ["sky", "jobs", "logs", job_id, "--tail", str(tail_lines)]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=60)

        if returncode != 0:
            raise RuntimeError(f"sky jobs logs failed: {stderr}")

        data = {
            "job_id": job_id,
            "log_type": "job",
            "lines": tail_lines,
            "content": stdout,
        }

        logger.info("get_skypilot_job_logs completed")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"get_skypilot_job_logs failed: {e}")
        return format_error_response(e, "get_skypilot_job_logs")


async def analyze_skypilot_job_performance(
    skypilot_client: SkypilotClient,
    limit: int = 100,
) -> str:
    """Analyze job performance trends.

    Args:
        skypilot_client: Skypilot client instance
        limit: Maximum number of jobs to analyze (default: 100)

    Returns:
        JSON string with performance analysis
    """
    try:
        logger.info("Analyzing Skypilot job performance")

        cmd = ["sky", "jobs", "queue", "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs queue failed: {stderr}")

        try:
            jobs_data = json.loads(stdout)
        except json.JSONDecodeError:
            jobs_data = _parse_sky_jobs_text_output(stdout)

        jobs_data = jobs_data[:limit]

        analysis = skypilot_analyzer.analyze_job_performance(jobs_data)

        logger.info("analyze_skypilot_job_performance completed")
        return format_success_response(analysis)

    except Exception as e:
        logger.warning(f"analyze_skypilot_job_performance failed: {e}")
        return format_error_response(e, "analyze_skypilot_job_performance")


async def get_skypilot_resource_utilization(
    skypilot_client: SkypilotClient,
    limit: int = 100,
) -> str:
    """Get resource utilization statistics.

    Args:
        skypilot_client: Skypilot client instance
        limit: Maximum number of jobs to analyze (default: 100)

    Returns:
        JSON string with resource utilization
    """
    try:
        logger.info("Getting Skypilot resource utilization")

        cmd = ["sky", "jobs", "queue", "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs queue failed: {stderr}")

        try:
            jobs_data = json.loads(stdout)
        except json.JSONDecodeError:
            jobs_data = _parse_sky_jobs_text_output(stdout)

        jobs_data = jobs_data[:limit]

        utilization = skypilot_analyzer.get_resource_utilization(jobs_data)

        logger.info("get_skypilot_resource_utilization completed")
        return format_success_response(utilization)

    except Exception as e:
        logger.warning(f"get_skypilot_resource_utilization failed: {e}")
        return format_error_response(e, "get_skypilot_resource_utilization")


async def compare_skypilot_job_configs(
    skypilot_client: SkypilotClient,
    job_ids: list[str],
) -> str:
    """Compare job configurations.

    Args:
        skypilot_client: Skypilot client instance
        job_ids: List of job IDs to compare

    Returns:
        JSON string with configuration comparison
    """
    try:
        logger.info(f"Comparing Skypilot job configs: {len(job_ids)} jobs")

        jobs_data = []
        for job_id in job_ids:
            cmd = ["sky", "jobs", "status", job_id, "--json"]
            stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

            if returncode == 0:
                try:
                    job_data = json.loads(stdout)
                    job_data["job_id"] = job_id
                    jobs_data.append(job_data)
                except json.JSONDecodeError:
                    jobs_data.append({"job_id": job_id, "status": "UNKNOWN", "config": {}})

        comparison = skypilot_analyzer.compare_job_configurations(jobs_data)

        logger.info("compare_skypilot_job_configs completed")
        return format_success_response(comparison)

    except Exception as e:
        logger.warning(f"compare_skypilot_job_configs failed: {e}")
        return format_error_response(e, "compare_skypilot_job_configs")


async def analyze_skypilot_job_failures(
    skypilot_client: SkypilotClient,
    limit: int = 100,
) -> str:
    """Analyze job failure patterns.

    Args:
        skypilot_client: Skypilot client instance
        limit: Maximum number of jobs to analyze (default: 100)

    Returns:
        JSON string with failure analysis
    """
    try:
        logger.info("Analyzing Skypilot job failures")

        cmd = ["sky", "jobs", "queue", "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs queue failed: {stderr}")

        try:
            jobs_data = json.loads(stdout)
        except json.JSONDecodeError:
            jobs_data = _parse_sky_jobs_text_output(stdout)

        jobs_data = jobs_data[:limit]

        analysis = skypilot_analyzer.analyze_job_failures(jobs_data)

        logger.info("analyze_skypilot_job_failures completed")
        return format_success_response(analysis)

    except Exception as e:
        logger.warning(f"analyze_skypilot_job_failures failed: {e}")
        return format_error_response(e, "analyze_skypilot_job_failures")


async def get_skypilot_job_cost_estimates(
    skypilot_client: SkypilotClient,
    limit: int = 100,
) -> str:
    """Get job cost estimates.

    Args:
        skypilot_client: Skypilot client instance
        limit: Maximum number of jobs to analyze (default: 100)

    Returns:
        JSON string with cost estimates
    """
    try:
        logger.info("Getting Skypilot job cost estimates")

        cmd = ["sky", "jobs", "queue", "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs queue failed: {stderr}")

        try:
            jobs_data = json.loads(stdout)
        except json.JSONDecodeError:
            jobs_data = _parse_sky_jobs_text_output(stdout)

        jobs_data = jobs_data[:limit]

        estimates = skypilot_analyzer.get_job_cost_estimates(jobs_data)

        logger.info("get_skypilot_job_cost_estimates completed")
        return format_success_response(estimates)

    except Exception as e:
        logger.warning(f"get_skypilot_job_cost_estimates failed: {e}")
        return format_error_response(e, "get_skypilot_job_cost_estimates")


async def link_skypilot_job_to_wandb_runs(
    skypilot_client: SkypilotClient,
    wandb_client: WandBClient,
    job_id: str,
    entity: str,
    project: str,
) -> str:
    """Link a Skypilot job to its WandB runs.

    Args:
        skypilot_client: Skypilot client instance
        wandb_client: WandB client instance
        job_id: Skypilot job ID
        entity: WandB entity (user/team)
        project: WandB project name

    Returns:
        JSON string with linked WandB runs
    """
    try:
        logger.info(f"Linking Skypilot job to WandB runs: {job_id}")

        cmd = ["sky", "jobs", "status", job_id, "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs status failed: {stderr}")

        try:
            job_data = json.loads(stdout)
        except json.JSONDecodeError:
            job_data = {"job_id": job_id, "name": ""}

        job_name = job_data.get("name", "")

        matching_runs = []
        if job_name:
            runs = wandb_client.api.runs(f"{entity}/{project}", filters={"name": {"$regex": job_name}})
            for run in runs:
                matching_runs.append(
                    {
                        "id": run.id,
                        "name": run.name,
                        "url": run.url,
                        "state": run.state,
                    }
                )

        data = {
            "skypilot_job": {
                "job_id": job_id,
                "name": job_name,
            },
            "wandb_runs": matching_runs,
            "count": len(matching_runs),
        }

        logger.info(f"link_skypilot_job_to_wandb_runs completed ({len(matching_runs)} runs)")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"link_skypilot_job_to_wandb_runs failed: {e}")
        return format_error_response(e, "link_skypilot_job_to_wandb_runs")


async def link_skypilot_job_to_s3_checkpoints(
    skypilot_client: SkypilotClient,
    s3_client: S3Client,
    job_id: str,
) -> str:
    """Link a Skypilot job to its S3 checkpoints.

    Args:
        skypilot_client: Skypilot client instance
        s3_client: S3 client instance
        job_id: Skypilot job ID

    Returns:
        JSON string with linked checkpoints
    """
    try:
        logger.info(f"Linking Skypilot job to S3 checkpoints: {job_id}")

        cmd = ["sky", "jobs", "status", job_id, "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

        if returncode != 0:
            raise RuntimeError(f"sky jobs status failed: {stderr}")

        try:
            job_data = json.loads(stdout)
        except json.JSONDecodeError:
            job_data = {"job_id": job_id, "name": ""}

        job_name = job_data.get("name", "")
        run_name = job_name

        checkpoints = []
        if run_name:
            s3_prefix = f"checkpoints/{run_name}/"
            paginator = s3_client.client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=s3_client.bucket, Prefix=s3_prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".mpt"):
                        filename = key.split("/")[-1]
                        checkpoint_info = {
                            "key": key,
                            "uri": f"s3://{s3_client.bucket}/{key}",
                            "filename": filename,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                        }

                        if ":v" in filename:
                            try:
                                epoch_str = filename.split(":v")[1].replace(".mpt", "")
                                checkpoint_info["epoch"] = int(epoch_str)
                            except ValueError:
                                pass

                        checkpoints.append(checkpoint_info)

        data = {
            "skypilot_job": {
                "job_id": job_id,
                "name": job_name,
            },
            "checkpoints": checkpoints,
            "count": len(checkpoints),
        }

        logger.info(f"link_skypilot_job_to_s3_checkpoints completed ({len(checkpoints)} checkpoints)")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"link_skypilot_job_to_s3_checkpoints failed: {e}")
        return format_error_response(e, "link_skypilot_job_to_s3_checkpoints")

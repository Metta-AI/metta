"""Skypilot tool handlers."""

import json
import logging
import subprocess
from typing import TYPE_CHECKING, Optional

from observatory_mcp.analyzers import skypilot_analyzer
from observatory_mcp.utils import format_error_response, format_success_response

if TYPE_CHECKING:
    from metta.adaptive.stores.wandb import WandbStore
    from metta.utils.s3 import S3Store

logger = logging.getLogger(__name__)


async def list_skypilot_jobs(
    status: Optional[str] = None,
    limit: int = 100,
) -> str:
    """List Skypilot jobs with optional status filter."""
    logger.info("Listing Skypilot jobs")

    cmd = ["sky", "jobs", "queue", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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
    job_id: str,
) -> str:
    """Get detailed status for a specific Skypilot job."""
    logger.info(f"Getting Skypilot job status: {job_id}")

    cmd = ["sky", "jobs", "status", job_id, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

    if returncode != 0:
        raise RuntimeError(f"sky jobs status failed: {stderr}")

    try:
        job_data = json.loads(stdout)
    except json.JSONDecodeError:
        job_data = {"job_id": job_id, "status": "UNKNOWN", "raw_output": stdout}

    logger.info("get_skypilot_job_status completed")
    return format_success_response(job_data)

async def get_skypilot_job_logs(
    job_id: str,
    tail_lines: int = 100,
) -> str:
    """Get logs for a Skypilot job."""
    logger.info(f"Getting Skypilot job logs: {job_id}")

    cmd = ["sky", "jobs", "logs", job_id, "--tail", str(tail_lines)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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

async def analyze_skypilot_job_performance(
    limit: int = 100,
) -> str:
    """Analyze job performance trends."""
    logger.info("Analyzing Skypilot job performance")

    cmd = ["sky", "jobs", "queue", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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

async def get_skypilot_resource_utilization(
    limit: int = 100,
) -> str:
    """Get resource utilization statistics."""
    logger.info("Getting Skypilot resource utilization")

    cmd = ["sky", "jobs", "queue", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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

async def compare_skypilot_job_configs(
    job_ids: list[str],
) -> str:
    """Compare job configurations."""
    logger.info(f"Comparing Skypilot job configs: {len(job_ids)} jobs")

    jobs_data = []
    for job_id in job_ids:
        cmd = ["sky", "jobs", "status", job_id, "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stdout, returncode = result.stdout, result.returncode

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

async def analyze_skypilot_job_failures(
    limit: int = 100,
) -> str:
    """Analyze job failure patterns."""
    logger.info("Analyzing Skypilot job failures")

    cmd = ["sky", "jobs", "queue", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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

async def get_skypilot_job_cost_estimates(
    limit: int = 100,
) -> str:
    """Get job cost estimates."""
    logger.info("Getting Skypilot job cost estimates")

    cmd = ["sky", "jobs", "queue", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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

async def link_skypilot_job_to_wandb_runs(
    wandb_store: "WandbStore",
    job_id: str,
    entity: str,
    project: str,
) -> str:
    """Link a Skypilot job to its WandB runs."""
    logger.info(f"Linking Skypilot job to WandB runs: {job_id}")

    cmd = ["sky", "jobs", "status", job_id, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

    if returncode != 0:
        raise RuntimeError(f"sky jobs status failed: {stderr}")

    try:
        job_data = json.loads(stdout)
    except json.JSONDecodeError:
        job_data = {"job_id": job_id, "name": ""}

    job_name = job_data.get("name", "")

    matching_runs = []
    if job_name:
        runs = wandb_store.list_runs(
            entity=entity,
            project=project,
            filters={"name": {"$regex": job_name}},
            limit=100,  # Reasonable limit for linking
        )
        matching_runs = [
            {
                "id": run.get("id"),
                "name": run.get("name"),
                "url": run.get("url"),
                "state": run.get("state"),
            }
            for run in runs
        ]

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

async def link_skypilot_job_to_s3_checkpoints(
    s3_store: "S3Store",
    job_id: str,
) -> str:
    """Link a Skypilot job to its S3 checkpoints."""
    logger.info(f"Linking Skypilot job to S3 checkpoints: {job_id}")

    cmd = ["sky", "jobs", "status", job_id, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stdout, stderr, returncode = result.stdout, result.stderr, result.returncode

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
        checkpoints = s3_store.list_checkpoints(prefix=s3_prefix, max_keys=1000)

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

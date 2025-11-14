"""Skypilot job analysis functions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def analyze_job_performance(
    jobs_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze job performance trends."""
    if not jobs_data:
        return {"performance": {}, "trends": {}}

    status_counts = {}
    for job in jobs_data:
        status = job.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    success_rate = status_counts.get("SUCCEEDED", 0) / len(jobs_data) if jobs_data else 0.0
    failure_rate = status_counts.get("FAILED", 0) / len(jobs_data) if jobs_data else 0.0

    performance = {
        "total_jobs": len(jobs_data),
        "status_distribution": status_counts,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
    }

    trends = {
        "most_common_status": max(status_counts.items(), key=lambda x: x[1])[0] if status_counts else None,
        "health_score": success_rate * 100,
    }

    return {
        "performance": performance,
        "trends": trends,
    }


def get_resource_utilization(
    jobs_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Get resource utilization statistics."""
    if not jobs_data:
        return {"utilization": {}}

    running_jobs = [j for j in jobs_data if j.get("status") == "RUNNING"]
    pending_jobs = [j for j in jobs_data if j.get("status") == "PENDING"]

    utilization = {
        "total_jobs": len(jobs_data),
        "running": len(running_jobs),
        "pending": len(pending_jobs),
        "utilization_rate": len(running_jobs) / len(jobs_data) if jobs_data else 0.0,
    }

    return {"utilization": utilization}


def compare_job_configurations(
    jobs_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare job configurations."""
    if not jobs_data:
        return {"comparisons": {}}

    configs = {}
    for job in jobs_data:
        job_id = job.get("job_id", "unknown")
        config = job.get("config", {})
        configs[job_id] = config

    comparisons = {
        "total_jobs": len(jobs_data),
        "unique_configs": len(set(str(c) for c in configs.values())),
        "configs": configs,
    }

    return {"comparisons": comparisons}


def analyze_job_failures(
    jobs_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze job failure patterns."""
    if not jobs_data:
        return {"failures": {}}

    failed_jobs = [j for j in jobs_data if j.get("status") == "FAILED"]
    cancelled_jobs = [j for j in jobs_data if j.get("status") == "CANCELLED"]

    failures = {
        "total_jobs": len(jobs_data),
        "failed": len(failed_jobs),
        "cancelled": len(cancelled_jobs),
        "failure_rate": len(failed_jobs) / len(jobs_data) if jobs_data else 0.0,
        "failed_job_ids": [j.get("job_id") for j in failed_jobs],
    }

    return {"failures": failures}


def get_job_cost_estimates(
    jobs_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Get job cost estimates."""
    if not jobs_data:
        return {"cost_estimates": {}}

    running_jobs = [j for j in jobs_data if j.get("status") == "RUNNING"]
    succeeded_jobs = [j for j in jobs_data if j.get("status") == "SUCCEEDED"]

    cost_estimates = {
        "total_jobs": len(jobs_data),
        "running_jobs": len(running_jobs),
        "succeeded_jobs": len(succeeded_jobs),
        "estimated_running_cost": len(running_jobs) * 0.1,
        "estimated_total_cost": len(jobs_data) * 0.1,
    }

    return {"cost_estimates": cost_estimates}

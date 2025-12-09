"""Job to workflow mapping for stable_suite integration.

This module maps stable_suite job names to Datadog schema workflow names.
Mapping is hard-coded based on Nishad's confirmed mapping.
"""

from __future__ import annotations


# Hard-coded mapping based on Nishad's confirmation
JOB_TO_WORKFLOW_MAP = {
    "arena_single_gpu_100m": "multigpu_arena_basic_easy_shaped",
    "arena_multi_gpu_2b": "multinode_learning_progress",
    "cvc_fixed_maps_mettabox_200ep": "local_arena_basic_easy_shaped",
    "cvc_fixed_maps_multi_gpu_2b": "multinode_learning_progress",
    "arena_evaluate": "remote_eval",
}


def map_job_to_workflow(job_name: str) -> str:
    """Map a stable_suite job name to a Datadog workflow name.

    Args:
        job_name: Full job name (e.g., "stable.v0.1.0.arena_single_gpu_100m")

    Returns:
        Workflow name: "multigpu_arena_basic_easy_shaped", "multinode_learning_progress",
        "local_arena_basic_easy_shaped", or "remote_eval"

    Raises:
        ValueError: If job name doesn't match any known pattern
    """
    # Extract job suffix (e.g., "arena_single_gpu_100m" from "stable.v0.1.0.arena_single_gpu_100m")
    parts = job_name.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid job name format: {job_name}")

    # Try to match the last part (job suffix)
    job_suffix = parts[-1]

    # Check direct match first
    if job_suffix in JOB_TO_WORKFLOW_MAP:
        return JOB_TO_WORKFLOW_MAP[job_suffix]

    # Try partial match (in case of variations)
    for job_pattern, workflow in JOB_TO_WORKFLOW_MAP.items():
        if job_pattern in job_suffix or job_suffix in job_pattern:
            return workflow

    raise ValueError(
        f"Unknown job name: {job_name}. "
        f"Known jobs: {list(JOB_TO_WORKFLOW_MAP.keys())}"
    )


def is_training_job(job_name: str) -> bool:
    """Check if a job is a training job (vs eval).

    Args:
        job_name: Job name

    Returns:
        True if training job, False if eval job
    """
    workflow = map_job_to_workflow(job_name)
    return workflow != "remote_eval"


def is_eval_job(job_name: str) -> bool:
    """Check if a job is an eval job.

    Args:
        job_name: Job name

    Returns:
        True if eval job, False otherwise
    """
    return not is_training_job(job_name)

"""Convert stable runner Job results to Datadog MetricSamples."""

from __future__ import annotations

import logging

from devops.datadog.models import MetricSample
from devops.stable.runner import Job, JobStatus

logger = logging.getLogger(__name__)


def _extract_recipe_path(job: Job) -> str | None:
    """Extract recipe path from job command.

    Job commands are like:
    - Local: ["uv", "run", "./tools/run.py", "recipes.prod.arena_basic_easy_shaped.train_100m", ...]
    - Remote: ["uv", "run", "./devops/skypilot/launch.py", "recipes.prod.arena_basic_easy_shaped.train_100m", ...]

    Returns the recipe path (e.g., "recipes.prod.arena_basic_easy_shaped.train_100m") or None if not found.
    """
    if not job.cmd or len(job.cmd) < 4:
        return None

    # The recipe path is always the 4th element (index 3) in the command
    recipe_path = job.cmd[3]
    if recipe_path.startswith("recipes."):
        return recipe_path
    return None


def _normalize_criterion_name(metric: str) -> str:
    """Normalize criterion metric name for tagging.

    Examples:
        "overview/sps" -> "overview_sps"
        "env_agent/heart.gained" -> "env_agent_heart_gained"
    """
    return metric.replace("/", "_").replace(".", "_")


def job_to_metrics(job: Job) -> list[MetricSample]:
    """Convert a Job result to Datadog MetricSamples using unified acceptance schema.

    Emits triples (value, target, status) for:
    - runs_success: Whether the job succeeded overall
    - Each acceptance criterion defined on the job

    Args:
        job: Completed job from the runner.

    Returns:
        List of MetricSamples ready to submit to Datadog.
    """
    samples: list[MetricSample] = []
    recipe_path = _extract_recipe_path(job)

    if not recipe_path:
        logger.debug("Skipping job %s: could not extract recipe path from command", job.name)
        return samples

    # New metrics use cleaned job tag (no "recipes." prefix)
    job_tag = recipe_path.replace("recipes.", "").replace(".", "_")
    # Legacy metrics use original format (with "recipes_" prefix) for backward compat
    legacy_job_tag = recipe_path.replace(".", "_")

    # Build base tags for additional context
    base_tags = {
        "job_name": job.name,
    }
    if job.is_remote:
        base_tags["execution_type"] = "remote"
        if job.remote_gpus:
            base_tags["gpus"] = str(job.remote_gpus)
        if job.remote_nodes:
            base_tags["nodes"] = str(job.remote_nodes)
    else:
        base_tags["execution_type"] = "local"

    # Determine overall success
    success = job.status == JobStatus.SUCCEEDED and job.acceptance_passed is not False

    # Emit runs_success criterion (always emitted for completed jobs)
    samples.extend(
        MetricSample.from_criterion(
            job=job_tag,
            category="training",
            criterion="runs_success",
            value=1.0 if success else 0.0,
            target=1.0,
            operator=">=",
            passed=success,
            base_tags=base_tags,
        )
    )

    # TODO(datadog-migration): Remove legacy metric after dashboard migration (2025-02-01)
    samples.append(
        MetricSample(
            name=f"metta.infra.cron.stable.{legacy_job_tag}.runs_success",
            value=1.0 if success else 0.0,
            tags=base_tags,
        )
    )

    # Emit each acceptance criterion
    if job.acceptance and job.status == JobStatus.SUCCEEDED and job.metrics:
        for criterion in job.acceptance:
            actual_value = job.metrics.get(criterion.metric)
            if actual_value is None:
                logger.debug(
                    "Skipping criterion %s for job %s: metric not found in job.metrics",
                    criterion.metric,
                    job.name,
                )
                continue

            criterion_passed = job.criterion_results.get(criterion.metric, False)
            criterion_name = _normalize_criterion_name(criterion.metric)

            # Build criterion-specific tags
            criterion_tags = dict(base_tags)

            # Handle tuple thresholds (for "in" operator)
            if isinstance(criterion.threshold, tuple):
                target_low, target_high = criterion.threshold
                # Use upper bound for target line overlay
                target_value = float(target_high)
                # Preserve range bounds in tags
                criterion_tags["target_low"] = str(target_low)
                criterion_tags["target_high"] = str(target_high)
            else:
                target_value = float(criterion.threshold)

            samples.extend(
                MetricSample.from_criterion(
                    job=job_tag,
                    category="training",
                    criterion=criterion_name,
                    value=float(actual_value),
                    target=target_value,
                    operator=criterion.operator,
                    passed=criterion_passed,
                    base_tags=criterion_tags,
                )
            )

            # TODO(datadog-migration): Remove legacy metric after dashboard migration (2025-02-01)
            samples.append(
                MetricSample(
                    name=f"metta.infra.cron.stable.{legacy_job_tag}.{criterion_name}",
                    value=float(actual_value),
                    tags={
                        **base_tags,
                        "criterion_passed": str(criterion_passed).lower(),
                        "criterion_threshold": str(criterion.threshold),
                        "criterion_operator": criterion.operator,
                    },
                )
            )

    return samples


def jobs_to_metrics(jobs: dict[str, Job]) -> list[MetricSample]:
    """Convert all job results to Datadog metrics.

    Args:
        jobs: Dictionary of job name to Job result.

    Returns:
        List of all MetricSamples from all jobs.
    """
    all_samples: list[MetricSample] = []
    for job in jobs.values():
        samples = job_to_metrics(job)
        all_samples.extend(samples)
    return all_samples

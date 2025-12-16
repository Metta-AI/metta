"""Convert stable runner Job results to Datadog MetricSamples."""

from __future__ import annotations

import logging

from devops.datadog.models import MetricSample
from devops.stable.runner import AcceptanceCriterion, Job, JobStatus

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


def job_to_metrics(job: Job) -> list[MetricSample]:
    """Convert a Job result to Datadog MetricSamples.

    Emits:
    - One "runs_success" metric per job (1.0 if succeeded and acceptance passed, 0.0 otherwise)
    - One metric per acceptance criterion (using criterion.metric_name if provided, or deriving from criterion.metric)

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

    base_tags = {
        "workflow_name": recipe_path,
        "job_name": job.name,
    }

    # Add remote/local tag
    if job.is_remote:
        base_tags["execution_type"] = "remote"
        if job.remote_gpus:
            base_tags["gpus"] = str(job.remote_gpus)
        if job.remote_nodes:
            base_tags["nodes"] = str(job.remote_nodes)
    else:
        base_tags["execution_type"] = "local"

    # Emit one "runs_success" metric per job
    success_value = 1.0 if job.status == JobStatus.SUCCEEDED and job.acceptance_passed is not False else 0.0
    samples.append(
        MetricSample(
            name=f"metta.infra.cron.stable.{recipe_path.replace('.', '_')}.runs_success",
            value=success_value,
            tags=base_tags,
        )
    )

    # Emit one metric per acceptance criterion
    if job.acceptance and job.status == JobStatus.SUCCEEDED and job.metrics:
        for criterion in job.acceptance:
            # Use metric_name if provided, otherwise derive from criterion.metric
            metric_name = criterion.metric_name
            if not metric_name:
                # Derive metric name from criterion.metric (e.g., "overview/sps" -> "overview_sps")
                metric_name = criterion.metric.replace("/", "_").replace(".", "_")

            # Get the actual value from job.metrics
            actual_value = job.metrics.get(criterion.metric)
            if actual_value is not None:
                # Use the evaluation result from the runner (don't recompute)
                criterion_passed = job.criterion_results.get(criterion.metric, False)
                samples.append(
                    MetricSample(
                        name=f"metta.infra.cron.stable.{recipe_path.replace('.', '_')}.{metric_name}",
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

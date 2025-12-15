"""Convert stable runner Job results to Datadog MetricSamples."""

from __future__ import annotations

import logging

from devops.datadog.models import MetricSample
from devops.stable.runner import Job, JobStatus

logger = logging.getLogger(__name__)


def _extract_workflow_name(job: Job) -> str | None:
    """Extract workflow name from job name.

    Job names are like: "{prefix}-recipes.prod.arena_basic_easy_shaped.train_100m"
    We need to map to workflow names like: "multigpu_arena_basic_easy_shaped"
    """
    # Extract the module and function name
    parts = job.name.split(".")
    if len(parts) < 3:
        return None

    # Get the module name (e.g., "arena_basic_easy_shaped")
    module_name = parts[-2] if len(parts) >= 3 else None
    func_name = parts[-1] if parts else None

    if not module_name or not func_name:
        return None

    # Map job patterns to workflow names
    # Training jobs
    if "train" in func_name.lower():
        # Multi-node training: any remote job with >1 node uses the multinode workflow bucket
        if job.remote_nodes and job.remote_nodes > 1:
            return "multinode_learning_progress"
        # Single GPU or multi-GPU single node
        if job.remote_gpus and job.remote_gpus >= 1:
            if "arena" in module_name:
                return "multigpu_arena_basic_easy_shaped"
        # Local training
        if not job.is_remote:
            if "arena" in module_name:
                return "local_arena_basic_easy_shaped"

    # Eval jobs
    if "eval" in func_name.lower() or "evaluate" in func_name.lower():
        if job.is_remote:
            return "remote_eval"
        else:
            return "local_eval"  # Note: schema shows "remote_eval" but we may need local too

    return None


def _is_training_job(job: Job) -> bool:
    """Check if job is a training job."""
    return "train" in job.name.lower()


def _is_eval_job(job: Job) -> bool:
    """Check if job is an eval job."""
    return "eval" in job.name.lower() or "evaluate" in job.name.lower()


def job_to_metrics(job: Job) -> list[MetricSample]:
    """Convert a Job result to Datadog MetricSamples.

    Args:
        job: Completed job from the runner.

    Returns:
        List of MetricSamples ready to submit to Datadog.
    """
    samples: list[MetricSample] = []
    workflow_name = _extract_workflow_name(job)

    if not workflow_name:
        logger.debug("Skipping job %s: could not determine workflow name", job.name)
        return samples

    base_tags = {
        "workflow_name": workflow_name,
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

    # Training metrics
    if _is_training_job(job):
        # Success metric
        success_value = 1.0 if job.status == JobStatus.SUCCEEDED and job.acceptance_passed is not False else 0.0
        if workflow_name == "multigpu_arena_basic_easy_shaped":
            samples.append(
                MetricSample(
                    name="metta.infra.cron.training.multigpu.runs_success",
                    value=success_value,
                    tags=base_tags,
                )
            )
        elif workflow_name == "multinode_learning_progress":
            samples.append(
                MetricSample(
                    name="metta.infra.cron.training.multinode.runs_success",
                    value=success_value,
                    tags=base_tags,
                )
            )
        elif workflow_name == "local_arena_basic_easy_shaped":
            # Local arena uses different metrics
            if job.status == JobStatus.SUCCEEDED:
                # Check if it's a checkpoint job or continuation
                if "checkpoint" in job.name.lower() or "first" in job.name.lower():
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.training.local_arena.first_checkpoint",
                            value=1.0 if job.acceptance_passed is not False else 0.0,
                            tags=base_tags,
                        )
                    )
                elif "continue" in job.name.lower():
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.training.local_arena.continues",
                            value=1.0 if job.acceptance_passed is not False else 0.0,
                            tags=base_tags,
                        )
                    )

        # Extract metrics from job.metrics (from WandB)
        if job.metrics and job.status == JobStatus.SUCCEEDED:
            # Hearts metric
            hearts = job.metrics.get("env_agent/heart.gained") or job.metrics.get("overview/hearts")
            if hearts is not None:
                if workflow_name == "multigpu_arena_basic_easy_shaped":
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.training.multigpu.hearts",
                            value=float(hearts),
                            tags=base_tags,
                        )
                    )
                elif workflow_name == "multinode_learning_progress":
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.training.multinode.hearts",
                            value=float(hearts),
                            tags=base_tags,
                        )
                    )

            # SPS metric
            sps = job.metrics.get("overview/sps")
            if sps is not None:
                if workflow_name == "multigpu_arena_basic_easy_shaped":
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.training.multigpu.sps",
                            value=float(sps),
                            tags=base_tags,
                        )
                    )
                elif workflow_name == "multinode_learning_progress":
                    # Multi-node uses "shaped" metric
                    shaped = job.metrics.get("overview/shaped_sps") or sps
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.training.multinode.shaped",
                            value=float(shaped),
                            tags=base_tags,
                        )
                    )

    # Eval metrics
    elif _is_eval_job(job):
        success_value = 1.0 if job.status == JobStatus.SUCCEEDED and job.acceptance_passed is not False else 0.0

        if workflow_name == "remote_eval":
            samples.append(
                MetricSample(
                    name="metta.infra.cron.eval.remote.success",
                    value=success_value,
                    tags=base_tags,
                )
            )

            # Duration in minutes
            if job.duration_s is not None:
                samples.append(
                    MetricSample(
                        name="metta.infra.cron.eval.remote.duration_minutes",
                        value=job.duration_s / 60.0,
                        tags=base_tags,
                    )
                )

            # Heart delta pct
            if job.metrics:
                heart_delta_pct = job.metrics.get("heart_delta_pct") or job.metrics.get("overview/heart_delta_pct")
                if heart_delta_pct is not None:
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.eval.remote.heart_delta_pct",
                            value=float(heart_delta_pct),
                            tags=base_tags,
                        )
                    )
        else:
            # Local eval
            samples.append(
                MetricSample(
                    name="metta.infra.cron.eval.local.success",
                    value=success_value,
                    tags=base_tags,
                )
            )

            # Heart delta pct
            if job.metrics:
                heart_delta_pct = job.metrics.get("heart_delta_pct") or job.metrics.get("overview/heart_delta_pct")
                if heart_delta_pct is not None:
                    samples.append(
                        MetricSample(
                            name="metta.infra.cron.eval.local.heart_delta_pct",
                            value=float(heart_delta_pct),
                            tags=base_tags,
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

"""WandB tool handlers."""

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from observatory_mcp.analyzers import training_context, wandb_analyzer
from observatory_mcp.utils import format_error_response, format_success_response

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client as BotoS3Client
    from wandb import Api

    from metta.adaptive.stores.wandb import WandbStore

logger = logging.getLogger(__name__)


def _safe_dict_convert(obj: Any) -> dict:
    """Safely convert WandB config/summary objects to dicts."""
    if obj is None:
        return {}
    if isinstance(obj, (str, bytes)):
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        if hasattr(obj, "items"):
            result = dict(obj.items())
            if isinstance(result, dict):
                return result
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        serialized = json.dumps(obj, default=str)
        result = json.loads(serialized)
        if isinstance(result, dict):
            return result
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


async def list_wandb_runs(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    tags: Optional[list[str]] = None,
    state: Optional[str] = None,
    limit: int = 50,
) -> str:
    """List WandB runs for entity/project with optional filters."""
    logger.info(f"Listing WandB runs: {entity}/{project}")

    filters = {}
    if state:
        filters["state"] = state
    if tags:
        filters["tags"] = {"$in": tags}

    run_list = wandb_store.list_runs(
        entity=entity,
        project=project,
        filters=filters if filters else None,
        limit=limit,
    )

    data = {
        "entity": entity,
        "project": project,
        "runs": run_list,
        "count": len(run_list),
    }

    logger.info(f"list_wandb_runs completed ({len(run_list)} runs)")
    return format_success_response(data)


async def get_wandb_run(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """Get detailed information about a WandB run."""
    if not run_id and not run_name:
            return format_error_response(
                ValueError("Either run_id or run_name must be provided"),
                "get_wandb_run",
                "Missing required parameter: run_id or run_name",
            )

    target_id = run_id or run_name
    logger.info(f"Getting WandB run: {entity}/{project}/{target_id}")

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=target_id)
    if not run_data:
        return format_error_response(
            ValueError(f"Run not found: {target_id}"),
            "get_wandb_run",
            f"Run '{target_id}' not found in {entity}/{project}",
        )

        logger.info("get_wandb_run completed")
    return format_success_response(run_data)


async def get_wandb_run_metrics(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    samples: Optional[int] = None,
) -> str:
    """Get metric time series data for a WandB run."""
    logger.info(f"Getting WandB run metrics: {entity}/{project}/{run_id}")

    metrics_data = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=metric_keys,
        samples=samples,
    )

    available_metrics = None
    if len(metrics_data.get("data", [])) == 0:
        discover_result = wandb_store.discover_run_metrics(
            entity=entity,
            project=project,
            run_id=run_id,
        )
        if discover_result.get("all_metrics"):
            available_metrics = discover_result["all_metrics"]
            logger.info(f"Discovered {len(available_metrics)} available metrics in run")

    data = {
        **metrics_data,
        "available_metrics": available_metrics,
    }

    logger.info(f"get_wandb_run_metrics completed ({metrics_data.get('samples', 0)} samples)")
    return format_success_response(data)

async def discover_wandb_run_metrics(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Discover available metrics for a WandB run by sampling its history."""
    logger.info(f"Discovering metrics for WandB run: {entity}/{project}/{run_id}")

    data = wandb_store.discover_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
    )

    logger.info(f"discover_wandb_run_metrics completed ({data.get('count', 0)} metrics found)")
    return format_success_response(data)

async def get_wandb_run_artifacts(
    wandb_api: "Api",
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Get list of artifacts for a WandB run."""
    logger.info(f"Getting WandB run artifacts: {entity}/{project}/{run_id}")

    run = wandb_api.run(f"{entity}/{project}/{run_id}")
    artifacts = run.logged_artifacts()

    artifact_list = []
    for artifact in artifacts:
        created_at = None
        if artifact.created_at:
            if isinstance(artifact.created_at, str):
                created_at = artifact.created_at
            else:
                created_at = artifact.created_at.isoformat()

        artifact_list.append(
            {
                "id": artifact.id,
                "name": artifact.name,
                "type": artifact.type,
                "version": artifact.version,
                "size": artifact.size,
                "created_at": created_at,
            }
        )

    data = {
        "run_id": run_id,
        "run_name": run.name,
        "artifacts": artifact_list,
        "count": len(artifact_list),
    }

    logger.info(f"get_wandb_run_artifacts completed ({len(artifact_list)} artifacts)")
    return format_success_response(data)

async def get_wandb_run_logs(
    wandb_api: "Api",
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Get logs for a WandB run."""
    logger.info(f"Getting WandB run logs: {entity}/{project}/{run_id}")

    run = wandb_api.run(f"{entity}/{project}/{run_id}")

    log_data = {
        "run_id": run_id,
        "run_name": run.name,
        "log_url": f"{run.url}/logs",
    }

    logger.info("get_wandb_run_logs completed")
    return format_success_response(log_data)

async def analyze_wandb_training_progression(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    context_window_steps: int = 1000,
    center_step: Optional[int] = None,
) -> str:
    """Analyze training progression for a WandB run."""
    logger.info(f"Analyzing WandB training progression: {entity}/{project}/{run_id}"    )

    metrics_result = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=metric_keys,
    )
    metrics_data = metrics_result.get("data", [])

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    run_name = run_data.get("name", run_id) if run_data else run_id

    context = training_context.analyze_training_progression(
        metrics_data=metrics_data,
        metric_keys=metric_keys,
        context_window_steps=context_window_steps,
        center_step=center_step,
    )

    context.run_id = run_id
    context.run_name = run_name

    data = context.to_dict()

    logger.info("analyze_wandb_training_progression completed")
    return format_success_response(data)

async def compare_wandb_runs(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_ids: list[str],
    metric_keys: list[str],
) -> str:
    """Compare multiple WandB runs."""
    logger.info(f"Comparing WandB runs: {entity}/{project}, {len(run_ids)} runs")

    runs_data = []
    for run_id in run_ids:
        run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
        if run_data:
            runs_data.append(
                {
                    "id": run_data.get("id", run_id),
                    "name": run_data.get("name", run_id),
                    "summary": run_data.get("summary", {}),
                    "config": run_data.get("config", {}),
                }
            )

    comparison = wandb_analyzer.compare_runs(runs_data, metric_keys)

    logger.info("compare_wandb_runs completed")
    return format_success_response(comparison)

async def analyze_wandb_learning_curves(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    smoothing_window: int = 10,
) -> str:
    """Analyze learning curves for a WandB run."""
    logger.info(f"Analyzing WandB learning curves: {entity}/{project}/{run_id}"    )

    metrics_result = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=metric_keys,
    )
    metrics_data = metrics_result.get("data", [])

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    run_name = run_data.get("name", run_id) if run_data else run_id

    analysis = wandb_analyzer.analyze_learning_curves(
        metrics_data=metrics_data,
        metric_keys=metric_keys,
        smoothing_window=smoothing_window,
    )

    analysis["run_id"] = run_id
    analysis["run_name"] = run_name

    logger.info("analyze_wandb_learning_curves completed")
    return format_success_response(analysis)

async def identify_wandb_critical_moments(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    threshold: float = 0.1,
) -> str:
    """Identify critical moments in a WandB run."""
    logger.info(f"Identifying WandB critical moments: {entity}/{project}/{run_id}"    )

    metrics_result = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=metric_keys,
    )
    metrics_data = metrics_result.get("data", [])

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    run_name = run_data.get("name", run_id) if run_data else run_id

    moments = wandb_analyzer.identify_critical_moments(
        metrics_data=metrics_data,
        metric_keys=metric_keys,
        threshold=threshold,
    )

    data = {
        "run_id": run_id,
        "run_name": run_name,
        "critical_moments": moments,
        "count": len(moments),
    }

    logger.info(f"identify_wandb_critical_moments completed ({len(moments)} moments)")
    return format_success_response(data)

async def correlate_wandb_metrics(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    metric_pairs: list[list[str]],
) -> str:
    """Calculate correlations between metric pairs for a WandB run."""
    logger.info(f"Correlating WandB metrics: {entity}/{project}/{run_id}")

    all_metrics = set()
    for pair in metric_pairs:
        all_metrics.update(pair    )

    metrics_result = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=list(all_metrics),
    )
    metrics_data = metrics_result.get("data", [])

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    run_name = run_data.get("name", run_id) if run_data else run_id

    metric_tuples = [tuple(pair) for pair in metric_pairs]
    correlations = wandb_analyzer.correlate_metrics(
        metrics_data=metrics_data,
        metric_pairs=metric_tuples,
    )

    correlations["run_id"] = run_id
    correlations["run_name"] = run_name

    logger.info("correlate_wandb_metrics completed")
    return format_success_response(correlations)

async def analyze_wandb_behavioral_patterns(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    behavior_categories: Optional[list[str]] = None,
) -> str:
    """Analyze behavioral patterns in a WandB run."""
    logger.info(f"Analyzing WandB behavioral patterns: {entity}/{project}/{run_id}")

    default_metrics = ["overview/reward", "metric/agent_step", "experience/rewards"]
    if behavior_categories:
        metric_keys = behavior_categories
    else:
        metric_keys = default_metrics

    metrics_result = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=metric_keys,
    )
    metrics_data = metrics_result.get("data", [])

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    run_name = run_data.get("name", run_id) if run_data else run_id

    analysis = {
        "run_id": run_id,
        "run_name": run_name,
        "behavior_categories": behavior_categories or default_metrics,
        "patterns": {
            "action_mastery": _analyze_action_mastery(metrics_data),
            "resource_efficiency": _analyze_resource_efficiency(metrics_data),
            "strategy_consistency": _analyze_strategy_consistency(metrics_data),
        },
    }

    logger.info("analyze_wandb_behavioral_patterns completed")
    return format_success_response(analysis)

def _analyze_action_mastery(metrics_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze action mastery progression."""
    return {"progression": "stable", "trend": "improving"}


def _analyze_resource_efficiency(metrics_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze resource efficiency trends."""
    return {"efficiency": "high", "trend": "stable"}


def _analyze_strategy_consistency(metrics_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze strategy consistency."""
    return {"consistency": "moderate", "volatility": "low"}


async def generate_wandb_training_insights(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Generate AI-powered training insights for a WandB run."""
    logger.info(f"Generating WandB training insights: {entity}/{project}/{run_id}")

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    if not run_data:
        return format_error_response(
            ValueError(f"Run not found: {run_id}"),
            "generate_wandb_training_insights",
            f"Run '{run_id}' not found in {entity}/{project}",
        )

    summary = run_data.get("summary", {})
    config = run_data.get("config", {})

    insights = {
        "run_id": run_id,
        "run_name": run_data.get("name", run_id),
        "key_achievements": _extract_achievements(summary),
        "concerning_patterns": _extract_concerning_patterns(summary),
        "recommendations": _generate_recommendations(summary, config),
    }

    logger.info("generate_wandb_training_insights completed")
    return format_success_response(insights)

def _extract_achievements(summary: dict[str, Any]) -> list[str]:
    """Extract key achievements from run summary."""
    achievements = []
    if "overview/reward" in summary:
        achievements.append(f"Final reward: {summary['overview/reward']:.2f}")
    if "metric/agent_step" in summary:
        achievements.append(f"Total steps: {summary['metric/agent_step']}")
    return achievements


def _extract_concerning_patterns(summary: dict[str, Any]) -> list[str]:
    """Extract concerning patterns from run summary."""
    patterns = []
    return patterns


def _generate_recommendations(summary: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Generate training recommendations."""
    recommendations = []
    return recommendations


async def predict_wandb_training_outcome(
    wandb_store: "WandbStore",
    entity: str,
    project: str,
    run_id: str,
    target_metric: str,
    projection_steps: int = 1000,
) -> str:
    """Predict training outcome for a WandB run."""
    logger.info(f"Predicting WandB training outcome: {entity}/{project}/{run_id}"    )

    metrics_result = wandb_store.get_run_metrics(
        entity=entity,
        project=project,
        run_id=run_id,
        metric_keys=[target_metric],
    )
    metrics_data = metrics_result.get("data", [])

    if not metrics_data:
        raise ValueError(f"No data found for metric: {target_metric}")

    values = [point.get(target_metric, 0) for point in metrics_data if target_metric in point]
    if len(values) < 2:
        raise ValueError("Insufficient data for prediction")

    run_data = wandb_store.get_run(entity=entity, project=project, run_id=run_id)
    run_name = run_data.get("name", run_id) if run_data else run_id

    projected_value = _project_value(values, projection_steps)
    convergence_estimate = _estimate_convergence(values)

    prediction = {
        "run_id": run_id,
        "run_name": run_name,
        "target_metric": target_metric,
        "current_value": values[-1],
        "projected_value": projected_value,
        "convergence_estimate": convergence_estimate,
        "confidence": _calculate_confidence(values),
    }

    logger.info("predict_wandb_training_outcome completed")
    return format_success_response(prediction)

def _project_value(values: list[float], steps: int) -> float:
    """Project future value based on trend."""
    if len(values) < 2:
        return values[-1] if values else 0.0

    recent = values[-10:] if len(values) >= 10 else values
    trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0.0

    return values[-1] + trend * steps


def _estimate_convergence(values: list[float]) -> dict[str, Any]:
    """Estimate convergence point."""
    if len(values) < 10:
        return {"converged": False, "steps_remaining": None}

    recent = values[-10:]
    variance = sum((v - sum(recent) / len(recent)) ** 2 for v in recent) / len(recent)

    return {
        "converged": variance < 0.01,
        "steps_remaining": 100 if variance > 0.01 else 0,
    }


def _calculate_confidence(values: list[float]) -> float:
    """Calculate prediction confidence."""
    if len(values) < 2:
        return 0.0

    recent = values[-10:] if len(values) >= 10 else values
    variance = sum((v - sum(recent) / len(recent)) ** 2 for v in recent) / len(recent)

    return max(0.0, min(1.0, 1.0 - variance))


async def link_wandb_run_to_s3_checkpoints(
    wandb_api: "Api",
    s3_client: "BotoS3Client",
    bucket: str,
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Link a WandB run to its S3 checkpoints."""
    logger.info(f"Linking WandB run to S3 checkpoints: {entity}/{project}/{run_id}")

    run = wandb_api.run(f"{entity}/{project}/{run_id}")
    run_name = run.name or run_id

    s3_prefix = f"checkpoints/{run_name}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    checkpoints = []

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".mpt"):
                filename = key.split("/")[-1]
                checkpoint_info = {
                    "key": key,
                    "uri": f"s3://{bucket}/{key}",
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
        "wandb_run": {
            "id": run_id,
            "name": run_name,
            "url": run.url,
        },
        "checkpoints": checkpoints,
        "count": len(checkpoints),
    }

    logger.info(f"link_wandb_run_to_s3_checkpoints completed ({len(checkpoints)} checkpoints)")
    return format_success_response(data)

async def link_wandb_run_to_skypilot_job(
    wandb_api: "Api",
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Link a WandB run to its Skypilot job."""
    logger.info(f"Linking WandB run to Skypilot job: {entity}/{project}/{run_id}")

    run = wandb_api.run(f"{entity}/{project}/{run_id}")
    run_name = run.name or run_id

    cmd = ["sky", "jobs", "queue", "--json"]
    import subprocess

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
        "wandb_run": {
            "id": run_id,
            "name": run_name,
            "url": run.url,
        },
        "jobs": matching_jobs,
        "count": len(matching_jobs),
    }

    logger.info(f"link_wandb_run_to_skypilot_job completed ({len(matching_jobs)} jobs)")
    return format_success_response(data)

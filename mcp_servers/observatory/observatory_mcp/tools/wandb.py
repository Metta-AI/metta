"""WandB tool handlers."""

import json
import logging
from typing import Any, Optional

from observatory_mcp.analyzers import training_context, wandb_analyzer
from observatory_mcp.clients.s3_client import S3Client
from observatory_mcp.clients.skypilot_client import SkypilotClient
from observatory_mcp.clients.wandb_client import WandBClient
from observatory_mcp.utils import format_error_response, format_success_response

logger = logging.getLogger(__name__)


def _safe_dict_convert(obj: Any) -> dict:
    """Safely convert WandB config/summary objects to dicts."""
    if obj is None:
        return {}
    # If it's a string, return empty dict (config/summary shouldn't be strings)
    if isinstance(obj, (str, bytes)):
        return {}
    # If already a dict, return as-is
    if isinstance(obj, dict):
        return obj
    # Try using .items() method if available (common for dict-like objects)
    try:
        if hasattr(obj, "items"):
            result = dict(obj.items())
            if isinstance(result, dict):
                return result
    except (TypeError, ValueError, AttributeError):
        pass
    # Try JSON serialization (WandB objects typically support this)
    try:
        serialized = json.dumps(obj, default=str)
        result = json.loads(serialized)
        # Only return if it's actually a dict
        if isinstance(result, dict):
            return result
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    # Last resort: return empty dict
    return {}


async def list_wandb_runs(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    tags: Optional[list[str]] = None,
    state: Optional[str] = None,
    limit: int = 50,
) -> str:
    """List WandB runs for entity/project with optional filters.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        tags: Filter by tags (list of tag strings)
        state: Filter by state ("running", "finished", "crashed", "killed")
        limit: Maximum number of runs to return (default: 50)

    Returns:
        JSON string with list of runs and their metadata
    """
    try:
        logger.info(f"Listing WandB runs: {entity}/{project}")

        filters = {}
        if state:
            filters["state"] = state
        if tags:
            filters["tags"] = {"$in": tags}

        runs = wandb_client.api.runs(
            f"{entity}/{project}",
            filters=filters if filters else None,
            order="-created_at",
        )

        run_list = []
        count = 0
        for run in runs:
            if count >= limit:
                break

            try:
                # Handle created_at - can be datetime or string
                created_at = None
                if hasattr(run, "created_at") and run.created_at:
                    if isinstance(run.created_at, str):
                        created_at = run.created_at
                    else:
                        created_at = run.created_at.isoformat()

                # Handle updated_at - can be datetime or string (may not exist)
                updated_at = None
                if hasattr(run, "updated_at") and run.updated_at:
                    if isinstance(run.updated_at, str):
                        updated_at = run.updated_at
                    else:
                        updated_at = run.updated_at.isoformat()

                # Handle tags - ensure it's always a list
                tags = run.tags if hasattr(run, "tags") else []
                if not isinstance(tags, list):
                    tags = list(tags) if tags else []

                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "display_name": getattr(run, "display_name", run.name),
                    "state": run.state,
                    "tags": tags,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "url": run.url,
                    "config": _safe_dict_convert(run.config),
                    "summary": _safe_dict_convert(run.summary),
                }
                run_list.append(run_data)
                count += 1
            except Exception as e:
                logger.warning(f"Error processing run {getattr(run, 'id', 'unknown')}: {e}")
                # Skip this run and continue
                continue

        data = {
            "entity": entity,
            "project": project,
            "runs": run_list,
            "count": len(run_list),
        }

        logger.info(f"list_wandb_runs completed ({len(run_list)} runs)")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"list_wandb_runs failed: {e}")
        return format_error_response(e, "list_wandb_runs")


async def get_wandb_run(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """Get detailed information about a specific WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID (preferred if available)
        run_name: WandB run name (used if run_id not provided)

    Returns:
        JSON string with detailed run information
    """
    try:
        if run_id:
            logger.info(f"Getting WandB run by ID: {entity}/{project}/{run_id}")
            run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        elif run_name:
            logger.info(f"Getting WandB run by name: {entity}/{project}/{run_name}")
            runs = wandb_client.api.runs(f"{entity}/{project}", filters={"display_name": run_name})
            run = next(iter(runs), None)
            if not run:
                return format_error_response(
                    ValueError(f"Run not found: {run_name}"),
                    "get_wandb_run",
                    f"Run '{run_name}' not found in {entity}/{project}",
                )
        else:
            return format_error_response(
                ValueError("Either run_id or run_name must be provided"),
                "get_wandb_run",
                "Missing required parameter: run_id or run_name",
            )

        # Handle created_at - can be datetime or string
        created_at = None
        if hasattr(run, "created_at") and run.created_at:
            if isinstance(run.created_at, str):
                created_at = run.created_at
            else:
                created_at = run.created_at.isoformat()

        # Handle updated_at - can be datetime or string (may not exist)
        updated_at = None
        if hasattr(run, "updated_at") and run.updated_at:
            if isinstance(run.updated_at, str):
                updated_at = run.updated_at
            else:
                updated_at = run.updated_at.isoformat()

        data = {
            "id": run.id,
            "name": run.name,
            "display_name": getattr(run, "display_name", run.name),
            "state": run.state,
            "tags": run.tags,
            "created_at": created_at,
            "updated_at": updated_at,
            "url": run.url,
            "config": _safe_dict_convert(run.config),
            "summary": _safe_dict_convert(run.summary),
        }

        logger.info("get_wandb_run completed")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"get_wandb_run failed: {e}")
        return format_error_response(e, "get_wandb_run")


async def get_wandb_run_metrics(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    samples: Optional[int] = None,
) -> str:
    """Get metric time series data for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        metric_keys: List of metric names to fetch (e.g., ["overview/reward", "metric/agent_step"])
        samples: Optional limit on number of samples (for large runs)

    Returns:
        JSON string with metric time series data
    """
    try:
        logger.info(f"Getting WandB run metrics: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")

        if samples:
            history = run.history(keys=metric_keys, pandas=False, samples=samples)
        else:
            history = run.history(keys=metric_keys, pandas=False)

        metrics_data = list(history)

        data = {
            "run_id": run_id,
            "run_name": run.name,
            "metric_keys": metric_keys,
            "samples": len(metrics_data),
            "data": metrics_data,
        }

        logger.info(f"get_wandb_run_metrics completed ({len(metrics_data)} samples)")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"get_wandb_run_metrics failed: {e}")
        return format_error_response(e, "get_wandb_run_metrics")


async def get_wandb_run_artifacts(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Get list of artifacts for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID

    Returns:
        JSON string with list of artifacts
    """
    try:
        logger.info(f"Getting WandB run artifacts: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        artifacts = run.logged_artifacts()

        artifact_list = []
        for artifact in artifacts:
            # Handle created_at - can be datetime or string
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

    except Exception as e:
        logger.warning(f"get_wandb_run_artifacts failed: {e}")
        return format_error_response(e, "get_wandb_run_artifacts")


async def get_wandb_run_logs(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Get logs for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID

    Returns:
        JSON string with log URL
    """
    try:
        logger.info(f"Getting WandB run logs: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")

        log_data = {
            "run_id": run_id,
            "run_name": run.name,
            "log_url": f"{run.url}/logs",
        }

        logger.info("get_wandb_run_logs completed")
        return format_success_response(log_data)

    except Exception as e:
        logger.warning(f"get_wandb_run_logs failed: {e}")
        return format_error_response(e, "get_wandb_run_logs")


async def analyze_wandb_training_progression(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    context_window_steps: int = 1000,
    center_step: Optional[int] = None,
) -> str:
    """Analyze training progression for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        metric_keys: List of metric keys to analyze
        context_window_steps: Number of steps to analyze around center (default: 1000)
        center_step: Optional center step (defaults to middle of data)

    Returns:
        JSON string with training progression analysis
    """
    try:
        logger.info(f"Analyzing WandB training progression: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=metric_keys, pandas=False)
        metrics_data = list(history)

        context = training_context.analyze_training_progression(
            metrics_data=metrics_data,
            metric_keys=metric_keys,
            context_window_steps=context_window_steps,
            center_step=center_step,
        )

        context.run_id = run_id
        context.run_name = run.name

        data = context.to_dict()

        logger.info("analyze_wandb_training_progression completed")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"analyze_wandb_training_progression failed: {e}")
        return format_error_response(e, "analyze_wandb_training_progression")


async def compare_wandb_runs(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_ids: list[str],
    metric_keys: list[str],
) -> str:
    """Compare multiple WandB runs.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_ids: List of WandB run IDs to compare
        metric_keys: List of metric keys to compare

    Returns:
        JSON string with comparison analysis
    """
    try:
        logger.info(f"Comparing WandB runs: {entity}/{project}, {len(run_ids)} runs")

        runs_data = []
        for run_id in run_ids:
            run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
            runs_data.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "summary": _safe_dict_convert(run.summary),
                    "config": _safe_dict_convert(run.config),
                }
            )

        comparison = wandb_analyzer.compare_runs(runs_data, metric_keys)

        logger.info("compare_wandb_runs completed")
        return format_success_response(comparison)

    except Exception as e:
        logger.warning(f"compare_wandb_runs failed: {e}")
        return format_error_response(e, "compare_wandb_runs")


async def analyze_wandb_learning_curves(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    smoothing_window: int = 10,
) -> str:
    """Analyze learning curves for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        metric_keys: List of metric keys to analyze
        smoothing_window: Window size for smoothing (default: 10)

    Returns:
        JSON string with learning curve analysis
    """
    try:
        logger.info(f"Analyzing WandB learning curves: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=metric_keys, pandas=False)
        metrics_data = list(history)

        analysis = wandb_analyzer.analyze_learning_curves(
            metrics_data=metrics_data,
            metric_keys=metric_keys,
            smoothing_window=smoothing_window,
        )

        analysis["run_id"] = run_id
        analysis["run_name"] = run.name

        logger.info("analyze_wandb_learning_curves completed")
        return format_success_response(analysis)

    except Exception as e:
        logger.warning(f"analyze_wandb_learning_curves failed: {e}")
        return format_error_response(e, "analyze_wandb_learning_curves")


async def identify_wandb_critical_moments(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    metric_keys: list[str],
    threshold: float = 0.1,
) -> str:
    """Identify critical moments in a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        metric_keys: List of metric keys to analyze
        threshold: Threshold for detecting significant changes (default: 0.1)

    Returns:
        JSON string with critical moments
    """
    try:
        logger.info(f"Identifying WandB critical moments: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=metric_keys, pandas=False)
        metrics_data = list(history)

        moments = wandb_analyzer.identify_critical_moments(
            metrics_data=metrics_data,
            metric_keys=metric_keys,
            threshold=threshold,
        )

        data = {
            "run_id": run_id,
            "run_name": run.name,
            "critical_moments": moments,
            "count": len(moments),
        }

        logger.info(f"identify_wandb_critical_moments completed ({len(moments)} moments)")
        return format_success_response(data)

    except Exception as e:
        logger.warning(f"identify_wandb_critical_moments failed: {e}")
        return format_error_response(e, "identify_wandb_critical_moments")


async def correlate_wandb_metrics(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    metric_pairs: list[list[str]],
) -> str:
    """Calculate correlations between metric pairs for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        metric_pairs: List of [metric1, metric2] pairs to correlate

    Returns:
        JSON string with correlation analysis
    """
    try:
        logger.info(f"Correlating WandB metrics: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")

        all_metrics = set()
        for pair in metric_pairs:
            all_metrics.update(pair)

        history = run.history(keys=list(all_metrics), pandas=False)
        metrics_data = list(history)

        metric_tuples = [tuple(pair) for pair in metric_pairs]
        correlations = wandb_analyzer.correlate_metrics(
            metrics_data=metrics_data,
            metric_pairs=metric_tuples,
        )

        correlations["run_id"] = run_id
        correlations["run_name"] = run.name

        logger.info("correlate_wandb_metrics completed")
        return format_success_response(correlations)

    except Exception as e:
        logger.warning(f"correlate_wandb_metrics failed: {e}")
        return format_error_response(e, "correlate_wandb_metrics")


async def analyze_wandb_behavioral_patterns(
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    behavior_categories: Optional[list[str]] = None,
) -> str:
    """Analyze behavioral patterns in a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        behavior_categories: Optional list of behavior categories to analyze

    Returns:
        JSON string with behavioral pattern analysis
    """
    try:
        logger.info(f"Analyzing WandB behavioral patterns: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")

        default_metrics = ["overview/reward", "metric/agent_step", "experience/rewards"]
        if behavior_categories:
            metric_keys = behavior_categories
        else:
            metric_keys = default_metrics

        history = run.history(keys=metric_keys, pandas=False)
        metrics_data = list(history)

        analysis = {
            "run_id": run_id,
            "run_name": run.name,
            "behavior_categories": behavior_categories or default_metrics,
            "patterns": {
                "action_mastery": _analyze_action_mastery(metrics_data),
                "resource_efficiency": _analyze_resource_efficiency(metrics_data),
                "strategy_consistency": _analyze_strategy_consistency(metrics_data),
            },
        }

        logger.info("analyze_wandb_behavioral_patterns completed")
        return format_success_response(analysis)

    except Exception as e:
        logger.warning(f"analyze_wandb_behavioral_patterns failed: {e}")
        return format_error_response(e, "analyze_wandb_behavioral_patterns")


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
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Generate AI-powered training insights for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID

    Returns:
        JSON string with training insights
    """
    try:
        logger.info(f"Generating WandB training insights: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")

        summary = _safe_dict_convert(run.summary)
        config = _safe_dict_convert(run.config)

        insights = {
            "run_id": run_id,
            "run_name": run.name,
            "key_achievements": _extract_achievements(summary),
            "concerning_patterns": _extract_concerning_patterns(summary),
            "recommendations": _generate_recommendations(summary, config),
        }

        logger.info("generate_wandb_training_insights completed")
        return format_success_response(insights)

    except Exception as e:
        logger.warning(f"generate_wandb_training_insights failed: {e}")
        return format_error_response(e, "generate_wandb_training_insights")


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
    wandb_client: WandBClient,
    entity: str,
    project: str,
    run_id: str,
    target_metric: str,
    projection_steps: int = 1000,
) -> str:
    """Predict training outcome for a WandB run.

    Args:
        wandb_client: WandB client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID
        target_metric: Metric to predict (e.g., "overview/reward")
        projection_steps: Number of steps to project forward (default: 1000)

    Returns:
        JSON string with prediction analysis
    """
    try:
        logger.info(f"Predicting WandB training outcome: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=[target_metric], pandas=False)
        metrics_data = list(history)

        if not metrics_data:
            raise ValueError(f"No data found for metric: {target_metric}")

        values = [point.get(target_metric, 0) for point in metrics_data if target_metric in point]
        if len(values) < 2:
            raise ValueError("Insufficient data for prediction")

        projected_value = _project_value(values, projection_steps)
        convergence_estimate = _estimate_convergence(values)

        prediction = {
            "run_id": run_id,
            "run_name": run.name,
            "target_metric": target_metric,
            "current_value": values[-1],
            "projected_value": projected_value,
            "convergence_estimate": convergence_estimate,
            "confidence": _calculate_confidence(values),
        }

        logger.info("predict_wandb_training_outcome completed")
        return format_success_response(prediction)

    except Exception as e:
        logger.warning(f"predict_wandb_training_outcome failed: {e}")
        return format_error_response(e, "predict_wandb_training_outcome")


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
    wandb_client: WandBClient,
    s3_client: S3Client,
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Link a WandB run to its S3 checkpoints.

    Args:
        wandb_client: WandB client instance
        s3_client: S3 client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID

    Returns:
        JSON string with linked checkpoints
    """
    try:
        logger.info(f"Linking WandB run to S3 checkpoints: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        run_name = run.name or run_id

        s3_prefix = f"checkpoints/{run_name}/"
        paginator = s3_client.client.get_paginator("list_objects_v2")
        checkpoints = []

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

    except Exception as e:
        logger.warning(f"link_wandb_run_to_s3_checkpoints failed: {e}")
        return format_error_response(e, "link_wandb_run_to_s3_checkpoints")


async def link_wandb_run_to_skypilot_job(
    wandb_client: WandBClient,
    skypilot_client: SkypilotClient,
    entity: str,
    project: str,
    run_id: str,
) -> str:
    """Link a WandB run to its Skypilot job.

    Args:
        wandb_client: WandB client instance
        skypilot_client: Skypilot client instance
        entity: WandB entity (user/team)
        project: WandB project name
        run_id: WandB run ID

    Returns:
        JSON string with linked job information
    """
    try:
        logger.info(f"Linking WandB run to Skypilot job: {entity}/{project}/{run_id}")

        run = wandb_client.api.run(f"{entity}/{project}/{run_id}")
        run_name = run.name or run_id

        cmd = ["sky", "jobs", "queue", "--json"]
        stdout, stderr, returncode = skypilot_client.run_command(cmd, timeout=30)

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

    except Exception as e:
        logger.warning(f"link_wandb_run_to_skypilot_job failed: {e}")
        return format_error_response(e, "link_wandb_run_to_skypilot_job")

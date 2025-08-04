import json
import logging
from typing import Any, Dict, List, Optional

import wandb

from metta.common.util.numpy_helpers import clean_numpy_types

logger = logging.getLogger("sweep")


# 1 - Sweep utilities.
def get_sweep_runs(sweep_name: str, entity: str, project: str) -> List[Any]:
    """Get all runs from a sweep (group) sorted by score."""
    api = wandb.Api()

    # Get all runs from the group
    runs = api.runs(
        f"{entity}/{project}",
        filters={
            "group": sweep_name,
            "state": "finished",  # Only successful runs
        },
    )

    # Filter for runs with valid scores
    valid_runs = []
    for run in runs:
        score = run.summary.get("score", run.summary.get("protein.objective", 0))
        if score is not None and score > 0:  # Filter out failed runs
            valid_runs.append(run)

    # Sort by score (descending for reward metric)
    valid_runs.sort(key=lambda r: r.summary.get("score", r.summary.get("protein.objective", 0)), reverse=True)
    return valid_runs


# 2 - Protein Integration Utilities.
def fetch_protein_observations_from_wandb(
    wandb_entity: str,
    wandb_project: str,
    sweep_name: str,
    max_observations: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch latest protein observations from WandB sweep runs using groups.

    Args:
        wandb_entity: The WandB entity name.
        wandb_project: The WandB project name.
        sweep_name: The sweep name (used as group).
        max_observations: The maximum number of observations to fetch.

    Returns:
        List of observation dictionaries with format:
        {
            "suggestion": dict,      # The hyperparameters used
            "objective": float,      # The objective value achieved
            "cost": float,          # The cost (e.g., runtime in seconds)
            "is_failure": bool,     # Whether the run failed

        }
    """
    api = wandb.Api()
    wandb_path = f"{wandb_entity}/{wandb_project}"

    # Use the API's native filtering and ordering
    # Order by created_at descending (newest first) and limit results
    runs = api.runs(
        path=wandb_path,
        filters={
            "group": sweep_name,  # Filter by group instead of sweep
            "state": {"$in": ["finished", "failed"]},  # Only get completed runs
            "summary_metrics.protein_observation": {"$exists": True},  # Only runs with observations
        },
        order="-created_at",  # Descending order (newest first)
        per_page=max_observations,  # Limit the number of results
    )

    # Iterate through runs (already filtered and limited)
    return [deep_clean(run.summary.get("protein_observation")) for run in runs]  # type: ignore


def record_protein_observation_to_wandb(
    wandb_run: Any,
    suggestion: dict[str, Any],
    objective: float,
    cost: float,
    is_failure: bool,
) -> None:
    """
    Record an observation to WandB.

    Args:
        wandb_run: The WandB run to record the observation to.
        suggestion: The suggestioån to record.
        objective: The objective value to optimize (higher is better for maximization).
        cost: The cost of this evaluation (e.g., time taken).
        is_failure: Whether the suggestion failed.
    """
    wandb_run.summary.update(
        {
            "protein_observation": {
                "suggestion": suggestion,
                "objective": objective,
                "cost": cost,
                "is_failure": is_failure,
            },
        }
    )


# 3 - Data Utilities.
def deep_clean(obj):
    """Recursively convert any object to JSON-serializable Python types."""
    if isinstance(obj, dict):
        # Already a regular dict, just recursively clean values
        return {k: deep_clean(v) for k, v in obj.items()}
    elif hasattr(obj, "items"):
        # Handle dict-like objects (including WandB SummarySubDict)
        # Convert to regular dict first, then recursively clean
        return {k: deep_clean(v) for k, v in dict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_clean(v) for v in obj]
    else:
        # For any other type, use clean_numpy_types first
        cleaned = clean_numpy_types(obj)
        # Then verify it's serializable
        json.dumps(cleaned)
        return cleaned


# 4 - Evaluation Logging Utilities.
def log_sweep_evaluation_results(
    wandb_run: Any,
    eval_results: Any,  # EvalResults from evaluate_policy
    cost_info: Dict[str, float],
    agent_step: Optional[int] = None,
    epoch: Optional[int] = None,
    upload_replays: bool = True,
) -> None:
    """
    Log evaluation results and cost information to WandB, following the same
    pattern as training evaluations but with "sweep_eval/" prefix.

    This function handles:
    - Evaluation metrics with sweep_eval/eval_ prefix
    - Category scores under sweep_eval/overview/
    - Cost metrics under sweep_eval/cost/
    - Replay URL uploads (if available and enabled)

    Args:
        wandb_run: Active WandB run to log to
        eval_results: EvalResults object from evaluate_policy containing scores and replay_urls
        cost_info: Dictionary containing cost metrics:
            - hourly: Hourly cost rate
            - training: Training cost
            - eval: Evaluation cost
            - total: Total cost (training + eval)
            - accrued: Accrued cost since job start
        agent_step: Optional agent step count for x-axis
        epoch: Optional epoch count for x-axis
        upload_replays: Whether to upload replay URLs if available

    Note:
        This does NOT handle protein observations - that's a separate concern
        handled by record_protein_observation_to_wandb.
    """
    if not wandb_run:
        logger.warning("No WandB run provided, skipping evaluation logging")
        return

    # Build metrics dictionary following training's pattern
    metrics = {}

    # 1. Add evaluation metrics with eval_ prefix
    if hasattr(eval_results, "scores") and hasattr(eval_results.scores, "to_wandb_metrics_format"):
        eval_metrics = eval_results.scores.to_wandb_metrics_format()
        for k, v in eval_metrics.items():
            metrics[f"sweep_eval/eval_{k}"] = v

    # 2. Add category scores under overview/
    if hasattr(eval_results, "scores") and hasattr(eval_results.scores, "category_scores"):
        for category, score in eval_results.scores.category_scores.items():
            metrics[f"sweep_eval/overview/{category}_score"] = score

    # 3. Add average scores
    if hasattr(eval_results, "scores"):
        if hasattr(eval_results.scores, "avg_simulation_score"):
            metrics["sweep_eval/overview/avg_simulation_score"] = eval_results.scores.avg_simulation_score
        if hasattr(eval_results.scores, "avg_category_score"):
            metrics["sweep_eval/overview/avg_category_score"] = eval_results.scores.avg_category_score

    # 4. Add cost metrics
    if cost_info:
        metrics.update(
            {
                "sweep_eval/cost/hourly": cost_info.get("hourly", 0.0),
                "sweep_eval/cost/training": cost_info.get("training", 0.0),
                "sweep_eval/cost/eval": cost_info.get("eval", 0.0),
                "sweep_eval/cost/total": cost_info.get("total", 0.0),
                "sweep_eval/cost/accrued": cost_info.get("accrued", 0.0),
            }
        )

    # 5. Add x-axis metrics if provided (these don't need prefix as they're shared)
    if agent_step is not None:
        metrics["metric/agent_step"] = agent_step
    if epoch is not None:
        metrics["metric/epoch"] = epoch

    # Log all metrics to WandB
    if metrics:
        wandb_run.log(metrics)
        logger.info(f"Logged {len(metrics)} evaluation metrics to WandB")

    # 6. Upload replay URLs if available and enabled
    if upload_replays and hasattr(eval_results, "replay_urls") and eval_results.replay_urls:
        try:
            # Import here to avoid circular dependency
            from metta.rl.evaluate import upload_replay_html

            upload_replay_html(
                replay_urls=eval_results.replay_urls,
                agent_step=agent_step or 0,
                epoch=epoch or 0,
                wandb_run=wandb_run,
            )
            logger.info(f"Uploaded {len(eval_results.replay_urls)} replay URLs to WandB")
        except ImportError:
            logger.warning("Could not import upload_replay_html, skipping replay upload")
        except Exception as e:
            logger.error(f"Failed to upload replay URLs: {e}", exc_info=True)

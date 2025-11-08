"""S3 checkpoint analysis functions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def analyze_checkpoint_progression(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze checkpoint progression over time.

    Args:
        checkpoints: List of checkpoint dictionaries with epoch and metadata

    Returns:
        Progression analysis dictionary
    """
    if not checkpoints:
        return {"progression": [], "trends": {}}

    checkpoints_with_epoch = [c for c in checkpoints if "epoch" in c]
    if not checkpoints_with_epoch:
        return {"progression": [], "trends": {"error": "No checkpoints with epoch information"}}

    checkpoints_with_epoch.sort(key=lambda x: x.get("epoch", 0))

    epochs = [c.get("epoch", 0) for c in checkpoints_with_epoch]
    sizes = [c.get("size", 0) for c in checkpoints_with_epoch]

    progression = []
    for i, checkpoint in enumerate(checkpoints_with_epoch):
        progression.append({
            "epoch": checkpoint.get("epoch", 0),
            "size": checkpoint.get("size", 0),
            "size_delta": sizes[i] - sizes[i - 1] if i > 0 else 0,
            "epoch_delta": epochs[i] - epochs[i - 1] if i > 0 else 0,
        })

    trends = {
        "size_trend": _calculate_trend(sizes),
        "frequency": _calculate_frequency(epochs),
        "total_checkpoints": len(checkpoints_with_epoch),
        "epoch_range": {"min": min(epochs), "max": max(epochs)} if epochs else {},
    }

    return {
        "progression": progression,
        "trends": trends,
    }


def find_best_checkpoint(
    checkpoints: list[dict[str, Any]],
    criteria: str = "latest",
) -> dict[str, Any] | None:
    """Find best checkpoint by criteria.

    Args:
        checkpoints: List of checkpoint dictionaries
        criteria: Criteria to use ("latest", "largest", "smallest", "earliest")

    Returns:
        Best checkpoint dictionary or None
    """
    if not checkpoints:
        return None

    checkpoints_with_epoch = [c for c in checkpoints if "epoch" in c]
    if not checkpoints_with_epoch:
        return None

    if criteria == "latest":
        return max(checkpoints_with_epoch, key=lambda x: x.get("epoch", 0))
    elif criteria == "earliest":
        return min(checkpoints_with_epoch, key=lambda x: x.get("epoch", 0))
    elif criteria == "largest":
        return max(checkpoints_with_epoch, key=lambda x: x.get("size", 0))
    elif criteria == "smallest":
        return min(checkpoints_with_epoch, key=lambda x: x.get("size", 0))
    else:
        return checkpoints_with_epoch[-1]


def analyze_checkpoint_usage(
    checkpoints: list[dict[str, Any]],
    time_window_days: int = 30,
) -> dict[str, Any]:
    """Analyze checkpoint usage patterns.

    Args:
        checkpoints: List of checkpoint dictionaries
        time_window_days: Time window in days to analyze (default: 30)

    Returns:
        Usage analysis dictionary
    """
    from datetime import datetime, timedelta

    if not checkpoints:
        return {"usage": {}, "patterns": {}}

    cutoff_date = datetime.now() - timedelta(days=time_window_days)

    recent_checkpoints = []
    for checkpoint in checkpoints:
        try:
            last_modified = datetime.fromisoformat(checkpoint.get("last_modified", "").replace("Z", "+00:00"))
            if last_modified.replace(tzinfo=None) >= cutoff_date:
                recent_checkpoints.append(checkpoint)
        except (ValueError, TypeError):
            continue

    usage = {
        "total_checkpoints": len(checkpoints),
        "recent_checkpoints": len(recent_checkpoints),
        "time_window_days": time_window_days,
    }

    patterns = {
        "creation_rate": len(recent_checkpoints) / time_window_days if time_window_days > 0 else 0,
        "average_size": sum(c.get("size", 0) for c in recent_checkpoints) / len(recent_checkpoints) if recent_checkpoints else 0,
    }

    return {
        "usage": usage,
        "patterns": patterns,
    }


def get_checkpoint_statistics(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Get statistics about checkpoints.

    Args:
        checkpoints: List of checkpoint dictionaries

    Returns:
        Statistics dictionary
    """
    if not checkpoints:
        return {"statistics": {}}

    sizes = [c.get("size", 0) for c in checkpoints]
    epochs = [c.get("epoch", 0) for c in checkpoints if "epoch" in c]

    stats = {
        "total_count": len(checkpoints),
        "with_epoch": len(epochs),
        "size_stats": {
            "total": sum(sizes),
            "average": sum(sizes) / len(sizes) if sizes else 0,
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
        },
    }

    if epochs:
        stats["epoch_stats"] = {
            "min": min(epochs),
            "max": max(epochs),
            "range": max(epochs) - min(epochs),
        }

    return {"statistics": stats}


def compare_checkpoints_across_runs(
    runs_data: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare checkpoints across multiple runs.

    Args:
        runs_data: Dictionary mapping run names to checkpoint lists

    Returns:
        Comparison analysis dictionary
    """
    if not runs_data:
        return {"comparisons": {}}

    comparisons = {}
    for run_name, checkpoints in runs_data.items():
        stats = get_checkpoint_statistics(checkpoints)
        comparisons[run_name] = {
            "count": len(checkpoints),
            "size_stats": stats.get("statistics", {}).get("size_stats", {}),
            "epoch_stats": stats.get("statistics", {}).get("epoch_stats", {}),
        }

    return {
        "runs": list(runs_data.keys()),
        "comparisons": comparisons,
        "total_runs": len(runs_data),
    }


def _calculate_trend(values: list[float]) -> str:
    """Calculate trend in values."""
    if len(values) < 2:
        return "stable"

    first_half = values[:len(values) // 2]
    second_half = values[len(values) // 2:]

    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0

    if second_avg > first_avg * 1.05:
        return "increasing"
    elif second_avg < first_avg * 0.95:
        return "decreasing"
    else:
        return "stable"


def _calculate_frequency(epochs: list[int]) -> float:
    """Calculate checkpoint frequency (checkpoints per epoch)."""
    if len(epochs) < 2:
        return 0.0

    epoch_range = max(epochs) - min(epochs)
    return len(epochs) / epoch_range if epoch_range > 0 else 0.0


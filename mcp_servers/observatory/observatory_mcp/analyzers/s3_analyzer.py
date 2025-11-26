"""S3 checkpoint analysis functions."""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def analyze_checkpoint_progression(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze checkpoint progression over time."""
    checkpoints_with_epoch = [c for c in checkpoints if "epoch" in c] if checkpoints else []
    if not checkpoints_with_epoch:
        return {"progression": [], "trends": {}}

    checkpoints_with_epoch.sort(key=lambda x: x.get("epoch", 0))

    epochs = [c.get("epoch", 0) for c in checkpoints_with_epoch]
    sizes = [c.get("size", 0) for c in checkpoints_with_epoch]

    progression = []
    for i, checkpoint in enumerate(checkpoints_with_epoch):
        progression.append(
            {
                "epoch": checkpoint.get("epoch", 0),
                "size": checkpoint.get("size", 0),
                "size_delta": sizes[i] - sizes[i - 1] if i > 0 else 0,
                "epoch_delta": epochs[i] - epochs[i - 1] if i > 0 else 0,
            }
        )

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
    """Find best checkpoint by criteria."""
    checkpoints_with_epoch = [c for c in checkpoints if "epoch" in c] if checkpoints else []
    if not checkpoints_with_epoch:
        return None

    criteria_map = {
        "latest": lambda: max(checkpoints_with_epoch, key=lambda x: x.get("epoch", 0)),
        "earliest": lambda: min(checkpoints_with_epoch, key=lambda x: x.get("epoch", 0)),
        "largest": lambda: max(checkpoints_with_epoch, key=lambda x: x.get("size", 0)),
        "smallest": lambda: min(checkpoints_with_epoch, key=lambda x: x.get("size", 0)),
    }
    return criteria_map.get(criteria, lambda: checkpoints_with_epoch[-1])()


def analyze_checkpoint_usage(
    checkpoints: list[dict[str, Any]],
    time_window_days: int = 30,
) -> dict[str, Any]:
    """Analyze checkpoint usage patterns."""

    if not checkpoints:
        return {"usage": {}, "patterns": {}}

    cutoff_date = datetime.now() - timedelta(days=time_window_days)

    recent_checkpoints = []
    for checkpoint in checkpoints:
            last_modified = datetime.fromisoformat(checkpoint.get("last_modified", "").replace("Z", "+00:00"))
            if last_modified.replace(tzinfo=None) >= cutoff_date:
                recent_checkpoints.append(checkpoint)

    usage = {
        "total_checkpoints": len(checkpoints),
        "recent_checkpoints": len(recent_checkpoints),
        "time_window_days": time_window_days,
    }

    patterns = {
        "creation_rate": len(recent_checkpoints) / time_window_days if time_window_days > 0 else 0,
        "average_size": (
            sum(c.get("size", 0) for c in recent_checkpoints) / len(recent_checkpoints) if recent_checkpoints else 0
        ),
    }

    return {
        "usage": usage,
        "patterns": patterns,
    }


def get_checkpoint_statistics(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Get statistics about checkpoints."""
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
    """Compare checkpoints across multiple runs."""
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

    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]

    first_avg = float(np.mean(first_half)) if first_half else 0.0
    second_avg = float(np.mean(second_half)) if second_half else 0.0

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

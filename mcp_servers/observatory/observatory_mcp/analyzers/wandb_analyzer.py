"""WandB analysis functions."""

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compare_runs(
    runs_data: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, Any]:
    """Compare multiple WandB runs."""
    if not runs_data:
        return {"runs": [], "comparisons": {}}

    comparisons = {}
    for key in metric_keys:
        values = []
        for run in runs_data:
            if key in run.get("summary", {}):
                values.append(run["summary"][key])

        if values:
            comparisons[key] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }

    return {
        "runs": runs_data,
        "comparisons": comparisons,
        "metric_keys": metric_keys,
    }


def analyze_learning_curves(
    metrics_data: list[dict[str, Any]],
    metric_keys: list[str],
    smoothing_window: int = 10,
) -> dict[str, Any]:
    """Analyze learning curves for trends and convergence."""
    analysis = {}

    for key in metric_keys:
        values = [point.get(key, 0) for point in metrics_data if key in point]
        if len(values) < 2:
            continue

        smoothed = _smooth_values(values, smoothing_window)
        trend = _detect_trend(smoothed)
        convergence = _detect_convergence(smoothed)
        plateau = _detect_plateau(smoothed)

        analysis[key] = {
            "trend": trend,
            "convergence": convergence,
            "plateau": plateau,
            "rate_of_change": _calculate_rate_of_change(smoothed),
            "final_value": smoothed[-1] if smoothed else None,
        }

    return {
        "metric_analyses": analysis,
        "total_samples": len(metrics_data),
    }


def identify_critical_moments(
    metrics_data: list[dict[str, Any]],
    metric_keys: list[str],
    threshold: float = 0.1,
) -> list[dict[str, Any]]:
    """Identify critical moments in training."""
    moments = []

    for key in metric_keys:
        values = [point.get(key, 0) for point in metrics_data if key in point]
        if len(values) < 3:
            continue

        for i in range(1, len(values) - 1):
            prev = values[i - 1]
            curr = values[i]
            next_val = values[i + 1]

            change_pct = abs((curr - prev) / prev) if prev != 0 else 0

            if change_pct > threshold:
                if curr > prev and next_val > curr:
                    moments.append(
                        {
                            "type": "breakthrough",
                            "metric": key,
                            "step": i,
                            "value": curr,
                            "change_percent": change_pct * 100,
                        }
                    )
                elif curr < prev and next_val < curr:
                    moments.append(
                        {
                            "type": "drop",
                            "metric": key,
                            "step": i,
                            "value": curr,
                            "change_percent": change_pct * 100,
                        }
                    )

    return sorted(moments, key=lambda x: x.get("change_percent", 0), reverse=True)


def correlate_metrics(
    metrics_data: list[dict[str, Any]],
    metric_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Calculate correlations between metric pairs."""
    correlations = {}

    for key1, key2 in metric_pairs:
        values1 = [point.get(key1, 0) for point in metrics_data if key1 in point]
        values2 = [point.get(key2, 0) for point in metrics_data if key2 in point]

        if len(values1) == len(values2) and len(values1) >= 2:
            correlation, p_value = stats.pearsonr(values1, values2)

            correlations[f"{key1}:{key2}"] = {
                "correlation": float(correlation),
                "p_value": float(p_value),
                "strength": _interpret_correlation(float(correlation)),
            }

    return {
        "correlations": correlations,
        "total_pairs": len(metric_pairs),
    }


def _smooth_values(values: list[float], window: int) -> list[float]:
    """Apply moving average smoothing."""
    if len(values) < window:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        window_values = values[start:end]
        smoothed.append(sum(window_values) / len(window_values))

    return smoothed


def _detect_trend(values: list[float]) -> str:
    """Detect trend in values."""
    if len(values) < 2:
        return "unknown"

    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]

    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0

    if second_avg > first_avg * 1.05:
        return "improving"
    elif second_avg < first_avg * 0.95:
        return "declining"
    else:
        return "stable"


def _detect_convergence(values: list[float], threshold: float = 0.01) -> bool:
    """Detect if values have converged."""
    if len(values) < 10:
        return False

    recent = values[-10:]
    variance = float(np.var(recent))
    mean = float(np.mean(recent))

    return variance / (mean * mean) < threshold if mean != 0 else False


def _detect_plateau(values: list[float], threshold: float = 0.01) -> bool:
    """Detect if values have plateaued."""
    if len(values) < 5:
        return False

    recent = values[-5:]
    change = abs(recent[-1] - recent[0])
    mean = float(np.mean(recent))

    return change / mean < threshold if mean != 0 else False


def _calculate_rate_of_change(values: list[float]) -> float:
    """Calculate average rate of change."""
    if len(values) < 2:
        return 0.0

    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    return sum(changes) / len(changes) if changes else 0.0


def _interpret_correlation(correlation: float) -> str:
    """Interpret correlation strength."""
    abs_corr = abs(correlation)
    if abs_corr >= 0.7:
        return "strong"
    elif abs_corr >= 0.4:
        return "moderate"
    elif abs_corr >= 0.2:
        return "weak"
    else:
        return "negligible"

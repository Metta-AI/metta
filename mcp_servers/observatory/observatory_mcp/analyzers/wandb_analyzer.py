"""WandB analysis functions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compare_runs(
    runs_data: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, Any]:
    """Compare multiple WandB runs.

    Args:
        runs_data: List of run data dictionaries with metrics
        metric_keys: List of metric keys to compare

    Returns:
        Comparison analysis dictionary
    """
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
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": _calculate_std(values),
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
    """Analyze learning curves for trends and convergence.

    Args:
        metrics_data: List of metric data points
        metric_keys: List of metric keys to analyze
        smoothing_window: Window size for smoothing (default: 10)

    Returns:
        Learning curve analysis dictionary
    """
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
    """Identify critical moments in training.

    Args:
        metrics_data: List of metric data points
        metric_keys: List of metric keys to analyze
        threshold: Threshold for detecting significant changes

    Returns:
        List of critical moments
    """
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
    """Calculate correlations between metric pairs.

    Args:
        metrics_data: List of metric data points
        metric_pairs: List of (metric1, metric2) tuples

    Returns:
        Correlation analysis dictionary
    """
    correlations = {}

    for key1, key2 in metric_pairs:
        values1 = [point.get(key1, 0) for point in metrics_data if key1 in point]
        values2 = [point.get(key2, 0) for point in metrics_data if key2 in point]

        if len(values1) == len(values2) and len(values1) >= 2:
            correlation = _pearson_correlation(values1, values2)
            p_value = _calculate_p_value(values1, values2, correlation)

            correlations[f"{key1}:{key2}"] = {
                "correlation": correlation,
                "p_value": p_value,
                "strength": _interpret_correlation(correlation),
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
    variance = _calculate_variance(recent)
    mean = sum(recent) / len(recent)

    return variance / (mean * mean) < threshold if mean != 0 else False


def _detect_plateau(values: list[float], threshold: float = 0.01) -> bool:
    """Detect if values have plateaued."""
    if len(values) < 5:
        return False

    recent = values[-5:]
    change = abs(recent[-1] - recent[0])
    mean = sum(recent) / len(recent)

    return change / mean < threshold if mean != 0 else False


def _calculate_rate_of_change(values: list[float]) -> float:
    """Calculate average rate of change."""
    if len(values) < 2:
        return 0.0

    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    return sum(changes) / len(changes) if changes else 0.0


def _calculate_std(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5


def _calculate_variance(values: list[float]) -> float:
    """Calculate variance."""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _calculate_p_value(x: list[float], y: list[float], correlation: float) -> float:
    """Calculate approximate p-value for correlation."""
    if len(x) < 3:
        return 1.0

    n = len(x)
    t_stat = correlation * ((n - 2) / (1 - correlation * correlation)) ** 0.5
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    return p_value


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF."""
    import math

    return 0.5 * (1 + math.erf(x / (2**0.5)))


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

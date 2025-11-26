"""Training context analysis for WandB runs."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class WandbTrainingContext:
    """Complete training context analysis from Wandb data."""

    run_id: str
    run_name: str
    training_stage: str
    learning_velocity: float
    performance_stability: float
    behavioral_adaptation_rate: float
    metric_correlations: dict[str, float]
    critical_moments: list[dict[str, Any]]
    learning_progressions: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "training_stage": self.training_stage,
            "learning_velocity": self.learning_velocity,
            "performance_stability": self.performance_stability,
            "behavioral_adaptation_rate": self.behavioral_adaptation_rate,
            "metric_correlations": self.metric_correlations,
            "critical_moments": self.critical_moments,
            "learning_progressions": self.learning_progressions,
        }


def analyze_training_progression(
    metrics_data: list[dict[str, Any]],
    metric_keys: list[str],
    context_window_steps: int = 1000,
    center_step: Optional[int] = None,
) -> WandbTrainingContext:
    """Analyze training progression from metrics data."""
    if not metrics_data:
        raise ValueError("No metrics data provided")

    if center_step is None:
        if metrics_data:
            center_step = len(metrics_data) // 2
        else:
            center_step = 0

    window_start = max(0, center_step - context_window_steps // 2)
    window_end = min(len(metrics_data), center_step + context_window_steps // 2)
    window_data = metrics_data[window_start:window_end]

    learning_progressions = {}
    for key in metric_keys:
        if key in window_data[0] if window_data else {}:
            values = [point.get(key, 0) for point in window_data if key in point]
            learning_progressions[key] = values

    learning_velocity = _calculate_learning_velocity(window_data, metric_keys)
    performance_stability = _calculate_performance_stability(window_data, metric_keys)
    behavioral_adaptation_rate = _calculate_behavioral_adaptation_rate(window_data)
    metric_correlations = _calculate_metric_correlations(window_data, metric_keys)
    critical_moments = _identify_critical_moments(window_data, metric_keys)
    training_stage = _determine_training_stage(len(metrics_data), center_step)

    return WandbTrainingContext(
        run_id="",
        run_name="",
        training_stage=training_stage,
        learning_velocity=learning_velocity,
        performance_stability=performance_stability,
        behavioral_adaptation_rate=behavioral_adaptation_rate,
        metric_correlations=metric_correlations,
        critical_moments=critical_moments,
        learning_progressions=learning_progressions,
    )


def _calculate_learning_velocity(metrics_data: list[dict[str, Any]], metric_keys: list[str]) -> float:
    """Calculate learning velocity (rate of improvement)."""
    if len(metrics_data) < 2:
        return 0.0

    velocities = []
    for key in metric_keys:
        values = [point.get(key, 0) for point in metrics_data if key in point]
        if len(values) >= 2:
            velocity = (values[-1] - values[0]) / len(values)
            velocities.append(velocity)

    return sum(velocities) / len(velocities) if velocities else 0.0


def _calculate_performance_stability(metrics_data: list[dict[str, Any]], metric_keys: list[str]) -> float:
    """Calculate performance stability (lower variance = more stable)."""
    if not metrics_data:
        return 0.0

    stabilities = []
    for key in metric_keys:
        values = [point.get(key, 0) for point in metrics_data if key in point]
        if len(values) >= 2:
            variance = float(np.var(values))
            stability = 1.0 / (1.0 + variance)
            stabilities.append(stability)

    return sum(stabilities) / len(stabilities) if stabilities else 0.0


def _calculate_behavioral_adaptation_rate(metrics_data: list[dict[str, Any]]) -> float:
    """Calculate behavioral adaptation rate (rate of change in metrics).

    Measures how quickly behavior is adapting by calculating the average absolute rate of change across all metrics."""
    if len(metrics_data) < 2:
        return 0.0

    metric_keys = set()
    for point in metrics_data:
        metric_keys.update(key for key in point.keys() if isinstance(point.get(key), (int, float)))

    if not metric_keys:
        return 0.0

    adaptation_rates = []
    for key in metric_keys:
        values = [point.get(key, 0) for point in metrics_data if key in point]
        if len(values) < 2:
            continue

        changes = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        if changes:
            avg_change = sum(changes) / len(changes)
            adaptation_rates.append(avg_change)

    return sum(adaptation_rates) / len(adaptation_rates) if adaptation_rates else 0.0


def _calculate_metric_correlations(metrics_data: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    """Calculate correlations between metrics."""
    correlations = {}
    if len(metric_keys) < 2 or len(metrics_data) < 2:
        return correlations

    for i, key1 in enumerate(metric_keys):
        for key2 in metric_keys[i + 1 :]:
            values1 = [point.get(key1, 0) for point in metrics_data if key1 in point]
            values2 = [point.get(key2, 0) for point in metrics_data if key2 in point]
            if len(values1) == len(values2) and len(values1) >= 2:
                correlation = _pearson_correlation(values1, values2)
                pair_key = f"{key1}:{key2}"
                correlations[pair_key] = correlation

    return correlations


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    from scipy import stats

    correlation, _ = stats.pearsonr(x, y)
    return float(correlation)


def _identify_critical_moments(metrics_data: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    """Identify critical learning moments (breakthroughs, drops, plateaus)."""
    moments = []
    if len(metrics_data) < 3:
        return moments

    for key in metric_keys:
        values = [point.get(key, 0) for point in metrics_data if key in point]
        if len(values) < 3:
            continue

        for i in range(1, len(values) - 1):
            prev = values[i - 1]
            curr = values[i]
            next_val = values[i + 1]

            change = curr - prev
            next_change = next_val - curr

            if abs(change) > abs(prev) * 0.1:
                if change > 0 and next_change > 0:
                    moments.append(
                        {
                            "type": "breakthrough",
                            "metric": key,
                            "step": i,
                            "value": curr,
                            "change": change,
                        }
                    )
                elif change < 0 and next_change < 0:
                    moments.append(
                        {
                            "type": "drop",
                            "metric": key,
                            "step": i,
                            "value": curr,
                            "change": change,
                        }
                    )
                elif abs(change) < abs(prev) * 0.01:
                    moments.append(
                        {
                            "type": "plateau",
                            "metric": key,
                            "step": i,
                            "value": curr,
                        }
                    )

    return moments


def _determine_training_stage(total_steps: int, current_step: int) -> str:
    """Determine training stage based on progress."""
    if total_steps == 0:
        return "unknown"

    progress = current_step / total_steps if total_steps > 0 else 0.0

    if progress < 0.33:
        return "early"
    elif progress < 0.66:
        return "mid"
    else:
        return "late"

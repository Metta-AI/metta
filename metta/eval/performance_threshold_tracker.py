"""
Performance threshold tracking for training and evaluation.

This module provides functionality to track when smoothed performance metrics
reach specified thresholds and calculate the associated samples, time, and cost.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThreshold:
    """Configuration for a performance threshold to track."""

    name: str
    metric: str
    target_value: float
    comparison: str = ">="  # ">=", "<=", "=="
    smoothing_factor: float = 0.1


@dataclass
class ThresholdResult:
    """Result of performance threshold tracking."""

    threshold_name: str
    metric: str
    target_value: float
    samples_to_threshold: Optional[int] = None
    minutes_to_threshold: Optional[float] = None
    cost_to_threshold: Optional[float] = None
    achieved: bool = False
    final_smoothed_value: Optional[float] = None


class PerformanceThresholdTracker:
    """
    Tracks performance thresholds and calculates samples/time/cost to reach them.

    Uses exponential moving average smoothing and tracks when smoothed values
    reach specified thresholds.
    """

    def __init__(self, thresholds: List[PerformanceThreshold], available_metrics: Optional[Set[str]] = None):
        self.thresholds = thresholds
        self.available_metrics = available_metrics or set()

        # Validate metrics if available_metrics provided
        if self.available_metrics:
            self._validate_metrics()

        self.smoothed_values = {t.name: None for t in thresholds}
        self.threshold_reached = {t.name: False for t in thresholds}
        self.results = {
            t.name: ThresholdResult(threshold_name=t.name, metric=t.metric, target_value=t.target_value)
            for t in thresholds
        }

        # Training tracking
        self.start_time = time.time()
        self.start_samples = 0
        self.current_samples = 0
        self.current_time = 0

    def _validate_metrics(self):
        """Validate that all threshold metrics are available."""
        for threshold in self.thresholds:
            if threshold.metric not in self.available_metrics:
                logger.warning(
                    f"Metric '{threshold.metric}' for threshold '{threshold.name}' not available in environment"
                )
                logger.warning(f"Available metrics: {sorted(self.available_metrics)}")

        # Cost calculation (AWS pricing)
        self.aws_pricing = {
            "g4dn.xlarge": {"on_demand": 0.526, "spot": 0.1578},  # 1 GPU
            "g5.xlarge": {"on_demand": 1.006, "spot": 0.3018},  # 1 GPU
            "g5.2xlarge": {"on_demand": 1.212, "spot": 0.3636},  # 1 GPU
            "g5.4xlarge": {"on_demand": 2.424, "spot": 0.7272},  # 1 GPU
            "g5.8xlarge": {"on_demand": 4.848, "spot": 1.4544},  # 1 GPU
            "g5.12xlarge": {"on_demand": 7.272, "spot": 2.1816},  # 4 GPU
            "g5.24xlarge": {"on_demand": 14.544, "spot": 4.3632},  # 8 GPU
            "p3.2xlarge": {"on_demand": 3.06, "spot": 0.918},  # 1 GPU
            "p3.8xlarge": {"on_demand": 12.24, "spot": 3.672},  # 4 GPU
            "p3.16xlarge": {"on_demand": 24.48, "spot": 7.344},  # 8 GPU
        }

    def update(
        self, metrics: Dict[str, float], samples: int, instance_type: str = "g5.4xlarge", use_spot: bool = False
    ):
        """
        Update tracker with new metrics and check for threshold crossings.

        Args:
            metrics: Dictionary of metric_name -> value
            samples: Current number of samples/timesteps
            instance_type: AWS instance type for cost calculation
            use_spot: Whether using spot instances
        """
        self.current_samples = samples
        self.current_time = time.time() - self.start_time

        for threshold in self.thresholds:
            if threshold.metric not in metrics:
                continue

            current_value = metrics[threshold.metric]

            # Update smoothed value
            if self.smoothed_values[threshold.name] is None:
                self.smoothed_values[threshold.name] = current_value
            else:
                self.smoothed_values[threshold.name] = (
                    threshold.smoothing_factor * current_value
                    + (1 - threshold.smoothing_factor) * self.smoothed_values[threshold.name]
                )

                # Check if threshold was just reached
            if not self.threshold_reached[threshold.name] and self._check_threshold(
                threshold, self.smoothed_values[threshold.name]
            ):
                self.threshold_reached[threshold.name] = True
                result = self.results[threshold.name]
                result.achieved = True
                result.samples_to_threshold = samples
                result.minutes_to_threshold = self.current_time / 60.0
                result.cost_to_threshold = self._calculate_cost(
                    self.current_time / 3600.0,  # Convert to hours
                    instance_type,
                    use_spot,
                )
                result.final_smoothed_value = self.smoothed_values[threshold.name]

                logger.info(
                    f"Threshold '{threshold.name}' ({threshold.metric} >= {threshold.target_value}) "
                    f"reached at {samples} samples, {result.minutes_to_threshold:.1f} minutes, "
                    f"${result.cost_to_threshold:.2f}"
                )

    def _calculate_cost(self, hours: float, instance_type: str, use_spot: bool) -> float:
        """Calculate AWS cost for given hours and instance type."""
        if instance_type not in self.aws_pricing:
            logger.warning(f"Unknown instance type {instance_type}, using g5.4xlarge pricing")
            instance_type = "g5.4xlarge"

        pricing = self.aws_pricing[instance_type]
        price_per_hour = pricing["spot"] if use_spot else pricing["on_demand"]

        return hours * price_per_hour

    def _check_threshold(self, threshold: PerformanceThreshold, value: float) -> bool:
        """Check if a value meets the threshold criteria."""
        if threshold.comparison == ">=":
            return value >= threshold.target_value
        elif threshold.comparison == "<=":
            return value <= threshold.target_value
        elif threshold.comparison == "==":
            return abs(value - threshold.target_value) < 1e-6
        else:
            logger.warning(f"Unknown comparison operator: {threshold.comparison}")
            return False

    def get_results(self) -> Dict[str, ThresholdResult]:
        """Get current threshold results."""
        # Update final smoothed values for all thresholds
        for threshold in self.thresholds:
            if self.smoothed_values[threshold.name] is not None:
                self.results[threshold.name].final_smoothed_value = self.smoothed_values[threshold.name]

        return self.results

    def get_wandb_metrics(self) -> Dict[str, Any]:
        """Convert results to WandB metrics format."""
        metrics = {}
        results = self.get_results()

        for threshold_name, result in results.items():
            prefix = f"performance_threshold/{threshold_name}"

            # Basic threshold info
            metrics[f"{prefix}/metric"] = result.metric
            metrics[f"{prefix}/target_value"] = result.target_value
            metrics[f"{prefix}/achieved"] = result.achieved
            metrics[f"{prefix}/final_smoothed_value"] = result.final_smoothed_value

            # Time/samples/cost to threshold
            if result.achieved:
                metrics[f"{prefix}/samples_to_threshold"] = result.samples_to_threshold
                metrics[f"{prefix}/minutes_to_threshold"] = result.minutes_to_threshold
                metrics[f"{prefix}/cost_to_threshold"] = result.cost_to_threshold
            else:
                # Use NaN for unachieved thresholds
                metrics[f"{prefix}/samples_to_threshold"] = float("nan")
                metrics[f"{prefix}/minutes_to_threshold"] = float("nan")
                metrics[f"{prefix}/cost_to_threshold"] = float("nan")

        return metrics

    def reset(self):
        """Reset tracker state for new training run."""
        self.smoothed_values = {t.name: None for t in self.thresholds}
        self.threshold_reached = {t.name: False for t in self.thresholds}
        self.results = {
            t.name: ThresholdResult(threshold_name=t.name, metric=t.metric, target_value=t.target_value)
            for t in self.thresholds
        }
        self.start_time = time.time()
        self.start_samples = 0
        self.current_samples = 0
        self.current_time = 0

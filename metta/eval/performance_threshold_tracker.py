"""
Performance threshold tracking for training runs.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from metta.eval.aws_pricing import calculate_total_cost

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThreshold:
    """Configuration for a performance threshold."""

    name: str
    metric: str
    target_value: float
    comparison: str = ">="  # ">=", "<=", "==", etc.
    smoothing_factor: float = 0.1


@dataclass
class ThresholdResult:
    """Result of threshold tracking."""

    achieved: bool = False
    samples_to_threshold: Optional[int] = None
    minutes_to_threshold: Optional[float] = None
    cost_to_threshold: Optional[float] = None
    final_smoothed_value: Optional[float] = None


class PerformanceThresholdTracker:
    """Track performance thresholds during training."""

    def __init__(
        self,
        thresholds: List[PerformanceThreshold],
        available_metrics: List[str],
        aws_config: Optional[Dict] = None,
    ):
        """Initialize the tracker.

        Args:
            thresholds: List of thresholds to track
            available_metrics: List of available metric names
            aws_config: AWS configuration for cost calculation
        """
        self.thresholds = thresholds
        self.available_metrics = set(available_metrics)
        self.aws_config = aws_config or {}

        # Initialize tracking state
        self.smoothed_values = {threshold.name: 0.0 for threshold in thresholds}
        self.threshold_reached = {threshold.name: False for threshold in thresholds}
        self.results = {threshold.name: ThresholdResult() for threshold in thresholds}
        self.current_time = 0.0  # in seconds

        # Validate metrics
        self._validate_metrics()

    def _validate_metrics(self):
        """Validate that all threshold metrics are available."""
        for threshold in self.thresholds:
            if threshold.metric not in self.available_metrics:
                logger.warning(
                    f"Metric '{threshold.metric}' for threshold '{threshold.name}' not available in environment"
                )
                logger.warning(f"Available metrics: {sorted(self.available_metrics)}")

    def update(
        self,
        metrics: Dict[str, float],
        samples: int,
        elapsed_time: float = None,
        instance_type: str = None,
        use_spot: bool = None,
        num_nodes: int = None,
        num_gpus_per_node: int = None,
        region: str = "us-east-1",
        profile: str = "softmax",
    ):
        """Update tracker with new metrics.

        Args:
            metrics: Dictionary of metric name to value
            samples: Number of samples processed so far
            elapsed_time: Time elapsed in seconds (if None, will use current_time)
            instance_type: AWS instance type (if None, will be detected)
            use_spot: Whether using spot instances (if None, will be detected)
            num_nodes: Number of nodes (if None, will be detected)
            num_gpus_per_node: Number of GPUs per node (if None, will be detected)
            region: AWS region
            profile: AWS profile to use
        """
        if elapsed_time is not None:
            self.current_time = elapsed_time

        # Update smoothed values for each threshold
        for threshold in self.thresholds:
            if threshold.metric in metrics:
                current_value = metrics[threshold.metric]
                smoothed_value = self.smoothed_values[threshold.name]

                # Apply exponential moving average
                alpha = threshold.smoothing_factor
                new_smoothed_value = alpha * current_value + (1 - alpha) * smoothed_value
                self.smoothed_values[threshold.name] = new_smoothed_value

                # Check if threshold is reached
                if not self.threshold_reached[threshold.name] and self._check_threshold(threshold, new_smoothed_value):
                    self.threshold_reached[threshold.name] = True
                    result = self.results[threshold.name]
                    result.achieved = True
                    result.samples_to_threshold = samples
                    result.minutes_to_threshold = self.current_time / 60.0

                    # Calculate cost using the new pricing system
                    hours = self.current_time / 3600.0
                    result.cost_to_threshold = calculate_total_cost(
                        hours=hours,
                        instance_type=instance_type,
                        use_spot=use_spot,
                        num_nodes=num_nodes,
                        num_gpus_per_node=num_gpus_per_node,
                        region=region,
                        profile=profile,
                    )
                    result.final_smoothed_value = new_smoothed_value

                    logger.info(
                        f"Threshold '{threshold.name}' "
                        f"({threshold.metric} {threshold.comparison} {threshold.target_value}) "
                        f"reached at {samples} samples, {result.minutes_to_threshold:.1f} minutes, "
                        f"${result.cost_to_threshold:.2f}"
                    )

    def _check_threshold(self, threshold: PerformanceThreshold, value: float) -> bool:
        """Check if a threshold is met."""
        if threshold.comparison == ">=":
            return value >= threshold.target_value
        elif threshold.comparison == "<=":
            return value <= threshold.target_value
        elif threshold.comparison == "==":
            return abs(value - threshold.target_value) < 1e-6
        elif threshold.comparison == ">":
            return value > threshold.target_value
        elif threshold.comparison == "<":
            return value < threshold.target_value
        else:
            logger.warning(f"Unknown comparison operator: {threshold.comparison}")
            return False

    def get_results(self) -> Dict[str, ThresholdResult]:
        """Get current threshold results."""
        return self.results

    def get_smoothed_values(self) -> Dict[str, float]:
        """Get current smoothed metric values."""
        return self.smoothed_values

    def is_complete(self) -> bool:
        """Check if all thresholds have been reached."""
        return all(self.threshold_reached.values())

    def get_wandb_metrics(self) -> Dict[str, Any]:
        """Convert results to WandB metrics format."""
        metrics = {}
        results = self.get_results()

        for threshold_name, result in results.items():
            prefix = f"performance_threshold/{threshold_name}"

            # Basic threshold info
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

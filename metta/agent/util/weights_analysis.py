import logging
from typing import Any, Dict, List, Union

import torch
from omegaconf import DictConfig, ListConfig
from scipy import stats

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent

# Set up logger
logger = logging.getLogger(__name__)


def analyze_weights(weights: torch.Tensor, delta: float = 0.01) -> Dict[str, float]:
    """Analyze weight matrix properties including singular values, effective rank,
    weight norms and other dynamics-related metrics.

    Args:
        weights: Weight tensor to analyze
        delta: Threshold for effective rank calculation (default: 0.01)

    Returns:
        Dictionary containing all computed metrics including:
        - Singular value statistics (ratio, largest, mean, collapsed dimensions)
        - Power law fit metrics (R² value, slope)
        - Weight norm
        - Effective rank
        - Error indicators for problematic tensors
    """
    metrics: Dict[str, float] = {}

    # Basic weight statistics
    try:
        weight_norm = torch.linalg.norm(weights).item()
        metrics["weight_norm"] = weight_norm
    except Exception as e:
        logger.warning(f"Could not compute weight norm: {e}")
        metrics["weight_norm"] = 0.0
        metrics["error_code"] = 1.0
        return metrics

    # Check for conditions that would make SVD analysis invalid
    if weights.numel() == 0:
        logger.warning("Cannot analyze empty tensor")
        metrics["error_code"] = 2.0
        metrics["error_type"] = 1.0  # Empty tensor
        return metrics

    if torch.isnan(weights).any() or torch.isinf(weights).any():
        logger.warning("Tensor contains NaN or infinite values")
        metrics["avg_abs_weight"] = (
            torch.abs(weights[~torch.isnan(weights) & ~torch.isinf(weights)]).mean().item()
            if torch.any(~torch.isnan(weights) & ~torch.isinf(weights))
            else 0.0
        )
        metrics["error_code"] = 3.0
        metrics["error_type"] = 2.0  # NaN/Inf values
        return metrics

    if weights.ndim < 2:
        logger.warning(f"Tensor must be at least 2-dimensional, got shape {weights.shape}")
        metrics["avg_abs_weight"] = torch.abs(weights).mean().item()
        metrics["error_code"] = 4.0
        metrics["error_type"] = 3.0  # Wrong dimensions
        return metrics

    try:
        # Add small noise to prevent convergence issues
        noise = torch.randn_like(weights) * 1e-7
        _, S, _ = torch.linalg.svd(weights + noise)

        # Analyze singular values
        sorted_sv = torch.sort(S, descending=True).values
        sorted_sv_non_zero = sorted_sv[sorted_sv > 1e-10]  # ignore singular values close to zero

        # Check if we have non-zero singular values
        if len(sorted_sv_non_zero) == 0:
            logger.warning("No non-zero singular values found")
            metrics["error_code"] = 5.0
            metrics["error_type"] = 4.0  # No non-zero singular values
            return metrics

        # Basic SV metrics
        metrics["mean_sv"] = torch.mean(S).item()
        metrics["largest_sv"] = sorted_sv[0].item()
        metrics["collapsed_dim_pct"] = (len(sorted_sv) - len(sorted_sv_non_zero)) / max(len(sorted_sv), 1)
        metrics["sv_ratio"] = sorted_sv[0].item() / sorted_sv_non_zero[-1].item()

        # Power law fit analysis (indicator of criticality)
        if len(sorted_sv_non_zero) > 5:
            log_indices = torch.log(torch.arange(1, len(sorted_sv_non_zero) + 1, device=S.device).float())
            log_sv = torch.log(sorted_sv_non_zero + 1e-10)

            try:
                # Calculate linear regression for power law analysis
                result = stats.linregress(log_indices.cpu().numpy(), log_sv.cpu().numpy())

                slope = float(result[0])  # type: ignore
                rvalue = float(result[2])  # type: ignore

                r_squared = rvalue**2
                metrics["sv_loglog_r2"] = r_squared
                metrics["sv_loglog_slope"] = slope
            except Exception as e:
                logger.warning(f"Failed to compute linear regression: {e}")
                metrics["regression_error"] = 1.0

        # Calculate effective rank
        if len(S) > 1:
            try:
                total_sum = S.sum()
                cumulative_sum = torch.cumsum(S, dim=0)
                threshold = (1 - delta) * total_sum
                effective_rank_idx = torch.where(cumulative_sum >= threshold)[0]

                if len(effective_rank_idx) > 0:
                    effective_rank = effective_rank_idx[0].item() + 1
                    metrics["effective_rank"] = float(effective_rank)
                else:
                    logger.warning("Could not determine effective rank")
                    metrics["effective_rank_error"] = 1.0
            except Exception as e:
                logger.warning(f"Failed to compute effective rank: {e}")
                metrics["effective_rank_error"] = 2.0

    except Exception as e:
        logger.warning(f"SVD analysis failed: {str(e)}")
        metrics["error_code"] = 6.0
        metrics["svd_error"] = 1.0
        try:
            metrics["avg_abs_weight"] = torch.abs(weights).mean().item()
        except Exception:
            metrics["avg_abs_weight"] = 0.0

    return metrics


class WeightsMetricsHelper:
    """Helper class for computing and storing weight metrics during training."""

    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        """Initialize the helper with configuration.

        Args:
            cfg: Configuration object containing agent settings
        """
        self.cfg = cfg
        self._weight_metrics: List[Dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, policy: MettaAgent | DistributedMettaAgent) -> None:
        """Compute weight metrics at the end of specified epochs.

        Args:
            epoch: Current training epoch
            policy: The policy model to analyze
        """
        if not hasattr(self.cfg.agent, "analyze_weights_interval"):
            logger.warning("cfg.agent.analyze_weights_interval not found")
            return

        if self.cfg.agent.analyze_weights_interval == 0:
            self._weight_metrics = []
            return

        if epoch % self.cfg.agent.analyze_weights_interval == 0:
            try:
                metrics = policy.compute_weight_metrics()

                # Make sure metrics is a list
                if not isinstance(metrics, list):
                    raise ValueError(f"Expected list from policy.compute_weight_metrics(), got {type(metrics)}")

                # Validate each item in the list - must be dictionaries
                validated_metrics = []
                for i, item in enumerate(metrics):
                    if isinstance(item, dict):
                        # Make sure the dictionary has a name
                        if "name" not in item:
                            item["name"] = f"component_{i}"
                        validated_metrics.append(item)
                    else:
                        # Strict validation - raise an error on invalid types
                        raise ValueError(
                            f"Item {i} in metrics from policy.compute_weight_metrics() "
                            f"is not a dictionary. Got {type(item)}"
                        )

                self._weight_metrics = validated_metrics

            except Exception as e:
                logger.exception(f"Error in on_epoch_end: {e}")
                self._weight_metrics = []
        else:
            # Not a weight analysis epoch
            self._weight_metrics = []

    def stats(self) -> Dict[str, float]:
        """Format weight metrics for logging.

        Returns:
            Dictionary of formatted metrics ready for logging
        """
        formatted_metrics: Dict[str, float] = {}

        if not self._weight_metrics:
            return formatted_metrics

        try:
            for metrics in self._weight_metrics:
                if not isinstance(metrics, dict):
                    logger.warning(f"Expected dict, got {type(metrics)} in _weight_metrics")
                    continue

                name = str(metrics.get("name", "unknown"))

                for key, value in metrics.items():
                    if key == "name":
                        continue

                    try:
                        # Convert value to float for logging
                        float_value = float(value)
                        metric_key = f"train/{key}/{name}"
                        formatted_metrics[metric_key] = float_value
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert metric {key}={value} to float: {e}")

        except Exception as e:
            logger.exception(f"Error in stats method: {e}")

        return formatted_metrics

    def reset(self) -> None:
        """Reset the stored weight metrics."""
        self._weight_metrics = []

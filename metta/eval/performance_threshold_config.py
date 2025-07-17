"""
Configuration loading for performance thresholds.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from metta.eval.performance_threshold_tracker import PerformanceThreshold


def load_performance_thresholds(config_path: str) -> tuple[List[PerformanceThreshold], Dict[str, Any]]:
    """
    Load performance threshold configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (thresholds, aws_config)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Performance threshold config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load thresholds
    thresholds = []
    for threshold_config in config.get("thresholds", []):
        threshold = PerformanceThreshold(
            name=threshold_config["name"],
            metric=threshold_config["metric"],
            target_value=threshold_config["target_value"],
            comparison=threshold_config.get("comparison", ">="),
            smoothing_factor=threshold_config.get("smoothing_factor", 0.1),
        )
        thresholds.append(threshold)

    # Load AWS configuration
    aws_config = config.get("aws", {"instance_type": "g5.4xlarge", "use_spot": False})

    return thresholds, aws_config


def get_default_arena_thresholds() -> tuple[List[PerformanceThreshold], Dict[str, Any]]:
    """
    Get default arena environment performance thresholds.

    Returns:
        Tuple of (thresholds, aws_config)
    """
    # Default arena thresholds
    thresholds = [
        PerformanceThreshold(
            name="heart_gained_2",
            metric="env_agent/heart.gained",
            target_value=2.0,
            comparison=">=",
            smoothing_factor=0.1,
        ),
        PerformanceThreshold(
            name="heart_gained_5",
            metric="env_agent/heart.gained",
            target_value=5.0,
            comparison=">=",
            smoothing_factor=0.1,
        ),
    ]

    aws_config = {"instance_type": "g5.4xlarge", "use_spot": False}

    return thresholds, aws_config

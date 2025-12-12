"""Metric extraction from structured job results.

This module provides helper functions to extract training and eval metrics from
structured job_result dictionaries and map them to Datadog schema metrics.

These utilities are called directly by the stable runner workflow - they do NOT
parse logs or read runner outputs. The runner passes structured data it already
has in memory.

Job results are expected to be dictionaries with the following structure:
{
    "name": str,  # Job name
    "acceptance_passed": bool | None,  # Whether acceptance criteria passed
    "exit_code": int | None,  # Exit code (0 = success)
    "metrics": Dict[str, float],  # Job metrics dictionary
    "duration_s": float | None,  # Duration in seconds (for eval jobs)
}
"""

from __future__ import annotations

from typing import Dict


def determine_success(acceptance_passed: bool | None, exit_code: int | None) -> float:
    """Determine job success from acceptance and exit code.

    Args:
        acceptance_passed: True if all acceptance criteria passed, False if failed, None if not evaluated
        exit_code: 0 for success, non-zero for failure, None if not completed

    Returns:
        1.0 if successful, 0.0 if failed

    Success logic: (exit_code == 0) and (acceptance_passed is not False)
    """
    if exit_code is None:
        return 0.0
    if exit_code != 0:
        return 0.0
    if acceptance_passed is False:
        return 0.0
    return 1.0


def extract_hearts(metrics: Dict[str, float]) -> float:
    """Extract hearts metric from metrics dictionary.

    Args:
        metrics: Job metrics dictionary

    Returns:
        Hearts value, or 0.0 if not available
    """
    return float(metrics.get("env_game/assembler.heart.created", 0.0))


def extract_sps(metrics: Dict[str, float]) -> float:
    """Extract SPS (steps per second) metric from metrics dictionary.

    Args:
        metrics: Job metrics dictionary

    Returns:
        SPS value, or 0.0 if not available
    """
    return float(metrics.get("overview/sps", 0.0))


def extract_shaped(metrics: Dict[str, float]) -> float:
    """Extract shaped SPS metric from metrics dictionary.

    Args:
        metrics: Job metrics dictionary

    Returns:
        Shaped SPS value, or 0.0 if not available

    For multinode jobs, shaped = sps (same value as overview/sps).
    """
    # For multinode jobs, shaped uses the same value as sps
    return extract_sps(metrics)


def extract_heart_delta_pct(metrics: Dict[str, float]) -> float:
    """Extract heart_delta_pct metric from metrics dictionary.

    Args:
        metrics: Job metrics dictionary

    Returns:
        Heart delta percentage, or 0.0 if not available
    """
    return float(metrics.get("heart_delta_pct", 0.0))


def extract_duration_minutes(duration_s: float | None) -> float:
    """Extract duration in minutes from duration_s.

    Args:
        duration_s: Duration in seconds, or None if not available

    Returns:
        Duration in minutes, or 0.0 if not available
    """
    if duration_s is None:
        return 0.0
    return duration_s / 60.0


def extract_training_metrics(job_result: Dict) -> Dict[str, float]:
    """Extract training metrics from structured job result.

    Args:
        job_result: Dictionary with keys: name, acceptance_passed, exit_code, metrics

    Returns:
        Dictionary with extracted metrics:
        {
            "success": 1.0 or 0.0,
            "hearts": float or 0.0,
            "sps": float or 0.0,
            "shaped": float or 0.0,  # Same as sps for multinode
        }
    """
    metrics = job_result.get("metrics", {})

    return {
        "success": determine_success(job_result.get("acceptance_passed"), job_result.get("exit_code")),
        "hearts": extract_hearts(metrics),
        "sps": extract_sps(metrics),
        "shaped": extract_shaped(metrics),
    }


def extract_eval_metrics(job_result: Dict) -> Dict[str, float]:
    """Extract eval metrics from structured job result.

    Args:
        job_result: Dictionary with keys: name, acceptance_passed, exit_code, metrics, duration_s

    Returns:
        Dictionary with extracted metrics:
        {
            "success": 1.0 or 0.0,
            "heart_delta_pct": float or 0.0,
            "duration_minutes": float or 0.0,
        }
    """
    metrics = job_result.get("metrics", {})

    return {
        "success": determine_success(job_result.get("acceptance_passed"), job_result.get("exit_code")),
        "heart_delta_pct": extract_heart_delta_pct(metrics),
        "duration_minutes": extract_duration_minutes(job_result.get("duration_s")),
    }

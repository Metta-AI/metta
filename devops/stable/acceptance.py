"""Acceptance criteria and metrics extraction for validations.

Defines how to extract metrics from logs and evaluate pass/fail criteria.
"""

import re
from typing import Protocol

from devops.stable.models import Outcome, RunResult, ThresholdCheck


def evaluate_thresholds(
    metrics: dict[str, float], checks: list[ThresholdCheck]
) -> tuple[Outcome, list[ThresholdCheck]]:
    """Evaluate metrics against threshold checks.

    Returns:
        Tuple of (outcome, failed_checks)
    """
    failed = []
    for c in checks:
        # Treat missing metrics as hard failures
        if c.key not in metrics:
            c.actual = None
            c.passed = False
            c.note = "metric missing"
            failed.append(c)
            continue

        actual = metrics[c.key]
        c.actual = actual
        ops = {
            ">=": lambda a, b: a >= b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            "<": lambda a, b: a < b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        ok = ops[c.op](actual, c.expected)
        c.passed = ok
        if not ok:
            c.note = f"expected {c.op} {c.expected}, saw {actual}"
            failed.append(c)
    return (Outcome.PASSED if not failed else Outcome.FAILED, failed)


class MetricsSource(Protocol):
    """Protocol for extracting metrics from logs."""

    def extract(self, run: RunResult, log_text: str) -> dict[str, float]:
        """Extract metrics from log text."""
        ...


class LogRegexMetrics:
    """Extract metrics from logs using regex patterns.

    Supports:
    - SPS (samples per second) - max and last value
    - Eval success rate
    """

    _SPS_RE = re.compile(r"\bSPS[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
    _EVAL_RATE_RE = re.compile(r"\beval[_\s-]?success[_\s-]?rate[:=]\s*(0?\.\d+|1(?:\.0)?)", re.IGNORECASE)

    def extract(self, run: RunResult, log_text: str) -> dict[str, float]:
        """Extract metrics from log text."""
        metrics: dict[str, float] = {}

        # Extract SPS values
        sps_matches = [float(x) for x in self._SPS_RE.findall(log_text)]
        if sps_matches:
            metrics["sps_max"] = max(sps_matches)
            metrics["sps_last"] = sps_matches[-1]

        # Extract eval success rate
        eval_matches = self._EVAL_RATE_RE.findall(log_text)
        if eval_matches:
            metrics["eval_success_rate"] = float(eval_matches[-1])

        return metrics

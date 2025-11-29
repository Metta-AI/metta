from __future__ import annotations

from datetime import datetime, timezone
from statistics import quantiles
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def parse_github_timestamp(value: str) -> datetime:
    """GitHub timestamps are ISO8601 with a trailing Z."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def parse_iso8601(value: str) -> datetime:
    """Parse generic ISO8601 timestamps (accepts trailing Z)."""
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def percentile(values: Sequence[float], percent: float) -> float:
    """Return percentile using statistics.quantiles fallback."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # statistics.quantiles returns cut points; convert percent to n=100 -> index
    qs = quantiles(values, n=100)
    index = min(max(int(percent) - 1, 0), len(qs) - 1)
    return qs[index]


def average(values: Iterable[float]) -> float:
    values_list: List[float] = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)

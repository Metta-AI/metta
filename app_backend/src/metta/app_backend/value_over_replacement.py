from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from pydantic import BaseModel, Field


class RunningStats:
    """Incrementally track weighted running mean/variance statistics.

    Uses West's online algorithm for weighted variance.
    When weight=1 (default), behaves like standard Welford's algorithm.
    """

    def __init__(self) -> None:
        self.count = 0  # Number of samples (episodes)
        self.total_weight = 0.0  # Sum of weights (total agents)
        self._mean = 0.0
        self._m2 = 0.0
        self._min = math.inf
        self._max = -math.inf

    def update(self, value: float, weight: int = 1) -> None:
        """Update stats with a new value and optional weight (e.g., agent count)."""
        self.count += 1
        self.total_weight += weight
        delta = value - self._mean
        self._mean += (weight / self.total_weight) * delta
        delta2 = value - self._mean
        self._m2 += weight * delta * delta2
        if self.count == 1:
            self._min = value
            self._max = value
            return
        self._min = min(self._min, value)
        self._max = max(self._max, value)

    @property
    def mean(self) -> float | None:
        return None if self.total_weight == 0 else self._mean

    @property
    def variance(self) -> float | None:
        if self.total_weight == 0:
            return None
        if self.count == 1:
            return 0.0
        # Bessel's correction for weighted variance
        return self._m2 / (self.total_weight - (self.total_weight / self.count))

    @property
    def std_dev(self) -> float | None:
        variance = self.variance
        return None if variance is None else math.sqrt(max(0, variance))

    @property
    def min(self) -> float | None:
        return None if self.count == 0 else self._min

    @property
    def max(self) -> float | None:
        return None if self.count == 0 else self._max

    def merge(self, other: RunningStats) -> None:
        """Merge another RunningStats into this one using parallel algorithm."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.total_weight = other.total_weight
            self._mean = other._mean
            self._m2 = other._m2
            self._min = other._min
            self._max = other._max
            return
        combined_weight = self.total_weight + other.total_weight
        delta = other._mean - self._mean
        self._mean = (self.total_weight * self._mean + other.total_weight * other._mean) / combined_weight
        self._m2 = self._m2 + other._m2 + delta * delta * self.total_weight * other.total_weight / combined_weight
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)
        self.count += other.count
        self.total_weight = combined_weight


@dataclass
class ScenarioAccumulator:
    candidate_count: int
    thinky_count: int
    ladybug_count: int
    scenario_kind: str
    candidate_stats: RunningStats = field(default_factory=RunningStats)
    replacement_stats: RunningStats = field(default_factory=RunningStats)


class ScenarioSummary(BaseModel):
    scenario_name: str
    scenario_kind: str
    candidate_count: int
    thinky_count: int
    ladybug_count: int
    candidate_mean: float | None
    candidate_std: float | None
    candidate_samples: int
    replacement_mean: float | None = None
    replacement_std: float | None = None
    replacement_samples: int = 0


class CandidateCountSummary(BaseModel):
    candidate_count: int
    mean: float | None
    variance: float | None
    std_dev: float | None
    min_value: float | None
    max_value: float | None
    samples: int  # Number of episodes
    total_agents: int  # Total agent weight (for weighted average)


class GraphPoint(BaseModel):
    candidate_count: int
    mean: float | None
    lower: float | None
    upper: float | None
    std_dev: float | None


class ValueOverReplacementSummary(BaseModel):
    policy_version_id: str
    scenario_summaries: list[ScenarioSummary]
    candidate_counts: list[CandidateCountSummary]
    replacement_summary: CandidateCountSummary | None
    value_over_replacement: dict[str, float | None]
    value_over_replacement_std: dict[str, float | None] = Field(default_factory=dict)
    graph_points: list[GraphPoint]
    # Overall VOR with global normalization (single number for policy comparison)
    overall_vor: float | None = None
    overall_vor_std: float | None = None
    total_candidate_agents: int = 0


def _variance_of_mean(summary: CandidateCountSummary | None) -> float | None:
    if summary is None or summary.total_agents <= 0 or summary.variance is None:
        return None
    return summary.variance / summary.total_agents


def compute_overall_vor_from_stats(
    candidate_count_stats: Mapping[int, RunningStats],
    replacement_baseline: float,
) -> float | None:
    """Compute VOR using a fixed replacement baseline.

    Args:
        candidate_count_stats: Stats per candidate_count level
        replacement_baseline: Hardcoded replacement mean (from REPLACEMENT_BASELINE_MEAN)

    Returns:
        VOR = candidate_avg - replacement_baseline
    """
    total_candidate_agents = 0
    total_candidate_weighted_sum = 0.0

    for candidate_count, stats in candidate_count_stats.items():
        if candidate_count > 0 and stats.mean is not None:
            total_candidate_agents += int(stats.total_weight)
            total_candidate_weighted_sum += stats.mean * stats.total_weight

    if total_candidate_agents <= 0:
        return None

    overall_candidate_avg = total_candidate_weighted_sum / total_candidate_agents
    return overall_candidate_avg - replacement_baseline

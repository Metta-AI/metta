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
    """Compute VOR using a replacement baseline.

    Args:
        candidate_count_stats: Stats per candidate_count level
        replacement_baseline: Mean reward of baseline policies (thinky/ladybug) from c0 episodes

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


def build_value_over_replacement_summary_from_stats(
    *,
    policy_version_id: str,
    scenario_stats: Mapping[str, ScenarioAccumulator],
    candidate_count_stats: Mapping[int, RunningStats],
) -> ValueOverReplacementSummary:
    scenario_summaries: list[ScenarioSummary] = []
    for scenario_name, payload in sorted(scenario_stats.items()):
        candidate_stats = payload.candidate_stats
        replacement_stats = payload.replacement_stats
        candidate_count = payload.candidate_count
        scenario_summaries.append(
            ScenarioSummary(
                scenario_name=scenario_name,
                scenario_kind=payload.scenario_kind,
                candidate_count=candidate_count,
                thinky_count=payload.thinky_count,
                ladybug_count=payload.ladybug_count,
                candidate_mean=candidate_stats.mean,
                candidate_std=candidate_stats.std_dev,
                candidate_samples=candidate_stats.count,
                replacement_mean=replacement_stats.mean if candidate_count == 0 else None,
                replacement_std=replacement_stats.std_dev if candidate_count == 0 else None,
                replacement_samples=replacement_stats.count if candidate_count == 0 else 0,
            )
        )

    candidate_count_summaries: list[CandidateCountSummary] = []
    for candidate_count, stats in sorted(candidate_count_stats.items()):
        candidate_count_summaries.append(
            CandidateCountSummary(
                candidate_count=candidate_count,
                mean=stats.mean,
                variance=stats.variance,
                std_dev=stats.std_dev,
                min_value=stats.min,
                max_value=stats.max,
                samples=stats.count,
                total_agents=int(stats.total_weight),
            )
        )

    replacement_summary = next(
        (summary for summary in candidate_count_summaries if summary.candidate_count == 0),
        None,
    )
    replacement_mean = replacement_summary.mean if replacement_summary else None
    replacement_var_mean = _variance_of_mean(replacement_summary)

    value_over_replacement: dict[str, float | None] = {}
    value_over_replacement_std: dict[str, float | None] = {}
    graph_points: list[GraphPoint] = []

    for summary in candidate_count_summaries:
        candidate_count = summary.candidate_count
        candidate_mean = summary.mean
        candidate_var_mean = _variance_of_mean(summary)

        if candidate_count == 0:
            vor_mean = 0.0
            vor_std = 0.0
        else:
            if replacement_mean is None or candidate_mean is None:
                vor_mean = None
            else:
                vor_mean = candidate_mean - replacement_mean

            if (
                candidate_var_mean is None
                or replacement_var_mean is None
                or summary.samples <= 0
                or (replacement_summary and replacement_summary.samples <= 0)
            ):
                vor_std = None
            else:
                vor_std = math.sqrt(candidate_var_mean + replacement_var_mean)

        # Add to dictionaries for all candidate_counts (including 0)
        key = str(candidate_count)
        value_over_replacement[key] = vor_mean
        value_over_replacement_std[key] = vor_std

        if vor_mean is None or vor_std is None:
            graph_points.append(
                GraphPoint(
                    candidate_count=candidate_count,
                    mean=vor_mean,
                    lower=None,
                    upper=None,
                    std_dev=vor_std,
                )
            )
        else:
            graph_points.append(
                GraphPoint(
                    candidate_count=candidate_count,
                    mean=vor_mean,
                    lower=vor_mean - vor_std,
                    upper=vor_mean + vor_std,
                    std_dev=vor_std,
                )
            )

    # Compute overall VOR with global normalization
    # Sum weighted rewards across ALL candidate_count levels, divide by total agents
    total_candidate_agents = sum(s.total_agents for s in candidate_count_summaries if s.candidate_count > 0)
    total_candidate_weighted_sum = sum(
        (s.mean or 0) * s.total_agents
        for s in candidate_count_summaries
        if s.candidate_count > 0 and s.mean is not None
    )

    if total_candidate_agents > 0 and replacement_mean is not None:
        overall_candidate_avg = total_candidate_weighted_sum / total_candidate_agents
        overall_vor = overall_candidate_avg - replacement_mean

        # Approximate overall std using pooled variance
        if replacement_var_mean is not None:
            total_candidate_var = sum(
                (s.variance or 0) * s.total_agents
                for s in candidate_count_summaries
                if s.candidate_count > 0 and s.variance is not None
            )
            if total_candidate_agents > 0:
                pooled_candidate_var = total_candidate_var / total_candidate_agents
                overall_vor_std = math.sqrt(pooled_candidate_var / total_candidate_agents + replacement_var_mean)
            else:
                overall_vor_std = None
        else:
            overall_vor_std = None
    else:
        overall_vor = None
        overall_vor_std = None

    return ValueOverReplacementSummary(
        policy_version_id=policy_version_id,
        scenario_summaries=scenario_summaries,
        candidate_counts=candidate_count_summaries,
        replacement_summary=replacement_summary,
        value_over_replacement=value_over_replacement,
        value_over_replacement_std=value_over_replacement_std,
        graph_points=graph_points,
        overall_vor=overall_vor,
        overall_vor_std=overall_vor_std,
        total_candidate_agents=total_candidate_agents,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
from uuid import UUID

import numpy as np

from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult


class ScoredMatchLike(Protocol):
    assignments: list[int]
    policy_version_ids: list[UUID]
    policy_scores: dict[UUID, float]


def compute_weighted_scores(
    policy_version_ids: Sequence[UUID],
    matches: Sequence[ScoredMatchLike],
) -> dict[UUID, float]:
    weighted_sums: dict[UUID, float] = {pv: 0.0 for pv in policy_version_ids}
    weight_totals: dict[UUID, float] = {pv: 0.0 for pv in policy_version_ids}

    for match in matches:
        total_agents = len(match.assignments)
        if total_agents == 0:
            continue

        policy_agent_counts: dict[UUID, int] = {}
        for policy_idx in match.assignments:
            if policy_idx < len(match.policy_version_ids):
                pv = match.policy_version_ids[policy_idx]
                policy_agent_counts[pv] = policy_agent_counts.get(pv, 0) + 1

        for pv, score in match.policy_scores.items():
            if pv not in weighted_sums:
                continue
            agent_count = policy_agent_counts.get(pv, 0)
            weight = agent_count / total_agents
            weighted_sums[pv] += score * weight
            weight_totals[pv] += weight

    return {pv: weighted_sums[pv] / weight_totals[pv] if weight_totals[pv] > 0 else 0.0 for pv in policy_version_ids}


def value_over_replacement(candidate_score: float, replacement_score: float) -> float:
    return candidate_score - replacement_score


def overall_value_over_replacement(weighted_sum: float, total_agents: int, replacement_score: float) -> float | None:
    if total_agents <= 0:
        return None
    return weighted_sum / total_agents - replacement_score


@dataclass
class VorScenarioSummary:
    candidate_mean: float | None
    replacement_mean: float | None
    candidate_episode_count: int


@dataclass
class VorTotals:
    replacement_mean: float | None = None
    total_candidate_weighted_sum: float = 0.0
    total_candidate_agents: int = 0

    def update(self, candidate_count: int, summary: VorScenarioSummary) -> None:
        if candidate_count == 0:
            self.replacement_mean = summary.replacement_mean
            return
        if summary.candidate_mean is None or summary.candidate_episode_count == 0:
            return
        self.total_candidate_weighted_sum += summary.candidate_mean * candidate_count * summary.candidate_episode_count
        self.total_candidate_agents += candidate_count * summary.candidate_episode_count


def summarize_vor_scenario(
    rollout: MultiEpisodeRolloutResult,
    *,
    candidate_policy_index: int,
    candidate_count: int,
) -> VorScenarioSummary:
    candidate_sum = 0.0
    candidate_episode_count = 0
    replacement_sum = 0.0
    replacement_episode_count = 0

    for episode in rollout.episodes:
        if episode.rewards.size == 0:
            continue
        if candidate_count == 0:
            replacement_sum += float(episode.rewards.mean())
            replacement_episode_count += 1
        else:
            mask = episode.assignments == candidate_policy_index
            if np.any(mask):
                candidate_sum += float(episode.rewards[mask].mean())
                candidate_episode_count += 1

    candidate_mean = candidate_sum / candidate_episode_count if candidate_episode_count else None
    replacement_mean = replacement_sum / replacement_episode_count if replacement_episode_count else None

    return VorScenarioSummary(
        candidate_mean=candidate_mean,
        replacement_mean=replacement_mean,
        candidate_episode_count=candidate_episode_count,
    )

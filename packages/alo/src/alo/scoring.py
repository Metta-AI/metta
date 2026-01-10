from __future__ import annotations

from typing import Mapping, Protocol, Sequence
from uuid import UUID


class ScoredMatchLike(Protocol):
    assignments: Sequence[int]
    policy_version_ids: Sequence[UUID]
    policy_scores: Mapping[UUID, float]


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

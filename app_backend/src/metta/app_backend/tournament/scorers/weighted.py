"""Weighted scorer for computing value-over-replacement scores."""

from uuid import UUID

from metta.app_backend.tournament.interfaces import ScoredMatchData, ScorerInterface


class WeightedScorer(ScorerInterface):
    """Computes policy scores weighted by agent assignment counts.

    For each match, a policy's contribution is weighted by the fraction of
    agents it controlled. This allows fair comparison between policies that
    played in different configurations (e.g., 1v3 vs 2v2).
    """

    def compute_scores(
        self,
        policy_version_ids: list[UUID],
        matches: list[ScoredMatchData],
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

        return {
            pv: weighted_sums[pv] / weight_totals[pv] if weight_totals[pv] > 0 else 0.0 for pv in policy_version_ids
        }

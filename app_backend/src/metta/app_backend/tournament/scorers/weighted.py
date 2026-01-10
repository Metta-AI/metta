"""Weighted scorer for computing value-over-replacement scores."""

from uuid import UUID

from alo.scoring import compute_weighted_scores

from metta.app_backend.tournament.referees.base import ScoredMatchData, ScorerInterface


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
        return compute_weighted_scores(policy_version_ids, matches)

from collections import defaultdict
from uuid import UUID

from metta.app_backend.tournament.interfaces import (
    Match,
    MatchPlayer,
    PolicyVersion,
    Pool,
    PoolPlayer,
    Season,
    SeasonScorerInterface,
)


class DefaultSeasonScorer(SeasonScorerInterface):
    def get_pool_rankings(
        self,
        pool: Pool,
        pool_players: list[PoolPlayer],
        match_history: list[Match],
        match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[tuple[PolicyVersion, float]]:
        scores: dict[UUID, list[float]] = defaultdict(list)

        for match in match_history:
            if match.status != "completed":
                continue
            players = match_players.get(match.id, [])
            for player in players:
                if player.score is not None:
                    scores[player.policy_version_id].append(player.score)

        rankings: list[tuple[UUID, float]] = []
        for pv_id, score_list in scores.items():
            if score_list:
                avg_score = sum(score_list) / len(score_list)
                rankings.append((pv_id, avg_score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return []

    def get_season_standings(
        self,
        season: Season,
        pools: list[Pool],
        all_pool_players: list[PoolPlayer],
        all_matches: list[Match],
        all_match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[tuple[PolicyVersion, float]]:
        return []

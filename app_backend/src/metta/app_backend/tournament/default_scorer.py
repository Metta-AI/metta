from metta.app_backend.tournament.interfaces import (
    MatchWithEvalStatus,
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
        match_history: list[MatchWithEvalStatus],
    ) -> list[tuple[PolicyVersion, float]]:
        return []

    def get_season_standings(
        self,
        season: Season,
        pools: list[Pool],
        all_pool_players: list[PoolPlayer],
        all_matches: list[MatchWithEvalStatus],
    ) -> list[tuple[PolicyVersion, float]]:
        return []

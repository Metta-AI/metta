import random
from uuid import UUID

from metta.app_backend.tournament.interfaces import (
    MatchRequest,
    MatchWithEvalStatus,
    Pool,
    PoolPlayer,
    RefereeInterface,
)


class DefaultReferee(RefereeInterface):
    def __init__(
        self,
        min_matches_per_player: int = 5,
        players_per_match: int = 2,
    ):
        self.min_matches_per_player = min_matches_per_player
        self.players_per_match = players_per_match

    def get_desired_matches(
        self,
        pool: Pool,
        pool_players: list[PoolPlayer],
        match_history: list[MatchWithEvalStatus],
    ) -> list[MatchRequest]:
        if len(pool_players) < self.players_per_match:
            return []

        completed_match_counts: dict[UUID, int] = {pp.policy_version_id: 0 for pp in pool_players}
        pending_match_counts: dict[UUID, int] = {pp.policy_version_id: 0 for pp in pool_players}

        for match in match_history:
            pv_ids = [p.policy_version_id for p in match.players]
            if match.is_completed:
                for pv_id in pv_ids:
                    if pv_id in completed_match_counts:
                        completed_match_counts[pv_id] += 1
            elif match.is_pending:
                for pv_id in pv_ids:
                    if pv_id in pending_match_counts:
                        pending_match_counts[pv_id] += 1

        active_pv_ids = [pp.policy_version_id for pp in pool_players if not pp.retired]
        needs_matches = [
            pv_id
            for pv_id in active_pv_ids
            if (completed_match_counts.get(pv_id, 0) + pending_match_counts.get(pv_id, 0)) < self.min_matches_per_player
        ]

        if not needs_matches:
            return []

        requests: list[MatchRequest] = []
        for pv_id in needs_matches:
            opponents = [other for other in active_pv_ids if other != pv_id]
            if len(opponents) < self.players_per_match - 1:
                continue

            selected = random.sample(opponents, self.players_per_match - 1)
            participants = [pv_id] + selected

            requests.append(MatchRequest(policy_version_ids=participants))

        return requests

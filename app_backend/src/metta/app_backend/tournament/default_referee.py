import random
from uuid import UUID

from metta.app_backend.tournament.interfaces import (
    Match,
    MatchPlayer,
    MatchRequest,
    Pool,
    PoolPlayer,
    RefereeInterface,
)


class DefaultReferee(RefereeInterface):
    def __init__(
        self,
        min_matches_per_player: int = 5,
        players_per_match: int = 2,
        environment_config: dict | None = None,
    ):
        self.min_matches_per_player = min_matches_per_player
        self.players_per_match = players_per_match
        self.environment_config = environment_config or {}

    def get_desired_matches(
        self,
        pool: Pool,
        pool_players: list[PoolPlayer],
        match_history: list[Match],
        match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[MatchRequest]:
        if len(pool_players) < self.players_per_match:
            return []

        match_counts: dict[UUID, int] = {pp.policy_version_id: 0 for pp in pool_players}
        for match in match_history:
            players = match_players.get(match.id, [])
            for player in players:
                if player.policy_version_id in match_counts:
                    match_counts[player.policy_version_id] += 1

        active_pv_ids = [pp.policy_version_id for pp in pool_players if not pp.retired]
        needs_matches = [pv_id for pv_id in active_pv_ids if match_counts.get(pv_id, 0) < self.min_matches_per_player]

        if not needs_matches:
            return []

        requests: list[MatchRequest] = []
        for pv_id in needs_matches:
            opponents = [other for other in active_pv_ids if other != pv_id]
            if len(opponents) < self.players_per_match - 1:
                continue

            selected = random.sample(opponents, self.players_per_match - 1)
            participants = [pv_id] + selected

            requests.append(
                MatchRequest(
                    environment_config=self.environment_config,
                    policy_version_ids=participants,
                )
            )

        return requests


class AcademyReferee(RefereeInterface):
    def __init__(
        self,
        bootstrap_self_play_count: int = 3,
        bootstrap_random_opponent_count: int = 2,
        environment_config: dict | None = None,
    ):
        self.bootstrap_self_play_count = bootstrap_self_play_count
        self.bootstrap_random_opponent_count = bootstrap_random_opponent_count
        self.environment_config = environment_config or {}

    def get_desired_matches(
        self,
        pool: Pool,
        pool_players: list[PoolPlayer],
        match_history: list[Match],
        match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[MatchRequest]:
        match_counts: dict[UUID, int] = {pp.policy_version_id: 0 for pp in pool_players}
        for match in match_history:
            players = match_players.get(match.id, [])
            for player in players:
                if player.policy_version_id in match_counts:
                    match_counts[player.policy_version_id] += 1

        active_pv_ids = [pp.policy_version_id for pp in pool_players if not pp.retired]
        new_players = [pv_id for pv_id in active_pv_ids if match_counts.get(pv_id, 0) == 0]

        requests: list[MatchRequest] = []
        for pv_id in new_players:
            for _ in range(self.bootstrap_self_play_count):
                requests.append(
                    MatchRequest(
                        environment_config=self.environment_config,
                        policy_version_ids=[pv_id, pv_id],
                    )
                )

            others = [other for other in active_pv_ids if other != pv_id]
            if others:
                num_opponents = min(self.bootstrap_random_opponent_count, len(others))
                selected = random.sample(others, num_opponents)
                for opponent in selected:
                    requests.append(
                        MatchRequest(
                            environment_config=self.environment_config,
                            policy_version_ids=[pv_id, opponent],
                        )
                    )

        return requests

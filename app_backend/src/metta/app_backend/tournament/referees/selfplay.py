"""Self-play referee for qualifying pools."""

from uuid import UUID

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.app_backend.models.tournament import PoolPlayer
from metta.app_backend.tournament.interfaces import MatchData, MatchRequest, RefereeInterface
from mettagrid.config.mettagrid_config import MettaGridConfig

NUM_AGENTS = 4


def _make_env() -> MettaGridConfig:
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = NUM_AGENTS
    return mission.make_env()


class SelfPlayReferee(RefereeInterface):
    """Schedules self-play matches where a single policy controls all agents.

    Each player gets up to `matches_per_player` self-play matches to establish
    a baseline score before being considered for promotion to competition pools.
    """

    matches_per_player: int = 1

    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        match_counts: dict[UUID, int] = {p.policy_version_id: 0 for p in players}

        for md in matches:
            pv_set = set(md.player_pv_ids)
            if len(pv_set) == 1 and md.player_pv_ids:
                pv = md.player_pv_ids[0]
                if pv in match_counts:
                    match_counts[pv] += 1

        requests: list[MatchRequest] = []
        for player in players:
            pv = player.policy_version_id
            needed = self.matches_per_player - match_counts.get(pv, 0)
            for _ in range(needed):
                requests.append(
                    MatchRequest(
                        policy_version_ids=[pv],
                        assignments=[0, 0, 0, 0],
                        env=_make_env(),
                        episode_tags={"match_type": "self_play"},
                    )
                )

        return requests

from typing import Tuple
from uuid import UUID

from cogames.cogs_vs_clips.missions import Machina1OpenWorldSharedRewardsMission
from metta.app_backend.models.tournament import PoolPlayer
from metta.app_backend.tournament.referees.base import MatchData, MatchRequest, RefereeBase, ScorerInterface
from metta.app_backend.tournament.scorers.weighted import WeightedScorer
from mettagrid.config.mettagrid_config import MettaGridConfig

NUM_AGENTS = 4


def _make_env() -> MettaGridConfig:
    mission = Machina1OpenWorldSharedRewardsMission.model_copy(deep=True)
    mission.num_cogs = NUM_AGENTS
    return mission.make_env()


class PairingReferee(RefereeBase):
    """Schedules matches between pairs of policies to find optimal teammates.

    Each unique pair of policies plays up to `matches_per_pair` matches together,
    with agents split evenly between the two policies (2v2 configuration).
    """

    scorer: ScorerInterface = WeightedScorer()
    matches_per_pair: int = 3

    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        pair_counts: dict[Tuple[UUID, UUID], int] = {}

        for md in matches:
            pv_set = set(md.player_pv_ids)
            if len(pv_set) == 2:
                pv_list = sorted(pv_set)
                pair: Tuple[UUID, UUID] = (pv_list[0], pv_list[1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        requests: list[MatchRequest] = []
        player_pvs = [p.policy_version_id for p in players]

        for i, pv1 in enumerate(player_pvs):
            for pv2 in player_pvs[i + 1 :]:
                pv_list = sorted([pv1, pv2])
                pair = (pv_list[0], pv_list[1])
                needed = self.matches_per_pair - pair_counts.get(pair, 0)
                for _ in range(needed):
                    requests.append(
                        MatchRequest(
                            policy_version_ids=[pv1, pv2],
                            assignments=[0, 0, 1, 1],
                            env=_make_env(),
                            episode_tags={"match_type": "pairing"},
                        )
                    )

        return requests

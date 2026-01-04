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


MATCH_CONFIGURATIONS: list[list[int]] = [
    [0, 1, 1, 1],  # 1v3
    [0, 0, 0, 1],  # 3v1
    [0, 0, 1, 1],  # 2v2
]


class PairingReferee(RefereeBase):
    """Schedules matches between pairs of policies with varying agent splits.

    Each unique pair of policies plays three matches with different configurations:
    1v3, 3v1, and 2v2. This tests whether policies can cooperate effectively
    regardless of team size.
    """

    scorer: ScorerInterface = WeightedScorer()

    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        existing_configs: set[Tuple[UUID, UUID, tuple[int, ...]]] = set()

        for md in matches:
            pv_set = set(md.player_pv_ids)
            if len(pv_set) == 2 and md.assignments:
                pv_list = sorted(pv_set)
                pair: Tuple[UUID, UUID] = (pv_list[0], pv_list[1])
                existing_configs.add((pair[0], pair[1], tuple(md.assignments)))

        requests: list[MatchRequest] = []
        player_pvs = [p.policy_version_id for p in players]

        for i, pv1 in enumerate(player_pvs):
            for pv2 in player_pvs[i + 1 :]:
                pv_list = sorted([pv1, pv2])
                pair = (pv_list[0], pv_list[1])

                for config in MATCH_CONFIGURATIONS:
                    key = (pair[0], pair[1], tuple(config))
                    if key not in existing_configs:
                        requests.append(
                            MatchRequest(
                                policy_version_ids=[pv1, pv2],
                                assignments=config,
                                env=_make_env(),
                                episode_tags={
                                    "match_type": "pairing",
                                    "assignments": str(config),
                                    "env": "machina1_open_world",
                                },
                            )
                        )

        return requests

from collections import defaultdict
from typing import Tuple
from uuid import UUID

from cogames.cogs_vs_clips.missions import Machina1OpenWorldSharedRewardsMission, MettaGridConfig
from metta.app_backend.models.tournament import PoolPlayer
from metta.app_backend.tournament.referees.base import MatchData, MatchRequest, RefereeBase, ScorerInterface
from metta.app_backend.tournament.scorers.weighted import WeightedScorer

NUM_AGENTS = 4


def _make_env(seed: int) -> MettaGridConfig:
    mission = Machina1OpenWorldSharedRewardsMission.model_copy(deep=True)
    mission.num_cogs = NUM_AGENTS
    env = mission.make_env()
    env.game.map_builder.seed = seed  # type: ignore
    return env


MATCH_CONFIGURATIONS: list[list[int]] = [
    [0, 1, 1, 1],  # 1v3
    [0, 0, 0, 1],  # 3v1
    [0, 0, 1, 1],  # 2v2
]


class PairingReferee(RefereeBase):
    """Schedules matches between pairs of policies with varying agent splits.

    Each unique pair of policies plays multiple matches with different configurations
    (1+3, 3+1, 2+2 agent splits). Matches are scheduled to maximize coverage first
    (one of each config per pair) before adding replications for statistical confidence.
    Multiple configurations enable value-over-replacement calculation via participation-weighted scoring.
    """

    scorer: ScorerInterface = WeightedScorer()
    matches_per_config: int = 5
    description: str = (
        "Pairwise matchups on Machina 1 Open World with varied agent splits (1+3, 3+1, 2+2) "
        "and shared rewards; scored by participation-weighted average"
    )

    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        config_counts: dict[Tuple[UUID, UUID, tuple[int, ...]], int] = defaultdict(int)

        for md in matches:
            pp_set = set(md.pool_player_ids)
            if len(pp_set) == 2 and md.assignments:
                pp_list = sorted(pp_set)
                pair: Tuple[UUID, UUID] = (pp_list[0], pp_list[1])
                config_counts[(pair[0], pair[1], tuple(md.assignments))] += 1

        pending: list[Tuple[int, UUID, UUID, list[int], int]] = []
        player_ids = [p.id for p in players]

        for i, pp1 in enumerate(player_ids):
            for pp2 in player_ids[i + 1 :]:
                pp_list = sorted([pp1, pp2])
                pair = (pp_list[0], pp_list[1])

                for c_idx, config in enumerate(MATCH_CONFIGURATIONS):
                    key = (pair[0], pair[1], tuple(config))
                    existing = config_counts[key]
                    needed = self.matches_per_config - existing
                    for match_i in range(needed):
                        pending.append((existing, pp1, pp2, config, c_idx + match_i + existing))
                        existing += 1

        pending.sort(key=lambda x: x[0])
        # Use relative_match_num so each pair plays on the same sequence of maps
        seed = 42
        return [
            MatchRequest(
                pool_player_ids=[pp1, pp2],
                assignments=config,
                env=_make_env(seed + relative_match_num),
                episode_tags={
                    "match_type": "pairing",
                    "assignments": str(config),
                },
                seed=seed,
            )
            for _, pp1, pp2, config, relative_match_num in pending
        ]

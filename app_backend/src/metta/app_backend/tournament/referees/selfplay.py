from uuid import UUID

from cogames.cogs_vs_clips.missions import Machina1OpenWorldSharedRewardsMission
from metta.app_backend.models.tournament import MatchStatus, PoolPlayer
from metta.app_backend.tournament.referees.base import MatchData, MatchRequest, RefereeBase, ScorerInterface
from metta.app_backend.tournament.scorers.weighted import WeightedScorer
from mettagrid.config.mettagrid_config import MettaGridConfig

NUM_AGENTS = 4
MAX_FAILED_ATTEMPTS = 5


def _make_env() -> MettaGridConfig:
    mission = Machina1OpenWorldSharedRewardsMission.model_copy(deep=True)
    mission.num_cogs = NUM_AGENTS
    return mission.make_env()


class SelfPlayReferee(RefereeBase):
    """Schedules self-play matches where a single policy controls all agents.

    Each player gets up to `matches_per_player` self-play matches to establish
    a baseline score before being considered for promotion to competition pools.
    Retries failed matches up to MAX_FAILED_ATTEMPTS times.
    """

    scorer: ScorerInterface = WeightedScorer()
    matches_per_player: int = 1

    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        completed_counts: dict[UUID, int] = {p.policy_version_id: 0 for p in players}
        failed_counts: dict[UUID, int] = {p.policy_version_id: 0 for p in players}
        in_progress_counts: dict[UUID, int] = {p.policy_version_id: 0 for p in players}

        for md in matches:
            pv_set = set(md.player_pv_ids)
            if len(pv_set) == 1 and md.player_pv_ids:
                pv = md.player_pv_ids[0]
                if pv in completed_counts:
                    if md.status == MatchStatus.completed:
                        completed_counts[pv] += 1
                    elif md.status == MatchStatus.failed:
                        failed_counts[pv] += 1
                    elif md.status in (MatchStatus.pending, MatchStatus.scheduled, MatchStatus.running):
                        in_progress_counts[pv] += 1

        requests: list[MatchRequest] = []
        for player in players:
            pv = player.policy_version_id
            completed = completed_counts.get(pv, 0)
            failed = failed_counts.get(pv, 0)
            in_progress = in_progress_counts.get(pv, 0)

            if failed >= MAX_FAILED_ATTEMPTS:
                continue
            if in_progress > 0:
                continue

            needed = self.matches_per_player - completed
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

    def should_retire_policy(self, matches: list[MatchData], policy_version_id: UUID) -> bool:
        """Returns True if policy has exhausted retry attempts without completing required matches."""
        completed = 0
        failed = 0
        for md in matches:
            if len(set(md.player_pv_ids)) == 1 and md.player_pv_ids and md.player_pv_ids[0] == policy_version_id:
                if md.status == MatchStatus.completed:
                    completed += 1
                elif md.status == MatchStatus.failed:
                    failed += 1
        return failed >= MAX_FAILED_ATTEMPTS and completed < self.matches_per_player

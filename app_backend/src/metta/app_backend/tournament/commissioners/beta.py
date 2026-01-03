import logging
from uuid import UUID

from sqlmodel import col, func, select

from metta.app_backend.database import get_db
from metta.app_backend.models.tournament import Match, MatchPlayer, MatchStatus, Pool, PoolPlayer
from metta.app_backend.tournament.interfaces import CommissionerInterface, MembershipChange, ScorerInterface
from metta.app_backend.tournament.referees.pairing import PairingReferee
from metta.app_backend.tournament.referees.selfplay import SelfPlayReferee
from metta.app_backend.tournament.scorers.weighted import WeightedScorer
from metta.app_backend.tournament.settings import PROMOTION_MIN_MATCHES, PROMOTION_MIN_SCORE

logger = logging.getLogger(__name__)


class BetaCommissioner(CommissionerInterface):
    scorer: ScorerInterface = WeightedScorer()
    season_name = "beta"
    referees = {
        "qualifying": SelfPlayReferee(),
        "competition": PairingReferee(),
    }

    def get_new_submission_membership_changes(self, policy_version_id: UUID) -> list[MembershipChange]:
        return [MembershipChange(pool_name="qualifying", policy_version_id=policy_version_id, action="add")]

    async def get_membership_changes(self, pools: dict[str, Pool]) -> list[MembershipChange]:
        if "qualifying" not in pools or "competition" not in pools:
            return []

        qualifying_pool = pools["qualifying"]
        competition_pool = pools["competition"]

        stats = await self._get_player_stats(qualifying_pool.id)
        existing_in_competition = await self._get_pool_member_ids(competition_pool.id)

        changes: list[MembershipChange] = []
        for pv_id, (avg_score, match_count) in stats.items():
            if match_count < PROMOTION_MIN_MATCHES:
                continue
            if avg_score is None or avg_score < PROMOTION_MIN_SCORE:
                continue
            if pv_id not in existing_in_competition:
                changes.append(MembershipChange(pool_name="competition", policy_version_id=pv_id, action="add"))

        return changes

    async def _get_player_stats(self, pool_id: UUID) -> dict[UUID, tuple[float | None, int]]:
        session = get_db()
        return {
            row[0]: (float(row[1]) if row[1] else None, row[2] or 0)
            for row in (
                await session.execute(
                    select(MatchPlayer.policy_version_id, func.avg(MatchPlayer.score), func.count())
                    .join(Match, MatchPlayer.match_id == Match.id)  # type: ignore[arg-type]
                    .where(Match.pool_id == pool_id)
                    .where(Match.status == MatchStatus.completed)
                    .where(col(MatchPlayer.score).is_not(None))
                    .group_by(col(MatchPlayer.policy_version_id))
                )
            ).all()
        }

    async def _get_pool_member_ids(self, pool_id: UUID) -> set[UUID]:
        session = get_db()
        return set(
            (await session.execute(select(PoolPlayer.policy_version_id).filter_by(pool_id=pool_id, retired=False)))
            .scalars()
            .all()
        )

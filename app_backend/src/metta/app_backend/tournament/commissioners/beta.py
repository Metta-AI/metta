import logging
from uuid import UUID

from sqlmodel import col, func, select

# pyright: reportArgumentType=false
from metta.app_backend.database import get_db
from metta.app_backend.models.tournament import Match, MatchPlayer, MatchStatus, Pool, PoolPlayer
from metta.app_backend.tournament.commissioners.base import CommissionerBase, MembershipChangeRequest
from metta.app_backend.tournament.referees.pairing import PairingReferee
from metta.app_backend.tournament.referees.selfplay import MAX_FAILED_ATTEMPTS, SelfPlayReferee
from metta.app_backend.tournament.settings import PROMOTION_MIN_SCORE

logger = logging.getLogger(__name__)


class BetaCommissioner(CommissionerBase):
    season_name = "beta"
    leaderboard_pool = "competition"
    referees = {
        "qualifying": SelfPlayReferee(),
        "competition": PairingReferee(),
    }
    summary = "Policies start in qualifying; promoted to competition if score meets threshold"

    def get_new_submission_membership_changes(self, policy_version_id: UUID) -> list[MembershipChangeRequest]:
        return [
            MembershipChangeRequest(
                pool_name="qualifying", policy_version_id=policy_version_id, action="add", notes="User submission"
            )
        ]

    async def get_membership_changes(self, pools: dict[str, Pool]) -> list[MembershipChangeRequest]:
        if "qualifying" not in pools or "competition" not in pools:
            return []

        qualifying_pool = pools["qualifying"]
        competition_pool = pools["competition"]
        qualifying_referee: SelfPlayReferee = self.referees["qualifying"]  # type: ignore[assignment]
        matches_required = qualifying_referee.matches_per_player

        stats = await self._get_qualifying_stats(qualifying_pool.id)
        existing_in_competition = competition_pool.active_member_ids

        changes: list[MembershipChangeRequest] = []
        for player in qualifying_pool.active_players:
            pv_id = player.policy_version_id
            avg_score, completed, failed = stats.get(player.id, (None, 0, 0))

            if completed < matches_required:
                if failed >= MAX_FAILED_ATTEMPTS:
                    changes.append(
                        MembershipChangeRequest(
                            pool_name="qualifying",
                            policy_version_id=pv_id,
                            action="remove",
                            notes=f"Exhausted retry attempts ({MAX_FAILED_ATTEMPTS} failures)",
                        )
                    )
                    logger.info(f"Retiring {pv_id}: exhausted {MAX_FAILED_ATTEMPTS} retries")
                continue

            if avg_score is not None and avg_score >= PROMOTION_MIN_SCORE:
                if pv_id not in existing_in_competition:
                    changes.append(
                        MembershipChangeRequest(
                            pool_name="competition",
                            policy_version_id=pv_id,
                            action="add",
                            notes=f"Promoted: avg_score={avg_score:.3f} >= {PROMOTION_MIN_SCORE}",
                        )
                    )
                changes.append(
                    MembershipChangeRequest(
                        pool_name="qualifying",
                        policy_version_id=pv_id,
                        action="remove",
                        notes=f"Graduated to competition: avg_score={avg_score:.3f}",
                    )
                )
                logger.info(f"Graduating {pv_id}: avg_score={avg_score:.3f}")
            else:
                avg_str = f"{avg_score:.3f}" if avg_score is not None else "None"
                changes.append(
                    MembershipChangeRequest(
                        pool_name="qualifying",
                        policy_version_id=pv_id,
                        action="remove",
                        notes=f"Score below threshold: avg_score={avg_str} < {PROMOTION_MIN_SCORE}",
                    )
                )
                logger.info(f"Retiring {pv_id}: avg_score={avg_str} < {PROMOTION_MIN_SCORE}")

        return changes

    async def _get_qualifying_stats(self, pool_id: UUID) -> dict[UUID, tuple[float | None, int, int]]:
        """Get (avg_score, completed_count, failed_count) for each pool_player in pool."""
        session = get_db()

        is_completed = col(Match.status) == MatchStatus.completed
        is_failed = col(Match.status) == MatchStatus.failed
        result = await session.execute(
            select(
                col(MatchPlayer.pool_player_id).label("pp_id"),
                func.avg(MatchPlayer.score).filter(is_completed).label("avg_score"),
                func.count().filter(is_completed, col(MatchPlayer.score).is_not(None)).label("completed"),
                func.count().filter(is_failed).label("failed"),
            )
            .join(MatchPlayer.match)
            .join(MatchPlayer.pool_player)
            .where(PoolPlayer.pool_id == pool_id)
            .where(col(Match.status).in_([MatchStatus.completed, MatchStatus.failed]))
            .group_by(col(MatchPlayer.pool_player_id))
        )
        return {
            row.pp_id: (
                float(row.avg_score) if row.avg_score is not None else None,
                row.completed or 0,
                row.failed or 0,
            )
            for row in result.all()
        }

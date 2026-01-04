import logging
from uuid import UUID

from sqlmodel import col, func, select

from metta.app_backend.database import get_db
from metta.app_backend.models.tournament import Match, MatchPlayer, MatchStatus, Pool, PoolPlayer
from metta.app_backend.tournament.commissioners.base import CommissionerBase, MembershipChange
from metta.app_backend.tournament.referees.base import MatchData
from metta.app_backend.tournament.referees.pairing import PairingReferee
from metta.app_backend.tournament.referees.selfplay import SelfPlayReferee
from metta.app_backend.tournament.settings import PROMOTION_MIN_SCORE

logger = logging.getLogger(__name__)


class BetaCommissioner(CommissionerBase):
    season_name = "beta"
    leaderboard_pool = "competition"
    referees = {
        "qualifying": SelfPlayReferee(),
        "competition": PairingReferee(),
    }

    def get_new_submission_membership_changes(self, policy_version_id: UUID) -> list[MembershipChange]:
        return [
            MembershipChange(
                pool_name="qualifying", policy_version_id=policy_version_id, action="add", notes="User submission"
            )
        ]

    async def get_membership_changes(self, pools: dict[str, Pool]) -> list[MembershipChange]:
        if "qualifying" not in pools or "competition" not in pools:
            return []

        qualifying_pool = pools["qualifying"]
        competition_pool = pools["competition"]
        qualifying_referee: SelfPlayReferee = self.referees["qualifying"]  # type: ignore[assignment]

        stats = await self._get_player_stats(qualifying_pool.id)
        existing_in_competition = await self._get_pool_member_ids(competition_pool.id)
        qualifying_players = await self._get_pool_players(qualifying_pool.id)
        qualifying_matches = await self._get_pool_matches_data(qualifying_pool.id)

        changes: list[MembershipChange] = []
        already_handled: set[UUID] = set()
        active_qualifying_ids = {p.policy_version_id for p in qualifying_players}

        for pv_id, (avg_score, match_count) in stats.items():
            if pv_id not in active_qualifying_ids:
                continue
            if match_count < qualifying_referee.matches_per_player:
                continue

            if avg_score is not None and avg_score >= PROMOTION_MIN_SCORE:
                if pv_id not in existing_in_competition:
                    changes.append(
                        MembershipChange(
                            pool_name="competition",
                            policy_version_id=pv_id,
                            action="add",
                            notes=f"Promoted: avg_score={avg_score:.3f} >= {PROMOTION_MIN_SCORE}",
                        )
                    )
            else:
                avg_str = f"{avg_score:.3f}" if avg_score is not None else "None"
                changes.append(
                    MembershipChange(
                        pool_name="qualifying",
                        policy_version_id=pv_id,
                        action="retire",
                        notes=f"Score below threshold: avg_score={avg_str} < {PROMOTION_MIN_SCORE}",
                    )
                )
                logger.info(f"Retiring {pv_id} from qualifying: avg_score={avg_score}, threshold={PROMOTION_MIN_SCORE}")
            already_handled.add(pv_id)

        for player in qualifying_players:
            pv_id = player.policy_version_id
            if pv_id in already_handled:
                continue
            if reason := qualifying_referee.should_retire_reason(qualifying_matches, pv_id):
                changes.append(
                    MembershipChange(
                        pool_name="qualifying",
                        policy_version_id=pv_id,
                        action="retire",
                        notes=reason,
                    )
                )
                logger.info(f"Retiring {pv_id} from qualifying: {reason}")

        return changes

    async def _get_pool_players(self, pool_id: UUID) -> list[PoolPlayer]:
        session = get_db()
        return list(
            (await session.execute(select(PoolPlayer).filter_by(pool_id=pool_id, retired=False))).scalars().all()
        )

    async def _get_pool_matches_data(self, pool_id: UUID) -> list[MatchData]:
        session = get_db()
        matches = list((await session.execute(select(Match).filter_by(pool_id=pool_id))).scalars().all())
        match_ids = [m.id for m in matches]
        if not match_ids:
            return []
        match_players = list(
            (await session.execute(select(MatchPlayer).where(col(MatchPlayer.match_id).in_(match_ids)))).scalars().all()
        )
        players_by_match: dict[UUID, list[UUID]] = {}
        for mp in match_players:
            players_by_match.setdefault(mp.match_id, []).append(mp.policy_version_id)
        return [
            MatchData(
                match_id=m.id,
                status=m.status,
                player_pv_ids=players_by_match.get(m.id, []),
                assignments=m.assignments or [],
            )
            for m in matches
        ]

    async def _get_player_stats(self, pool_id: UUID) -> dict[UUID, tuple[float | None, int]]:
        session = get_db()
        return {
            row[0]: (float(row[1]) if row[1] is not None else None, row[2] or 0)
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

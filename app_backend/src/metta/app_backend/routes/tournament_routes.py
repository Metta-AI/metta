from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import db_session
from metta.app_backend.models.tournament import (
    Match,
    MatchPlayer,
    MatchStatus,
    MembershipChangeRecord,
    Pool,
    PoolPlayer,
    Season,
)
from metta.app_backend.route_logger import timed_http_handler
from metta.app_backend.tournament.registry import SEASONS


async def get_session():
    async with db_session() as session:
        yield session


class SeasonDetail(BaseModel):
    name: str
    pools: list[str]


class LeaderboardEntry(BaseModel):
    rank: int
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    score: float
    matches: int


class PolicyPoolStatus(BaseModel):
    pool_name: str
    status: str
    matches_completed: int
    avg_score: float | None


class PolicySummary(BaseModel):
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    pools: list[PolicyPoolStatus]


class SubmitRequest(BaseModel):
    policy_version_id: UUID


class SubmitResponse(BaseModel):
    pools: list[str]


class MatchPlayerSummary(BaseModel):
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    policy_index: int
    score: float | None


class MatchSummary(BaseModel):
    id: UUID
    pool_name: str
    status: MatchStatus
    assignments: list[int]
    players: list[MatchPlayerSummary]
    episode_id: str | None
    episode_tags: dict[str, str]
    created_at: str


class MembershipHistoryEntry(BaseModel):
    id: UUID
    pool_name: str
    action: str
    notes: str | None
    created_at: str


class PlayerDetail(BaseModel):
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    membership_history: list[MembershipHistoryEntry]


def create_tournament_router() -> APIRouter:
    router = APIRouter(prefix="/tournament", tags=["tournament"])

    @router.get("/seasons")
    @timed_http_handler
    async def list_seasons(_user: UserOrToken, session: AsyncSession = Depends(get_session)) -> list[str]:
        result = await session.execute(select(Season.name))
        existing = {row[0] for row in result.all()}
        return [name for name in SEASONS.keys() if name in existing]

    @router.get("/seasons/{season_name}")
    @timed_http_handler
    async def get_season(season_name: str, _user: UserOrToken) -> SeasonDetail:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        return SeasonDetail(name=season_name, pools=list(commissioner.referees.keys()))

    @router.get("/seasons/{season_name}/leaderboard")
    @timed_http_handler
    async def get_leaderboard(
        season_name: str, _user: UserOrToken, session: AsyncSession = Depends(get_session)
    ) -> list[LeaderboardEntry]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")

        commissioner = SEASONS[season_name]()
        leaderboard = await commissioner.get_leaderboard()

        if not leaderboard:
            return []

        pv_ids = [pv_id for pv_id, _, _ in leaderboard]
        result = await session.execute(
            text("""
                SELECT pv.id, p.name, pv.version
                FROM policy_versions pv
                JOIN policies p ON p.id = pv.policy_id
                WHERE pv.id = ANY(:pv_ids)
            """),
            {"pv_ids": pv_ids},
        )
        pv_info = {row[0]: (row[1], row[2]) for row in result.all()}

        return [
            LeaderboardEntry(
                rank=i + 1,
                policy_version_id=pv_id,
                policy_name=pv_info.get(pv_id, (None, None))[0],
                policy_version=pv_info.get(pv_id, (None, None))[1],
                score=score,
                matches=match_count,
            )
            for i, (pv_id, score, match_count) in enumerate(leaderboard)
        ]

    @router.get("/seasons/{season_name}/policies")
    @timed_http_handler
    async def get_policies(
        season_name: str, _user: UserOrToken, session: AsyncSession = Depends(get_session)
    ) -> list[PolicySummary]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")

        season = (await session.execute(select(Season).filter_by(name=season_name))).scalar_one_or_none()
        if not season:
            return []

        pools: dict[UUID, str] = {
            p.id: p.name
            for p in (await session.execute(select(Pool).filter_by(season_id=season.id))).scalars().all()
            if p.name is not None
        }
        if not pools:
            return []

        pool_players = list(
            (await session.execute(select(PoolPlayer).where(col(PoolPlayer.pool_id).in_(pools.keys())))).scalars().all()
        )

        all_pv_ids = list({pp.policy_version_id for pp in pool_players})
        if not all_pv_ids:
            return []

        result = await session.execute(
            text("""
                SELECT pv.id, p.name, pv.version
                FROM policy_versions pv
                JOIN policies p ON p.id = pv.policy_id
                WHERE pv.id = ANY(:pv_ids)
            """),
            {"pv_ids": all_pv_ids},
        )
        pv_info = {row[0]: (row[1], row[2]) for row in result.all()}

        match_stats = {}
        stats_result = await session.execute(
            select(
                MatchPlayer.policy_version_id,
                Match.pool_id,
                func.count().label("match_count"),
                func.avg(MatchPlayer.score).label("avg_score"),
            )
            .join(Match, col(MatchPlayer.match_id) == col(Match.id))
            .where(col(Match.pool_id).in_(pools.keys()))
            .where(Match.status == MatchStatus.completed)
            .group_by(col(MatchPlayer.policy_version_id), col(Match.pool_id))
        )
        for row in stats_result.all():
            match_stats[(row[0], row[1])] = (row[2], float(row[3]) if row[3] else None)

        policies_by_pv: dict[UUID, list[PolicyPoolStatus]] = {}
        for pp in pool_players:
            pv_id = pp.policy_version_id
            pool_name = pools[pp.pool_id]
            stats = match_stats.get((pv_id, pp.pool_id), (0, None))

            status = "retired" if pp.retired else "active"
            pool_status = PolicyPoolStatus(
                pool_name=pool_name,
                status=status,
                matches_completed=stats[0],
                avg_score=stats[1],
            )

            if pv_id not in policies_by_pv:
                policies_by_pv[pv_id] = []
            policies_by_pv[pv_id].append(pool_status)

        return [
            PolicySummary(
                policy_version_id=pv_id,
                policy_name=pv_info.get(pv_id, (None, None))[0],
                policy_version=pv_info.get(pv_id, (None, None))[1],
                pools=pool_statuses,
            )
            for pv_id, pool_statuses in policies_by_pv.items()
        ]

    @router.get("/seasons/{season_name}/matches")
    @timed_http_handler
    async def get_matches(
        season_name: str,
        _user: UserOrToken,
        session: AsyncSession = Depends(get_session),
        pool_name: str | None = None,
        policy_version_id: UUID | None = None,
        limit: int = 50,
    ) -> list[MatchSummary]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")

        season = (await session.execute(select(Season).filter_by(name=season_name))).scalar_one_or_none()
        if not season:
            return []

        pools: dict[UUID, str] = {
            p.id: p.name
            for p in (await session.execute(select(Pool).filter_by(season_id=season.id))).scalars().all()
            if p.name is not None
        }
        if not pools:
            return []

        pool_ids = list(pools.keys())
        if pool_name:
            pool_ids = [pid for pid, pname in pools.items() if pname == pool_name]
            if not pool_ids:
                return []

        query = (
            select(Match).where(col(Match.pool_id).in_(pool_ids)).order_by(col(Match.created_at).desc()).limit(limit)
        )
        matches = list((await session.execute(query)).scalars().all())
        if not matches:
            return []

        match_ids = [m.id for m in matches]
        players_result = await session.execute(select(MatchPlayer).where(col(MatchPlayer.match_id).in_(match_ids)))
        all_players = list(players_result.scalars().all())

        all_pv_ids = list({p.policy_version_id for p in all_players})
        pv_info: dict[UUID, tuple[str | None, int | None]] = {}
        if all_pv_ids:
            result = await session.execute(
                text("""
                    SELECT pv.id, p.name, pv.version
                    FROM policy_versions pv
                    JOIN policies p ON p.id = pv.policy_id
                    WHERE pv.id = ANY(:pv_ids)
                """),
                {"pv_ids": all_pv_ids},
            )
            pv_info = {row[0]: (row[1], row[2]) for row in result.all()}

        players_by_match: dict[UUID, list[MatchPlayer]] = {}
        for p in all_players:
            players_by_match.setdefault(p.match_id, []).append(p)

        if policy_version_id:
            matches = [
                m
                for m in matches
                if any(p.policy_version_id == policy_version_id for p in players_by_match.get(m.id, []))
            ]

        from metta.app_backend.models.job_request import JobRequest

        job_ids = [m.job_id for m in matches if m.job_id]
        episode_by_job: dict[UUID, str | None] = {}
        tags_by_job: dict[UUID, dict[str, str]] = {}
        if job_ids:
            jobs_result = await session.execute(
                select(JobRequest.id, JobRequest.job, JobRequest.result).where(col(JobRequest.id).in_(job_ids))
            )
            for row in jobs_result.all():
                job_id, job_spec, result = row[0], row[1], row[2]
                if result and isinstance(result, dict):
                    episode_by_job[job_id] = result.get("episode_id")
                if job_spec and isinstance(job_spec, dict):
                    tags_by_job[job_id] = job_spec.get("episode_tags", {})

        summaries = []
        for m in matches:
            match_players = sorted(players_by_match.get(m.id, []), key=lambda p: p.policy_index)
            episode_id = episode_by_job.get(m.job_id) if m.job_id else None
            episode_tags = tags_by_job.get(m.job_id, {}) if m.job_id else {}
            summaries.append(
                MatchSummary(
                    id=m.id,
                    pool_name=pools.get(m.pool_id, "unknown"),
                    status=m.status,
                    assignments=m.assignments or [],
                    players=[
                        MatchPlayerSummary(
                            policy_version_id=p.policy_version_id,
                            policy_name=pv_info.get(p.policy_version_id, (None, None))[0],
                            policy_version=pv_info.get(p.policy_version_id, (None, None))[1],
                            policy_index=p.policy_index,
                            score=p.score,
                        )
                        for p in match_players
                    ],
                    episode_id=episode_id,
                    episode_tags=episode_tags,
                    created_at=m.created_at.isoformat() if m.created_at else "",
                )
            )
        return summaries

    @router.post("/seasons/{season_name}/submit")
    @timed_http_handler
    async def submit_policy(season_name: str, request: SubmitRequest, _user: UserOrToken) -> SubmitResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        try:
            pool_names = await commissioner.submit(request.policy_version_id)
            return SubmitResponse(pools=pool_names)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.get("/seasons/{season_name}/players/{policy_version_id}")
    @timed_http_handler
    async def get_player(
        season_name: str, policy_version_id: UUID, _user: UserOrToken, session: AsyncSession = Depends(get_session)
    ) -> PlayerDetail:
        season = (await session.execute(select(Season).filter_by(name=season_name))).scalar_one_or_none()
        if not season:
            raise HTTPException(status_code=404, detail="Season not found")

        pools = {
            p.id: p.name
            for p in (await session.execute(select(Pool).filter_by(season_id=season.id))).scalars().all()
            if p.name is not None
        }
        pool_ids = list(pools.keys())

        pv_result = await session.execute(
            text("""
                SELECT pv.id, p.name, pv.version
                FROM policy_versions pv
                JOIN policies p ON p.id = pv.policy_id
                WHERE pv.id = :pv_id
            """),
            {"pv_id": policy_version_id},
        )
        pv_row = pv_result.one_or_none()
        if not pv_row:
            raise HTTPException(status_code=404, detail="Policy version not found")

        changes = list(
            (
                await session.execute(
                    select(MembershipChangeRecord)
                    .where(col(MembershipChangeRecord.policy_version_id) == policy_version_id)
                    .where(col(MembershipChangeRecord.pool_id).in_(pool_ids))
                    .order_by(col(MembershipChangeRecord.created_at).desc())
                )
            )
            .scalars()
            .all()
        )

        history = [
            MembershipHistoryEntry(
                id=c.id,
                pool_name=pools.get(c.pool_id, "unknown"),
                action=c.action.value,
                notes=c.notes,
                created_at=c.created_at.isoformat() if c.created_at else "",
            )
            for c in changes
        ]

        return PlayerDetail(
            policy_version_id=policy_version_id,
            policy_name=pv_row[1],
            policy_version=pv_row[2],
            membership_history=history,
        )

    return router

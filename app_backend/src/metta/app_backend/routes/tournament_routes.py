from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlmodel import col, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import db_session
from metta.app_backend.models.job_request import JobRequest
from metta.app_backend.models.policies import PolicyVersion
from metta.app_backend.models.tournament import (
    Match,
    MatchPlayer,
    MembershipChange,
    Pool,
    PoolPlayer,
    Season,
)
from metta.app_backend.route_logger import timed_http_handler
from metta.app_backend.tournament.registry import SEASONS


async def get_session():
    async with db_session() as session:
        yield session


async def get_season_pools(session: AsyncSession, season_name: str) -> dict[UUID, str]:
    result = await session.execute(
        select(Pool)
        .join(Season, Pool.season_id == Season.id)  # type: ignore[arg-type]
        .where(Season.name == season_name)
        .where(col(Pool.name).isnot(None))
    )
    return {p.id: p.name for p in result.scalars().all() if p.name is not None}


async def get_policy_versions(session: AsyncSession, pv_ids: list[UUID]) -> dict[UUID, PolicyVersion]:
    if not pv_ids:
        return {}
    result = await session.execute(
        select(PolicyVersion)
        .options(selectinload(PolicyVersion.policy))  # type: ignore[arg-type]
        .where(col(PolicyVersion.id).in_(pv_ids))
    )
    return {pv.id: pv for pv in result.scalars().all()}


class LeaderboardEntry(BaseModel):
    rank: int
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    score: float
    matches: int


class PoolMembership(BaseModel):
    pool_name: str
    active: bool


class PolicySummary(BaseModel):
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    pools: list[PoolMembership]


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
    status: str
    assignments: list[int]
    players: list[MatchPlayerSummary]
    episode_id: str | None
    created_at: str


class MembershipHistoryEntry(BaseModel):
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

    @router.get("/seasons/{season_name}")
    @timed_http_handler
    async def get_season(_user: UserOrToken, season_name: str) -> dict:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        return {"name": season_name, "pools": list(commissioner.referees.keys())}

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

        pvs = await get_policy_versions(session, [pv_id for pv_id, _, _ in leaderboard])
        return [
            LeaderboardEntry(
                rank=i + 1,
                policy_version_id=pv_id,
                policy_name=pvs[pv_id].policy.name if pv_id in pvs else None,
                policy_version=pvs[pv_id].version if pv_id in pvs else None,
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

        pools = await get_season_pools(session, season_name)
        if not pools:
            return []

        pool_players = list(
            (await session.execute(select(PoolPlayer).where(col(PoolPlayer.pool_id).in_(pools.keys())))).scalars().all()
        )
        pv_ids = list({pp.policy_version_id for pp in pool_players})
        if not pv_ids:
            return []

        pvs = await get_policy_versions(session, pv_ids)

        policies_by_pv: dict[UUID, list[PoolMembership]] = {}
        for pp in pool_players:
            membership = PoolMembership(pool_name=pools[pp.pool_id], active=not pp.retired)
            policies_by_pv.setdefault(pp.policy_version_id, []).append(membership)

        return [
            PolicySummary(
                policy_version_id=pv_id,
                policy_name=pvs[pv_id].policy.name if pv_id in pvs else None,
                policy_version=pvs[pv_id].version if pv_id in pvs else None,
                pools=memberships,
            )
            for pv_id, memberships in policies_by_pv.items()
        ]

    @router.get("/seasons/{season_name}/matches")
    @timed_http_handler
    async def get_matches(
        season_name: str,
        _user: UserOrToken,
        session: AsyncSession = Depends(get_session),
        limit: int = 100,
    ) -> list[MatchSummary]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")

        pools = await get_season_pools(session, season_name)
        if not pools:
            return []

        matches = list(
            (
                await session.execute(
                    select(Match)
                    .where(col(Match.pool_id).in_(pools.keys()))
                    .order_by(col(Match.created_at).desc())
                    .limit(limit)
                    .options(selectinload(Match.players).selectinload(MatchPlayer.pool_player))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )
        if not matches:
            return []

        pv_ids = list({mp.pool_player.policy_version_id for m in matches for mp in m.players})
        pvs = await get_policy_versions(session, pv_ids)

        job_ids = [m.job_id for m in matches if m.job_id]
        episode_by_job: dict[UUID, str | None] = {}
        if job_ids:
            jobs_result = await session.execute(
                select(col(JobRequest.id).label("job_id"), col(JobRequest.result).label("result")).where(
                    col(JobRequest.id).in_(job_ids)
                )
            )
            for row in jobs_result.all():
                if row.result and isinstance(row.result, dict):
                    episode_by_job[row.job_id] = row.result.get("episode_id")

        return [
            MatchSummary(
                id=m.id,
                pool_name=pools.get(m.pool_id, "unknown"),
                status=m.status.value,
                assignments=m.assignments or [],
                players=[
                    MatchPlayerSummary(
                        policy_version_id=mp.pool_player.policy_version_id,
                        policy_name=(
                            pvs[mp.pool_player.policy_version_id].policy.name
                            if mp.pool_player.policy_version_id in pvs
                            else None
                        ),
                        policy_version=(
                            pvs[mp.pool_player.policy_version_id].version
                            if mp.pool_player.policy_version_id in pvs
                            else None
                        ),
                        policy_index=mp.policy_index,
                        score=mp.score,
                    )
                    for mp in sorted(m.players, key=lambda p: p.policy_index)
                ],
                episode_id=episode_by_job.get(m.job_id) if m.job_id else None,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in matches
        ]

    @router.post("/seasons/{season_name}/submit")
    @timed_http_handler
    async def submit_policy(season_name: str, request: SubmitRequest, _user: UserOrToken) -> SubmitResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        pool_names = await commissioner.submit(request.policy_version_id)
        return SubmitResponse(pools=pool_names)

    @router.get("/seasons/{season_name}/players/{policy_version_id}")
    @timed_http_handler
    async def get_player(
        season_name: str, policy_version_id: UUID, _user: UserOrToken, session: AsyncSession = Depends(get_session)
    ) -> PlayerDetail:
        pools = await get_season_pools(session, season_name)
        if not pools:
            raise HTTPException(status_code=404, detail="Season not found")

        pvs = await get_policy_versions(session, [policy_version_id])
        if policy_version_id not in pvs:
            raise HTTPException(status_code=404, detail="Policy version not found")
        pv = pvs[policy_version_id]

        changes = list(
            (
                await session.execute(
                    select(MembershipChange)
                    .join(PoolPlayer, MembershipChange.pool_player_id == PoolPlayer.id)  # type: ignore[arg-type]
                    .where(PoolPlayer.policy_version_id == policy_version_id)
                    .where(col(PoolPlayer.pool_id).in_(pools.keys()))
                    .order_by(col(MembershipChange.created_at).desc())
                    .options(selectinload(MembershipChange.pool_player))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )

        return PlayerDetail(
            policy_version_id=policy_version_id,
            policy_name=pv.policy.name,
            policy_version=pv.version,
            membership_history=[
                MembershipHistoryEntry(
                    pool_name=pools.get(c.pool_player.pool_id, "unknown"),
                    action=c.action.value,
                    notes=c.notes,
                    created_at=c.created_at.isoformat() if c.created_at else "",
                )
                for c in changes
            ],
        )

    return router

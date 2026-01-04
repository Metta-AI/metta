from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlmodel import col, select

# pyright: reportArgumentType=false
# SQLModel's Relationship() returns the target type, not SQLAlchemy's InstrumentedAttribute,
# causing false positives on join() and selectinload() calls.
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


class PolicyVersionSummary(BaseModel):
    id: UUID
    name: str | None
    version: int | None

    @classmethod
    def from_model(cls, pv: PolicyVersion) -> "PolicyVersionSummary":
        return cls(id=pv.id, name=pv.policy.name, version=pv.version)


class LeaderboardEntry(BaseModel):
    rank: int
    policy: PolicyVersionSummary
    score: float
    matches: int


class PoolMembership(BaseModel):
    pool_name: str
    active: bool


class PolicySummary(BaseModel):
    policy: PolicyVersionSummary
    pools: list[PoolMembership]
    entered_at: str


class SubmitRequest(BaseModel):
    policy_version_id: UUID


class SubmitResponse(BaseModel):
    pools: list[str]


class MatchPlayerSummary(BaseModel):
    policy: PolicyVersionSummary
    policy_index: int
    score: float | None


class MatchSummary(BaseModel):
    id: UUID
    pool_name: str
    status: str
    assignments: list[int]
    players: list[MatchPlayerSummary]
    job_id: UUID | None
    episode_id: str | None
    created_at: str


class MembershipHistoryEntry(BaseModel):
    season_name: str
    pool_name: str
    action: str
    notes: str | None
    created_at: str


class PoolDescriptionResponse(BaseModel):
    name: str
    description: str


class SeasonDescriptionResponse(BaseModel):
    summary: str
    pools: list[PoolDescriptionResponse]


class SeasonResponse(BaseModel):
    name: str
    description: SeasonDescriptionResponse
    pools: list[str]

    @classmethod
    def from_commissioner(cls, season_name: str) -> "SeasonResponse":
        commissioner = SEASONS[season_name]()
        desc = commissioner.description
        return cls(
            name=season_name,
            description=SeasonDescriptionResponse(
                summary=desc.summary,
                pools=[PoolDescriptionResponse(name=p.name, description=p.description) for p in desc.pools],
            ),
            pools=[p.name for p in desc.pools],
        )


def create_tournament_router() -> APIRouter:
    router = APIRouter(prefix="/tournament", tags=["tournament"])

    @router.get("/seasons")
    @timed_http_handler
    async def list_seasons(_user: UserOrToken) -> list[SeasonResponse]:
        return [SeasonResponse.from_commissioner(name) for name in SEASONS.keys()]

    @router.get("/seasons/{season_name}")
    @timed_http_handler
    async def get_season(_user: UserOrToken, season_name: str) -> SeasonResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        return SeasonResponse.from_commissioner(season_name)

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
        pvs_result = (
            (
                await session.execute(
                    select(PolicyVersion)
                    .where(col(PolicyVersion.id).in_(pv_ids))
                    .options(selectinload(PolicyVersion.policy))
                )
            )
            .scalars()
            .all()
        )
        pvs = {pv.id: pv for pv in pvs_result}

        return [
            LeaderboardEntry(
                rank=i + 1,
                policy=PolicyVersionSummary.from_model(pvs[pv_id])
                if pv_id in pvs
                else PolicyVersionSummary(id=pv_id, name=None, version=None),
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

        policy_versions = (
            (
                await session.execute(
                    select(PolicyVersion)
                    .join(PolicyVersion.pool_players)
                    .join(PoolPlayer.pool)
                    .join(Pool.season)
                    .where(Season.name == season_name)
                    .options(
                        selectinload(PolicyVersion.policy),
                        selectinload(PolicyVersion.pool_players).selectinload(PoolPlayer.pool),
                    )
                    .distinct()
                )
            )
            .scalars()
            .all()
        )

        if not policy_versions:
            return []

        results = [
            PolicySummary(
                policy=PolicyVersionSummary.from_model(pv),
                pools=[
                    PoolMembership(pool_name=pp.pool.name or "unknown", active=not pp.retired) for pp in pv.pool_players
                ],
                entered_at=min(pp.created_at for pp in pv.pool_players).isoformat() if pv.pool_players else "",
            )
            for pv in policy_versions
        ]

        return sorted(results, key=lambda x: x.entered_at, reverse=True)

    @router.get("/seasons/{season_name}/matches")
    @timed_http_handler
    async def get_matches(
        season_name: str,
        _user: UserOrToken,
        session: AsyncSession = Depends(get_session),
        limit: int = 50,
        offset: int = 0,
        pool_names: list[str] | None = Query(default=None),
        policy_version_ids: list[UUID] | None = Query(default=None),
    ) -> list[MatchSummary]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")

        query = (
            select(Match, JobRequest.episode_id)
            .join(Match.job)
            .join(Match.pool)
            .join(Pool.season)
            .where(Season.name == season_name)
        )

        if pool_names:
            query = query.where(col(Pool.name).in_(pool_names))

        if policy_version_ids:
            for pv_id in policy_version_ids:
                subq = (
                    select(MatchPlayer.match_id)
                    .join(MatchPlayer.pool_player)
                    .where(PoolPlayer.policy_version_id == pv_id)
                )
                query = query.where(Match.id.in_(subq))

        query = (
            query.order_by(col(Match.created_at).desc())
            .limit(limit)
            .offset(offset)
            .options(
                selectinload(Match.players)
                .selectinload(MatchPlayer.pool_player)
                .selectinload(PoolPlayer.policy_version)
                .selectinload(PolicyVersion.policy),
                selectinload(Match.pool),
            )
        )

        rows = (await session.execute(query)).all()
        if not rows:
            return []

        return [
            MatchSummary(
                id=m.id,
                pool_name=m.pool.name,
                status=m.status.value,
                assignments=m.assignments or [],
                players=[
                    MatchPlayerSummary(
                        policy=PolicyVersionSummary.from_model(mp.pool_player.policy_version),
                        policy_index=mp.policy_index,
                        score=mp.score,
                    )
                    for mp in sorted(m.players, key=lambda p: p.policy_index)
                ],
                job_id=m.job_id,
                episode_id=episode_id,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m, episode_id in rows
        ]

    @router.post("/seasons/{season_name}/submit")
    @timed_http_handler
    async def submit_policy(season_name: str, request: SubmitRequest, _user: UserOrToken) -> SubmitResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        pool_names = await commissioner.submit(request.policy_version_id)
        return SubmitResponse(pools=pool_names)

    @router.get("/players/{policy_version_id}/memberships")
    @timed_http_handler
    async def get_player_memberships(
        policy_version_id: UUID, _user: UserOrToken, session: AsyncSession = Depends(get_session)
    ) -> list[MembershipHistoryEntry]:
        changes = (
            (
                await session.execute(
                    select(MembershipChange)
                    .join(MembershipChange.pool_player)
                    .join(PoolPlayer.pool)
                    .join(Pool.season)
                    .where(PoolPlayer.policy_version_id == policy_version_id)
                    .order_by(col(MembershipChange.created_at).desc())
                    .options(
                        selectinload(MembershipChange.pool_player)
                        .selectinload(PoolPlayer.pool)
                        .selectinload(Pool.season)
                    )
                )
            )
            .scalars()
            .all()
        )

        return [
            MembershipHistoryEntry(
                season_name=c.pool_player.pool.season.name if c.pool_player.pool.season else "unknown",
                pool_name=c.pool_player.pool.name or "unknown",
                action=c.action.value,
                notes=c.notes,
                created_at=c.created_at.isoformat(),
            )
            for c in changes
        ]

    return router

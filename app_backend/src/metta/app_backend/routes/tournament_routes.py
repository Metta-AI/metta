from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
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
    episode_id: str | None
    created_at: str


class MembershipHistoryEntry(BaseModel):
    pool_name: str
    action: str
    notes: str | None
    created_at: str


class PlayerDetail(BaseModel):
    policy: PolicyVersionSummary
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

        pool_players = (
            (
                await session.execute(
                    select(PoolPlayer)
                    .join(PoolPlayer.pool)
                    .join(Pool.season)
                    .where(Season.name == season_name)
                    .options(
                        selectinload(PoolPlayer.policy_version).selectinload(PolicyVersion.policy),
                        selectinload(PoolPlayer.pool),
                    )
                )
            )
            .scalars()
            .all()
        )
        if not pool_players:
            return []

        policies_by_pv: dict[UUID, tuple[PolicyVersion, list[PoolMembership]]] = {}
        for pp in pool_players:
            membership = PoolMembership(pool_name=pp.pool.name or "unknown", active=not pp.retired)
            if pp.policy_version_id not in policies_by_pv:
                policies_by_pv[pp.policy_version_id] = (pp.policy_version, [])
            policies_by_pv[pp.policy_version_id][1].append(membership)

        return [
            PolicySummary(policy=PolicyVersionSummary.from_model(pv), pools=memberships)
            for pv, memberships in policies_by_pv.values()
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

        rows = (
            await session.execute(
                select(Match, JobRequest.episode_id)
                .join(Match.job)
                .join(Match.pool)
                .join(Pool.season)
                .where(Season.name == season_name)
                .order_by(col(Match.created_at).desc())
                .limit(limit)
                .options(
                    selectinload(Match.players)
                    .selectinload(MatchPlayer.pool_player)
                    .selectinload(PoolPlayer.policy_version)
                    .selectinload(PolicyVersion.policy),
                    selectinload(Match.pool),
                )
            )
        ).all()
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

    @router.get("/seasons/{season_name}/players/{policy_version_id}")
    @timed_http_handler
    async def get_player(
        season_name: str, policy_version_id: UUID, _user: UserOrToken, session: AsyncSession = Depends(get_session)
    ) -> PlayerDetail:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")

        pv = (
            await session.execute(
                select(PolicyVersion)
                .where(PolicyVersion.id == policy_version_id)
                .options(selectinload(PolicyVersion.policy))
            )
        ).scalar_one_or_none()
        if not pv:
            raise HTTPException(status_code=404, detail="Policy version not found")

        changes = (
            (
                await session.execute(
                    select(MembershipChange)
                    .join(MembershipChange.pool_player)
                    .join(PoolPlayer.pool)
                    .join(Pool.season)
                    .where(PoolPlayer.policy_version_id == policy_version_id)
                    .where(Season.name == season_name)
                    .order_by(col(MembershipChange.created_at).desc())
                    .options(selectinload(MembershipChange.pool_player).selectinload(PoolPlayer.pool))
                )
            )
            .scalars()
            .all()
        )

        return PlayerDetail(
            policy=PolicyVersionSummary.from_model(pv),
            membership_history=[
                MembershipHistoryEntry(
                    pool_name=c.pool_player.pool.name or "unknown",
                    action=c.action.value,
                    notes=c.notes,
                    created_at=c.created_at.isoformat(),
                )
                for c in changes
            ],
        )

    return router

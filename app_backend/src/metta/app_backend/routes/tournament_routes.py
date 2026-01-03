import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import col, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import db_session
from metta.app_backend.models.tournament import (
    MatchStatus,
    MembershipAction,
    MembershipChangeRecord,
    Pool,
    Season,
)
from metta.app_backend.route_logger import timed_http_handler
from metta.app_backend.tournament.registry import SEASONS

logger = logging.getLogger(__name__)


class SeasonDetail(BaseModel):
    name: str
    pools: list[str]


class LeaderboardEntry(BaseModel):
    rank: int
    policy_version_id: UUID
    score: float


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


class MatchPlayerSummary(BaseModel):
    policy_version_id: UUID
    policy_index: int
    score: float | None


class MatchSummary(BaseModel):
    id: UUID
    job_id: UUID | None
    status: MatchStatus
    players: list[MatchPlayerSummary]
    created_at: str
    completed_at: str | None


class SeasonSubmissionRequest(BaseModel):
    policy_version_id: UUID


class SubmissionResponse(BaseModel):
    status: str
    pool_names: list[str]


class MembershipChangeResponse(BaseModel):
    id: UUID
    pool_id: UUID
    pool_name: str | None
    policy_version_id: UUID
    action: MembershipAction
    created_at: str


def create_tournament_router() -> APIRouter:
    router = APIRouter(prefix="/tournament", tags=["tournament"])

    @router.get("/seasons")
    @timed_http_handler
    async def list_seasons(_user: UserOrToken) -> list[str]:
        async with db_session() as session:
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

    @router.post("/seasons/{season_name}/submit")
    @timed_http_handler
    async def submit_to_season(
        season_name: str, request: SeasonSubmissionRequest, _user: UserOrToken
    ) -> SubmissionResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        try:
            pool_names = await commissioner.submit(request.policy_version_id)
            return SubmissionResponse(status="submitted", pool_names=pool_names)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.get("/seasons/{season_name}/leaderboard")
    @timed_http_handler
    async def get_season_leaderboard(
        season_name: str, _user: UserOrToken, pool_name: str | None = Query(default=None)
    ) -> LeaderboardResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        try:
            scores = await commissioner.get_leaderboard(pool_name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        entries = [
            LeaderboardEntry(rank=i + 1, policy_version_id=pv_id, score=score)
            for i, (pv_id, score) in enumerate(scores)
        ]
        return LeaderboardResponse(entries=entries)

    @router.get("/seasons/{season_name}/pools/{pool_name}/matches")
    @timed_http_handler
    async def get_pool_matches(
        season_name: str,
        pool_name: str,
        _user: UserOrToken,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> list[MatchSummary]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        try:
            matches = await commissioner.get_matches(pool_name, limit, offset)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return [
            MatchSummary(
                id=m.id,
                job_id=m.job_id,
                status=m.status,
                players=[
                    MatchPlayerSummary(
                        policy_version_id=p.policy_version_id,
                        policy_index=p.policy_index,
                        score=p.score,
                    )
                    for p in m.players
                ],
                created_at=m.created_at.isoformat(),
                completed_at=m.completed_at.isoformat() if m.completed_at else None,
            )
            for m in matches
        ]

    @router.get("/seasons/{season_name}/membership-changes")
    @timed_http_handler
    async def get_membership_changes(
        season_name: str,
        _user: UserOrToken,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> list[MembershipChangeResponse]:
        async with db_session() as session:
            season = (await session.execute(select(Season).filter_by(name=season_name))).scalar_one_or_none()
            if not season:
                raise HTTPException(status_code=404, detail="Season not found")

            pool_ids = [row[0] for row in (await session.execute(select(Pool.id).filter_by(season_id=season.id))).all()]
            if not pool_ids:
                return []

            changes = list(
                (
                    await session.execute(
                        select(MembershipChangeRecord)
                        .where(col(MembershipChangeRecord.pool_id).in_(pool_ids))
                        .order_by(col(MembershipChangeRecord.created_at).desc())
                        .offset(offset)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

            pool_names = {
                row[0]: row[1]
                for row in (await session.execute(select(Pool.id, Pool.name).filter_by(season_id=season.id))).all()
            }

        return [
            MembershipChangeResponse(
                id=c.id,
                pool_id=c.pool_id,
                pool_name=pool_names.get(c.pool_id),
                policy_version_id=c.policy_version_id,
                action=c.action,
                created_at=c.created_at.isoformat(),
            )
            for c in changes
        ]

    return router

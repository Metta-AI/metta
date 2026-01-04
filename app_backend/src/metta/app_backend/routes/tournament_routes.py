import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import col, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import db_session
from metta.app_backend.models.job_request import JobRequest
from metta.app_backend.models.tournament import (
    Match,
    MatchStatus,
    MembershipAction,
    MembershipChangeRecord,
    Pool,
    PoolPlayer,
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
    policy_name: str | None
    policy_version: int | None
    score: float


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


class MatchPlayerSummary(BaseModel):
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    policy_index: int
    score: float | None


class MatchSummary(BaseModel):
    id: UUID
    job_id: UUID | None
    status: MatchStatus
    assignments: list[int]
    players: list[MatchPlayerSummary]
    episode_tags: dict[str, Any]
    episode_id: str | None
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


class PoolMember(BaseModel):
    policy_version_id: UUID
    policy_name: str | None
    policy_version: int | None
    added_at: str
    retired: bool
    retired_at: str | None


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
    async def get_season_leaderboard(season_name: str, _user: UserOrToken) -> LeaderboardResponse:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        scores = await commissioner.get_leaderboard()

        pv_ids = [pv_id for pv_id, _ in scores]
        pv_info: dict[UUID, tuple[str, int]] = {}
        if pv_ids:
            async with db_session() as session:
                result = await session.execute(
                    text("""
                        SELECT pv.id, p.name, pv.version
                        FROM policy_versions pv
                        JOIN policies p ON p.id = pv.policy_id
                        WHERE pv.id = ANY(:pv_ids)
                    """),
                    {"pv_ids": pv_ids},
                )
                for row in result.all():
                    pv_info[row[0]] = (row[1], row[2])

        entries = [
            LeaderboardEntry(
                rank=i + 1,
                policy_version_id=pv_id,
                policy_name=pv_info.get(pv_id, (None, None))[0],
                policy_version=pv_info.get(pv_id, (None, None))[1],
                score=score,
            )
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
        include_policy_info: bool = Query(default=True),
    ) -> list[MatchSummary]:
        if season_name not in SEASONS:
            raise HTTPException(status_code=404, detail="Season not found")
        commissioner = SEASONS[season_name]()
        try:
            matches: list[Match] = await commissioner.get_matches(pool_name, limit, offset)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        if not matches:
            return []

        async with db_session() as session:
            job_ids = [m.job_id for m in matches if m.job_id]
            jobs_by_id: dict[UUID, JobRequest] = {}
            if job_ids:
                jobs = (
                    (await session.execute(select(JobRequest).where(col(JobRequest.id).in_(job_ids)))).scalars().all()
                )
                jobs_by_id = {j.id: j for j in jobs}

            pv_info: dict[UUID, tuple[str, int]] = {}
            if include_policy_info:
                all_pv_ids = list({p.policy_version_id for m in matches for p in m.players})
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
                    for row in result.all():
                        pv_info[row[0]] = (row[1], row[2])

        summaries = []
        for m in matches:
            job = jobs_by_id.get(m.job_id) if m.job_id else None
            episode_tags: dict[str, Any] = {}
            episode_id: str | None = None
            if job:
                if job.job and isinstance(job.job, dict):
                    episode_tags = job.job.get("episode_tags", {})
                if job.result and isinstance(job.result, dict):
                    episode_id = job.result.get("episode_id")

            summaries.append(
                MatchSummary(
                    id=m.id,
                    job_id=m.job_id,
                    status=m.status,
                    assignments=m.assignments,
                    players=[
                        MatchPlayerSummary(
                            policy_version_id=p.policy_version_id,
                            policy_name=pv_info.get(p.policy_version_id, (None, None))[0],
                            policy_version=pv_info.get(p.policy_version_id, (None, None))[1],
                            policy_index=p.policy_index,
                            score=p.score,
                        )
                        for p in m.players
                    ],
                    episode_tags=episode_tags,
                    episode_id=episode_id,
                    created_at=m.created_at.isoformat(),
                    completed_at=m.completed_at.isoformat() if m.completed_at else None,
                )
            )
        return summaries

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

    @router.get("/seasons/{season_name}/pools/{pool_name}/members")
    @timed_http_handler
    async def get_pool_members(
        season_name: str,
        pool_name: str,
        _user: UserOrToken,
    ) -> list[PoolMember]:
        async with db_session() as session:
            season = (await session.execute(select(Season).filter_by(name=season_name))).scalar_one_or_none()
            if not season:
                raise HTTPException(status_code=404, detail="Season not found")

            pool = (
                await session.execute(select(Pool).filter_by(season_id=season.id, name=pool_name))
            ).scalar_one_or_none()
            if not pool:
                raise HTTPException(status_code=404, detail="Pool not found")

            players = list(
                (
                    await session.execute(
                        select(PoolPlayer).filter_by(pool_id=pool.id).order_by(col(PoolPlayer.added_at))
                    )
                )
                .scalars()
                .all()
            )

            if not players:
                return []

            pv_ids = [p.policy_version_id for p in players]
            pv_info: dict[UUID, tuple[str, int]] = {}
            result = await session.execute(
                text("""
                    SELECT pv.id, p.name, pv.version
                    FROM policy_versions pv
                    JOIN policies p ON p.id = pv.policy_id
                    WHERE pv.id = ANY(:pv_ids)
                """),
                {"pv_ids": pv_ids},
            )
            for row in result.all():
                pv_info[row[0]] = (row[1], row[2])

        return [
            PoolMember(
                policy_version_id=p.policy_version_id,
                policy_name=pv_info.get(p.policy_version_id, (None, None))[0],
                policy_version=pv_info.get(p.policy_version_id, (None, None))[1],
                added_at=p.added_at.isoformat(),
                retired=p.retired,
                retired_at=p.removed_at.isoformat() if p.removed_at else None,
            )
            for p in players
        ]

    return router

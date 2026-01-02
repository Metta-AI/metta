import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import col, func, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import get_session
from metta.app_backend.models.tournament import (
    Match,
    MatchPlayer,
    MatchStatus,
    Pool,
    PoolPlayer,
    Season,
    SeasonCreate,
)
from metta.app_backend.route_logger import timed_http_handler

logger = logging.getLogger(__name__)


class SeasonSummary(BaseModel):
    id: UUID
    name: str
    description: str | None
    player_count: int


class PoolSummary(BaseModel):
    id: UUID
    name: str | None
    is_academy: bool
    player_count: int


class SeasonDetail(BaseModel):
    id: UUID
    name: str
    description: str | None
    pools: list[PoolSummary]


class PoolPlayerSummary(BaseModel):
    policy_version_id: UUID
    added_at: str
    retired: bool
    avg_score: float | None
    match_count: int


class PoolDetail(BaseModel):
    id: UUID
    name: str | None
    is_academy: bool
    season_id: UUID | None
    players: list[PoolPlayerSummary]


class MatchPlayerSummary(BaseModel):
    policy_version_id: UUID
    policy_index: int
    score: float | None


class MatchSummary(BaseModel):
    id: UUID
    status: MatchStatus
    environment_name: str
    players: list[MatchPlayerSummary]
    created_at: str
    completed_at: str | None


class MatchDetail(MatchSummary):
    pool_id: UUID
    job_id: UUID | None
    assignments: list[int]


class LeaderboardEntry(BaseModel):
    rank: int
    policy_version_id: UUID
    avg_score: float
    match_count: int


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


class SeasonSubmissionRequest(BaseModel):
    policy_version_id: UUID


class SubmissionResponse(BaseModel):
    status: str
    pool_id: UUID


def create_tournament_router() -> APIRouter:
    router = APIRouter(prefix="/tournament", tags=["tournament"])

    # --- Seasons ---

    @router.get("/seasons")
    @timed_http_handler
    async def list_seasons(_user: UserOrToken) -> list[SeasonSummary]:
        async with get_session() as session:
            seasons_result = await session.execute(select(Season).order_by(col(Season.created_at).desc()))
            seasons = list(seasons_result.scalars().all())

            summaries = []
            for season in seasons:
                count_result = await session.execute(
                    select(func.count(PoolPlayer.id))
                    .join(Pool, PoolPlayer.pool_id == Pool.id)
                    .where(Pool.season_id == season.id)
                    .where(PoolPlayer.retired == False)  # noqa: E712
                )
                player_count = count_result.scalar() or 0
                summaries.append(
                    SeasonSummary(
                        id=season.id,
                        name=season.name,
                        description=season.description,
                        player_count=player_count,
                    )
                )
            return summaries

    @router.post("/seasons")
    @timed_http_handler
    async def create_season(request: SeasonCreate, _user: UserOrToken) -> Season:
        async with get_session() as session:
            season = Season(**request.model_dump())
            session.add(season)
            await session.flush()
            pool = Pool(season_id=season.id, name="Main")
            session.add(pool)
            await session.commit()
            await session.refresh(season)
            return season

    @router.get("/seasons/{season_id}")
    @timed_http_handler
    async def get_season(season_id: UUID, _user: UserOrToken) -> SeasonDetail:
        async with get_session() as session:
            result = await session.execute(select(Season).where(Season.id == season_id))
            season = result.scalar_one_or_none()
            if not season:
                raise HTTPException(status_code=404, detail="Season not found")

            pools_result = await session.execute(select(Pool).where(Pool.season_id == season_id))
            pools = list(pools_result.scalars().all())

            pool_summaries = []
            for pool in pools:
                count_result = await session.execute(
                    select(func.count(PoolPlayer.id))
                    .where(PoolPlayer.pool_id == pool.id)
                    .where(PoolPlayer.retired == False)  # noqa: E712
                )
                player_count = count_result.scalar() or 0
                pool_summaries.append(
                    PoolSummary(id=pool.id, name=pool.name, is_academy=pool.is_academy, player_count=player_count)
                )

            return SeasonDetail(id=season.id, name=season.name, description=season.description, pools=pool_summaries)

    @router.post("/seasons/{season_id}/submit")
    @timed_http_handler
    async def submit_to_season(
        season_id: UUID, request: SeasonSubmissionRequest, user: UserOrToken
    ) -> SubmissionResponse:
        async with get_session() as session:
            # Validate season exists
            season_result = await session.execute(select(Season).where(Season.id == season_id))
            season = season_result.scalar_one_or_none()
            if not season:
                raise HTTPException(status_code=404, detail="Season not found")

            # Get season's default pool
            pool_result = await session.execute(
                select(Pool).where(Pool.season_id == season_id).order_by(col(Pool.created_at).asc())
            )
            pool = pool_result.scalars().first()
            if not pool:
                raise HTTPException(status_code=500, detail="Season has no pools")

            # Check not already in pool
            existing_result = await session.execute(
                select(PoolPlayer)
                .where(PoolPlayer.pool_id == pool.id)
                .where(PoolPlayer.policy_version_id == request.policy_version_id)
                .where(PoolPlayer.retired == False)  # noqa: E712
            )
            if existing_result.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Already submitted to this season")

            # Add to pool
            pool_player = PoolPlayer(pool_id=pool.id, policy_version_id=request.policy_version_id)
            session.add(pool_player)
            await session.commit()

            return SubmissionResponse(status="submitted", pool_id=pool.id)

    @router.get("/seasons/{season_id}/leaderboard")
    @timed_http_handler
    async def get_season_leaderboard(season_id: UUID, _user: UserOrToken) -> LeaderboardResponse:
        return await _get_leaderboard_for_season(season_id)

    # --- Pools ---

    @router.get("/pools/{pool_id}")
    @timed_http_handler
    async def get_pool(pool_id: UUID, _user: UserOrToken) -> PoolDetail:
        async with get_session() as session:
            result = await session.execute(select(Pool).where(Pool.id == pool_id))
            pool = result.scalar_one_or_none()
            if not pool:
                raise HTTPException(status_code=404, detail="Pool not found")
            return await _get_pool_detail(session, pool)

    @router.get("/pools/{pool_id}/matches")
    @timed_http_handler
    async def get_pool_matches(
        pool_id: UUID,
        _user: UserOrToken,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> list[MatchSummary]:
        async with get_session() as session:
            result = await session.execute(
                select(Match)
                .where(Match.pool_id == pool_id)
                .order_by(col(Match.created_at).desc())
                .offset(offset)
                .limit(limit)
            )
            matches = list(result.scalars().all())

            summaries = []
            for match in matches:
                players_result = await session.execute(select(MatchPlayer).where(MatchPlayer.match_id == match.id))
                players = list(players_result.scalars().all())
                summaries.append(
                    MatchSummary(
                        id=match.id,
                        status=match.status,
                        environment_name=match.environment_name,
                        players=[
                            MatchPlayerSummary(
                                policy_version_id=p.policy_version_id, policy_index=p.policy_index, score=p.score
                            )
                            for p in players
                        ],
                        created_at=match.created_at.isoformat(),
                        completed_at=match.completed_at.isoformat() if match.completed_at else None,
                    )
                )
            return summaries

    @router.get("/pools/{pool_id}/leaderboard")
    @timed_http_handler
    async def get_pool_leaderboard(pool_id: UUID, _user: UserOrToken) -> LeaderboardResponse:
        return await _get_leaderboard_for_pool(pool_id)

    # --- Matches ---

    @router.get("/matches/{match_id}")
    @timed_http_handler
    async def get_match(match_id: UUID, _user: UserOrToken) -> MatchDetail:
        async with get_session() as session:
            result = await session.execute(select(Match).where(Match.id == match_id))
            match = result.scalar_one_or_none()
            if not match:
                raise HTTPException(status_code=404, detail="Match not found")

            players_result = await session.execute(select(MatchPlayer).where(MatchPlayer.match_id == match.id))
            players = list(players_result.scalars().all())

            return MatchDetail(
                id=match.id,
                pool_id=match.pool_id,
                job_id=match.job_id,
                status=match.status,
                environment_name=match.environment_name,
                assignments=match.assignments,
                players=[
                    MatchPlayerSummary(
                        policy_version_id=p.policy_version_id, policy_index=p.policy_index, score=p.score
                    )
                    for p in players
                ],
                created_at=match.created_at.isoformat(),
                completed_at=match.completed_at.isoformat() if match.completed_at else None,
            )

    return router


async def _get_pool_detail(session, pool: Pool) -> PoolDetail:
    players_result = await session.execute(
        select(PoolPlayer).where(PoolPlayer.pool_id == pool.id).order_by(col(PoolPlayer.added_at).desc())
    )
    players = list(players_result.scalars().all())

    player_summaries = []
    for pp in players:
        # Get average score and match count for this player in this pool
        stats_result = await session.execute(
            select(func.avg(MatchPlayer.score), func.count(MatchPlayer.id))
            .join(Match, MatchPlayer.match_id == Match.id)
            .where(Match.pool_id == pool.id)
            .where(Match.status == MatchStatus.completed)
            .where(MatchPlayer.policy_version_id == pp.policy_version_id)
            .where(MatchPlayer.score.is_not(None))
        )
        avg_score, match_count = stats_result.one()

        player_summaries.append(
            PoolPlayerSummary(
                policy_version_id=pp.policy_version_id,
                added_at=pp.added_at.isoformat(),
                retired=pp.retired,
                avg_score=float(avg_score) if avg_score else None,
                match_count=match_count or 0,
            )
        )

    return PoolDetail(
        id=pool.id,
        name=pool.name,
        is_academy=pool.is_academy,
        season_id=pool.season_id,
        players=player_summaries,
    )


async def _get_leaderboard_for_pool(pool_id: UUID) -> LeaderboardResponse:
    async with get_session() as session:
        # Get average score per policy for completed matches in this pool
        result = await session.execute(
            select(
                MatchPlayer.policy_version_id,
                func.avg(MatchPlayer.score).label("avg_score"),
                func.count(MatchPlayer.id).label("match_count"),
            )
            .join(Match, MatchPlayer.match_id == Match.id)
            .where(Match.pool_id == pool_id)
            .where(Match.status == MatchStatus.completed)
            .where(MatchPlayer.score.is_not(None))
            .group_by(MatchPlayer.policy_version_id)
            .order_by(func.avg(MatchPlayer.score).desc())
        )
        rows = result.all()

        entries = []
        for rank, (policy_version_id, avg_score, match_count) in enumerate(rows, 1):
            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    policy_version_id=policy_version_id,
                    avg_score=float(avg_score),
                    match_count=match_count,
                )
            )
        return LeaderboardResponse(entries=entries)


async def _get_leaderboard_for_season(season_id: UUID) -> LeaderboardResponse:
    async with get_session() as session:
        # Get all pools for this season
        pools_result = await session.execute(select(Pool.id).where(Pool.season_id == season_id))
        pool_ids = [row[0] for row in pools_result.all()]

        if not pool_ids:
            return LeaderboardResponse(entries=[])

        # Get average score per policy across all season pools
        result = await session.execute(
            select(
                MatchPlayer.policy_version_id,
                func.avg(MatchPlayer.score).label("avg_score"),
                func.count(MatchPlayer.id).label("match_count"),
            )
            .join(Match, MatchPlayer.match_id == Match.id)
            .where(col(Match.pool_id).in_(pool_ids))
            .where(Match.status == MatchStatus.completed)
            .where(MatchPlayer.score.is_not(None))
            .group_by(MatchPlayer.policy_version_id)
            .order_by(func.avg(MatchPlayer.score).desc())
        )
        rows = result.all()

        entries = []
        for rank, (policy_version_id, avg_score, match_count) in enumerate(rows, 1):
            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    policy_version_id=policy_version_id,
                    avg_score=float(avg_score),
                    match_count=match_count,
                )
            )
        return LeaderboardResponse(entries=entries)

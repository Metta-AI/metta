import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from metta.app_backend.auth import UserOrToken
from metta.app_backend.metta_repo import (
    MatchRow,
    MatchWithPlayers,
    MettaRepo,
    PoolPlayerWithPolicy,
    PoolRow,
    SeasonRow,
)
from metta.app_backend.route_logger import timed_http_handler


class UUIDResponse(BaseModel):
    id: uuid.UUID


class SeasonCreate(BaseModel):
    name: str
    commissioner_class: str
    scorer_class: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class SeasonResponse(BaseModel):
    season: SeasonRow


class SeasonsResponse(BaseModel):
    seasons: list[SeasonRow]


class PoolCreate(BaseModel):
    referee_class: str
    season_id: uuid.UUID | None = None
    name: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class PoolResponse(BaseModel):
    pool: PoolRow


class PoolsResponse(BaseModel):
    pools: list[PoolRow]


class PoolPlayersResponse(BaseModel):
    players: list[PoolPlayerWithPolicy]


class AddPoolPlayerRequest(BaseModel):
    policy_version_id: uuid.UUID
    attributes: dict[str, Any] = Field(default_factory=dict)


class SubmitToSeasonRequest(BaseModel):
    policy_version_id: uuid.UUID
    season_id: uuid.UUID


class MatchCreate(BaseModel):
    pool_id: uuid.UUID
    policy_version_ids: list[uuid.UUID]


class MatchResponse(BaseModel):
    match: MatchWithPlayers


class MatchesResponse(BaseModel):
    matches: list[MatchRow]


class UnscheduledMatchesResponse(BaseModel):
    matches: list[MatchWithPlayers]


class SetMatchEvalTaskRequest(BaseModel):
    eval_task_id: int


def create_tournament_router(repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/tournament", tags=["tournament"])

    @router.post("/seasons")
    @timed_http_handler
    async def create_season(request: SeasonCreate, user: UserOrToken) -> UUIDResponse:
        season_id = await repo.create_season(
            name=request.name,
            commissioner_class=request.commissioner_class,
            scorer_class=request.scorer_class,
            attributes=request.attributes,
        )
        return UUIDResponse(id=season_id)

    @router.get("/seasons")
    @timed_http_handler
    async def get_seasons(limit: int = 50, offset: int = 0) -> SeasonsResponse:
        seasons = await repo.get_seasons(limit=limit, offset=offset)
        return SeasonsResponse(seasons=seasons)

    @router.get("/seasons/{season_id}")
    @timed_http_handler
    async def get_season(season_id: str) -> SeasonResponse:
        season = await repo.get_season(uuid.UUID(season_id))
        if season is None:
            raise HTTPException(status_code=404, detail="Season not found")
        return SeasonResponse(season=season)

    @router.get("/seasons/by-name/{name}")
    @timed_http_handler
    async def get_season_by_name(name: str) -> SeasonResponse:
        season = await repo.get_season_by_name(name)
        if season is None:
            raise HTTPException(status_code=404, detail="Season not found")
        return SeasonResponse(season=season)

    @router.get("/seasons/{season_id}/pools")
    @timed_http_handler
    async def get_pools_for_season(season_id: str) -> PoolsResponse:
        pools = await repo.get_pools_for_season(uuid.UUID(season_id))
        return PoolsResponse(pools=pools)

    @router.post("/pools")
    @timed_http_handler
    async def create_pool(request: PoolCreate, user: UserOrToken) -> UUIDResponse:
        pool_id = await repo.create_pool(
            referee_class=request.referee_class,
            season_id=request.season_id,
            name=request.name,
            attributes=request.attributes,
        )
        return UUIDResponse(id=pool_id)

    @router.get("/pools/{pool_id}")
    @timed_http_handler
    async def get_pool(pool_id: str) -> PoolResponse:
        pool = await repo.get_pool(uuid.UUID(pool_id))
        if pool is None:
            raise HTTPException(status_code=404, detail="Pool not found")
        return PoolResponse(pool=pool)

    @router.get("/pools/{pool_id}/players")
    @timed_http_handler
    async def get_pool_players(pool_id: str, include_removed: bool = False) -> PoolPlayersResponse:
        players = await repo.get_pool_players(uuid.UUID(pool_id), include_removed=include_removed)
        return PoolPlayersResponse(players=players)

    @router.post("/pools/{pool_id}/players")
    @timed_http_handler
    async def add_pool_player(pool_id: str, request: AddPoolPlayerRequest, user: UserOrToken) -> UUIDResponse:
        player_id = await repo.add_pool_player(
            policy_version_id=request.policy_version_id,
            pool_id=uuid.UUID(pool_id),
            attributes=request.attributes,
        )
        return UUIDResponse(id=player_id)

    @router.delete("/pools/{pool_id}/players/{policy_version_id}")
    @timed_http_handler
    async def remove_pool_player(pool_id: str, policy_version_id: str, user: UserOrToken) -> None:
        await repo.remove_pool_player(
            policy_version_id=uuid.UUID(policy_version_id),
            pool_id=uuid.UUID(pool_id),
        )

    @router.post("/pools/{pool_id}/players/{policy_version_id}/retire")
    @timed_http_handler
    async def retire_pool_player(pool_id: str, policy_version_id: str, user: UserOrToken) -> None:
        await repo.retire_pool_player(
            policy_version_id=uuid.UUID(policy_version_id),
            pool_id=uuid.UUID(pool_id),
        )

    @router.post("/matches")
    @timed_http_handler
    async def create_match(request: MatchCreate, user: UserOrToken) -> UUIDResponse:
        match_id = await repo.create_match(
            pool_id=request.pool_id,
            policy_version_ids=request.policy_version_ids,
        )
        return UUIDResponse(id=match_id)

    @router.get("/matches/{match_id}")
    @timed_http_handler
    async def get_match(match_id: str) -> MatchResponse:
        match = await repo.get_match(uuid.UUID(match_id))
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return MatchResponse(match=match)

    @router.get("/matches/by-eval-task/{eval_task_id}")
    @timed_http_handler
    async def get_match_by_eval_task(eval_task_id: int) -> MatchResponse:
        match = await repo.get_match_by_eval_task(eval_task_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return MatchResponse(match=match)

    @router.get("/matches/unscheduled")
    @timed_http_handler
    async def get_unscheduled_matches(limit: int = 100) -> UnscheduledMatchesResponse:
        matches = await repo.get_unscheduled_matches(limit=limit)
        return UnscheduledMatchesResponse(matches=matches)

    @router.get("/pools/{pool_id}/matches")
    @timed_http_handler
    async def get_matches_for_pool(
        pool_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> MatchesResponse:
        matches = await repo.get_matches_for_pool(
            pool_id=uuid.UUID(pool_id),
            limit=limit,
            offset=offset,
        )
        return MatchesResponse(matches=matches)

    @router.post("/matches/{match_id}/eval-task")
    @timed_http_handler
    async def set_match_eval_task(match_id: str, request: SetMatchEvalTaskRequest, user: UserOrToken) -> None:
        await repo.set_match_eval_task(uuid.UUID(match_id), request.eval_task_id)

    @router.get("/policies/{policy_version_id}/matches")
    @timed_http_handler
    async def get_policy_match_history(
        policy_version_id: str,
        pool_id: str | None = None,
        limit: int = 100,
    ) -> UnscheduledMatchesResponse:
        matches = await repo.get_match_history_for_policy(
            policy_version_id=uuid.UUID(policy_version_id),
            pool_id=uuid.UUID(pool_id) if pool_id else None,
            limit=limit,
        )
        return UnscheduledMatchesResponse(matches=matches)

    @router.post("/seasons/submit")
    @timed_http_handler
    async def submit_to_season(request: SubmitToSeasonRequest, user: UserOrToken) -> UUIDResponse:
        season = await repo.get_season(request.season_id)
        if season is None:
            raise HTTPException(status_code=404, detail="Season not found")

        pools = await repo.get_pools_for_season(request.season_id)
        if not pools:
            raise HTTPException(status_code=400, detail="Season has no pools configured")

        target_pool = pools[0]

        player_id = await repo.add_pool_player(
            policy_version_id=request.policy_version_id,
            pool_id=target_pool.id,
        )

        return UUIDResponse(id=player_id)

    return router

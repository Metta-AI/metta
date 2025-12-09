import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from metta.app_backend.auth import UserOrToken
from metta.app_backend.metta_repo import (
    MatchRow,
    MatchStatus,
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
    environment_config: dict[str, Any]
    policy_version_ids: list[uuid.UUID]
    attributes: dict[str, Any] = Field(default_factory=dict)


class MatchResponse(BaseModel):
    match: MatchWithPlayers


class MatchesResponse(BaseModel):
    matches: list[MatchRow]


class PendingMatchesResponse(BaseModel):
    matches: list[MatchWithPlayers]


class FinishMatchRequest(BaseModel):
    status: MatchStatus
    result: dict[str, Any] | None = None
    player_scores: dict[str, float] | None = None


class UploadPolicyRequest(BaseModel):
    name: str
    s3_path: str | None = None
    git_hash: str | None = None
    policy_spec: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)


class UploadPolicyResponse(BaseModel):
    policy_id: uuid.UUID
    policy_version_id: uuid.UUID
    academy_pool_player_id: uuid.UUID | None


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

    @router.get("/pools/academy")
    @timed_http_handler
    async def get_academy_pool() -> PoolResponse:
        pool = await repo.get_academy_pool()
        if pool is None:
            raise HTTPException(status_code=404, detail="Academy pool not found")
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
            environment_config=request.environment_config,
            policy_version_ids=request.policy_version_ids,
            attributes=request.attributes,
        )
        return UUIDResponse(id=match_id)

    @router.get("/matches/{match_id}")
    @timed_http_handler
    async def get_match(match_id: str) -> MatchResponse:
        match = await repo.get_match(uuid.UUID(match_id))
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return MatchResponse(match=match)

    @router.get("/matches/pending")
    @timed_http_handler
    async def get_pending_matches(limit: int = 100) -> PendingMatchesResponse:
        matches = await repo.get_pending_matches(limit=limit)
        return PendingMatchesResponse(matches=matches)

    @router.get("/pools/{pool_id}/matches")
    @timed_http_handler
    async def get_matches_for_pool(
        pool_id: str,
        status: MatchStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> MatchesResponse:
        matches = await repo.get_matches_for_pool(
            pool_id=uuid.UUID(pool_id),
            status=status,
            limit=limit,
            offset=offset,
        )
        return MatchesResponse(matches=matches)

    @router.post("/matches/{match_id}/start")
    @timed_http_handler
    async def start_match(match_id: str, user: UserOrToken) -> None:
        await repo.start_match(uuid.UUID(match_id))

    @router.post("/matches/{match_id}/finish")
    @timed_http_handler
    async def finish_match(match_id: str, request: FinishMatchRequest, user: UserOrToken) -> None:
        player_scores = None
        if request.player_scores:
            player_scores = {uuid.UUID(k): v for k, v in request.player_scores.items()}
        await repo.finish_match(
            match_id=uuid.UUID(match_id),
            status=request.status,
            result=request.result,
            player_scores=player_scores,
        )

    @router.get("/policies/{policy_version_id}/matches")
    @timed_http_handler
    async def get_policy_match_history(
        policy_version_id: str,
        pool_id: str | None = None,
        limit: int = 100,
    ) -> PendingMatchesResponse:
        matches = await repo.get_match_history_for_policy(
            policy_version_id=uuid.UUID(policy_version_id),
            pool_id=uuid.UUID(pool_id) if pool_id else None,
            limit=limit,
        )
        return PendingMatchesResponse(matches=matches)

    @router.post("/policies/upload")
    @timed_http_handler
    async def upload_policy(request: UploadPolicyRequest, user: UserOrToken) -> UploadPolicyResponse:
        policy_id = await repo.upsert_policy(
            name=request.name,
            user_id=user,
            attributes=request.attributes,
        )
        policy_version_id = await repo.create_policy_version(
            policy_id=policy_id,
            s3_path=request.s3_path,
            git_hash=request.git_hash,
            policy_spec=request.policy_spec,
            attributes=request.attributes,
        )

        academy_pool = await repo.get_academy_pool()
        academy_pool_player_id = None
        if academy_pool:
            academy_pool_player_id = await repo.add_pool_player(
                policy_version_id=policy_version_id,
                pool_id=academy_pool.id,
            )

        return UploadPolicyResponse(
            policy_id=policy_id,
            policy_version_id=policy_version_id,
            academy_pool_player_id=academy_pool_player_id,
        )

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

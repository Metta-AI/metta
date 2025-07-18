import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.query_logger import execute_query_and_log
from metta.app_backend.route_logger import timed_route


def _parse_filter_query(filter_query: str) -> str:
    """Parse a filter query string into SQL WHERE clause.

    Supports basic syntax like:
    - policy_name=test
    - created_at > '2023-01-01'
    - policy_name=test AND created_at > '2023-01-01'

    This is a simple implementation that handles common cases.
    """
    stripped = filter_query.strip()
    if stripped:
        return f"WHERE {stripped}"
    else:
        return "WHERE 1=1"


# Request/Response Models for Episode Tagging
class EpisodeTagRequest(BaseModel):
    episode_ids: List[str]
    tag: str


class EpisodeTagByFilterRequest(BaseModel):
    filter_query: str
    tag: str


class EpisodeTagResponse(BaseModel):
    episodes_affected: int


class EpisodeTagsResponse(BaseModel):
    tags_by_episode: Dict[str, List[str]]


class AllTagsResponse(BaseModel):
    tags: List[str]


# Models for Episode Filtering
class Episode(BaseModel):
    id: str
    created_at: str
    primary_policy_id: str
    eval_category: Optional[str]
    env_name: Optional[str]
    attributes: Dict[str, Any]
    # Policy information
    policy_name: Optional[str]
    # Training run information
    training_run_id: Optional[str]
    training_run_name: Optional[str]
    training_run_user_id: Optional[str]
    # Episode tags
    tags: List[str]


class EpisodeFilterResponse(BaseModel):
    episodes: List[Episode]
    total_count: int
    page: int
    page_size: int
    total_pages: int


def create_episode_router(stats_repo: MettaRepo) -> APIRouter:
    """Create an episode router with the given MettaRepo instance."""
    router = APIRouter(prefix="/episodes", tags=["episodes"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    # Episode filtering route
    @router.get("", response_model=EpisodeFilterResponse)
    @timed_route("filter_episodes")
    async def filter_episodes(
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=100),
        filter_query: str = Query(
            "", description="Filter expression like 'policy_name=test AND created_at > 2023-01-01'"
        ),
        user_or_token: str = user_or_token,
    ) -> EpisodeFilterResponse:
        """Filter episodes with pagination."""
        try:
            async with stats_repo.connect() as con:
                # Build the WHERE clause from the filter query
                where_clause = ""
                query_params = []

                where_clause = _parse_filter_query(filter_query)

                # Count total episodes
                count_query = f"""
                    SELECT COUNT(*)
                    FROM wide_episodes
                    {where_clause}
                """

                count_result = await execute_query_and_log(con, count_query, query_params, "count_episodes")
                total_count = count_result[0][0]

                # Calculate pagination
                offset = (page - 1) * page_size
                total_pages = (total_count + page_size - 1) // page_size

                # Main query using wide_episodes view
                episodes_query = f"""
                    SELECT
                        id, created_at, primary_policy_id,
                        eval_category, env_name, attributes,
                        policy_name,
                        training_run_id, training_run_name, training_run_user_id
                    FROM wide_episodes
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """

                episode_rows = await execute_query_and_log(
                    con, episodes_query, query_params + [page_size, offset], "get_episodes"
                )

                # Get episode IDs for tag lookup
                episode_ids = [row[0] for row in episode_rows]

                # Get tags for all episodes
                episode_tags = await stats_repo.get_episode_tags(episode_ids)

                # Build response
                episodes = []
                for row in episode_rows:
                    episode_id = str(row[0])
                    episode = Episode(
                        id=episode_id,
                        created_at=row[1].isoformat(),
                        primary_policy_id=str(row[2]),
                        eval_category=row[3],
                        env_name=row[4],
                        attributes=row[5] or {},
                        policy_name=row[6],
                        training_run_id=str(row[7]) if row[7] else None,
                        training_run_name=row[8],
                        training_run_user_id=row[9],
                        tags=episode_tags.get(episode_id, []),
                    )
                    episodes.append(episode)

                return EpisodeFilterResponse(
                    episodes=episodes, total_count=total_count, page=page, page_size=page_size, total_pages=total_pages
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to filter episodes: {str(e)}") from e

    # Episode tagging routes
    @router.post("/tags/add", response_model=EpisodeTagResponse)
    @timed_route("add_episode_tags")
    async def add_episode_tags(request: EpisodeTagRequest, user: str = user_or_token) -> EpisodeTagResponse:
        """Add a tag to multiple episodes."""
        try:
            # Convert string UUIDs to UUID objects
            episode_uuids = [uuid.UUID(episode_id) for episode_id in request.episode_ids]
            episodes_affected = await stats_repo.add_episode_tags(episode_ids=episode_uuids, tag=request.tag)
            return EpisodeTagResponse(episodes_affected=episodes_affected)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add episode tags: {str(e)}") from e

    @router.post("/tags/remove", response_model=EpisodeTagResponse)
    @timed_route("remove_episode_tags")
    async def remove_episode_tags(request: EpisodeTagRequest, user: str = user_or_token) -> EpisodeTagResponse:
        """Remove a tag from multiple episodes."""
        try:
            # Convert string UUIDs to UUID objects
            episode_uuids = [uuid.UUID(episode_id) for episode_id in request.episode_ids]
            episodes_affected = await stats_repo.remove_episode_tags(episode_ids=episode_uuids, tag=request.tag)
            return EpisodeTagResponse(episodes_affected=episodes_affected)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove episode tags: {str(e)}") from e

    @router.get("/tags", response_model=EpisodeTagsResponse)
    @timed_route("get_episode_tags")
    async def get_episode_tags(
        episode_ids: List[str] = Query(default=[]), user: str = user_or_token
    ) -> EpisodeTagsResponse:
        """Get all tags for the given episode UUIDs."""
        try:
            # Convert string UUIDs to UUID objects
            episode_uuids = [uuid.UUID(episode_id) for episode_id in episode_ids]
            tags_by_episode = await stats_repo.get_episode_tags(episode_ids=episode_uuids)
            return EpisodeTagsResponse(tags_by_episode=tags_by_episode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get episode tags: {str(e)}") from e

    @router.get("/tags/all", response_model=AllTagsResponse)
    @timed_route("get_all_episode_tags")
    async def get_all_episode_tags(user: str = user_or_token) -> AllTagsResponse:
        """Get all distinct tags that exist in the system."""
        try:
            tags = await stats_repo.get_all_episode_tags()
            return AllTagsResponse(tags=tags)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get all episode tags: {str(e)}") from e

    @router.post("/tags/add-by-filter", response_model=EpisodeTagResponse)
    @timed_route("add_episode_tags_by_filter")
    async def add_episode_tags_by_filter(
        request: EpisodeTagByFilterRequest, user: str = user_or_token
    ) -> EpisodeTagResponse:
        """Add a tag to all episodes matching a filter query."""
        try:
            async with stats_repo.connect() as con:
                # Build the WHERE clause from the filter query
                where_clause = ""
                query_params = []

                if request.filter_query.strip():
                    where_clause = f"WHERE {_parse_filter_query(request.filter_query)}"

                # Get all episode IDs that match the filter
                ids_query = f"""
                    SELECT id
                    FROM wide_episodes
                    {where_clause}
                """

                episode_rows = await execute_query_and_log(con, ids_query, query_params, "get_filtered_episode_ids")
                episode_uuids = [row[0] for row in episode_rows]  # Already UUID objects from DB

                if not episode_uuids:
                    return EpisodeTagResponse(episodes_affected=0)

                # Add tags to all matching episodes
                episodes_affected = await stats_repo.add_episode_tags(episode_ids=episode_uuids, tag=request.tag)
                return EpisodeTagResponse(episodes_affected=episodes_affected)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add episode tags by filter: {str(e)}") from e

    @router.post("/tags/remove-by-filter", response_model=EpisodeTagResponse)
    @timed_route("remove_episode_tags_by_filter")
    async def remove_episode_tags_by_filter(
        request: EpisodeTagByFilterRequest, user: str = user_or_token
    ) -> EpisodeTagResponse:
        """Remove a tag from all episodes matching a filter query."""
        try:
            async with stats_repo.connect() as con:
                # Build the WHERE clause from the filter query
                where_clause = ""
                query_params = []

                if request.filter_query.strip():
                    where_clause = f"WHERE {_parse_filter_query(request.filter_query)}"

                # Get all episode IDs that match the filter
                ids_query = f"""
                    SELECT id
                    FROM wide_episodes
                    {where_clause}
                """

                episode_rows = await execute_query_and_log(con, ids_query, query_params, "get_filtered_episode_ids")
                episode_uuids = [row[0] for row in episode_rows]  # Already UUID objects from DB

                if not episode_uuids:
                    return EpisodeTagResponse(episodes_affected=0)

                # Remove tags from all matching episodes
                episodes_affected = await stats_repo.remove_episode_tags(episode_ids=episode_uuids, tag=request.tag)
                return EpisodeTagResponse(episodes_affected=episodes_affected)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove episode tags by filter: {str(e)}") from e

    return router

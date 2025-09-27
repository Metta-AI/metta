import logging
import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from softmax.orchestrator.auth import create_user_or_token_dependency
from softmax.orchestrator.metta_repo import LeaderboardRow, MettaRepo
from softmax.orchestrator.route_logger import timed_route

# Set up logging for leaderboard routes
logger = logging.getLogger("leaderboard_routes")
logger.setLevel(logging.INFO)


class LeaderboardCreateOrUpdate(BaseModel):
    name: str
    evals: List[str]
    metric: str
    start_date: str


class LeaderboardResponse(BaseModel):
    id: str
    name: str
    user_id: str
    evals: List[str]
    metric: str
    start_date: str
    latest_episode: int
    created_at: str
    updated_at: str

    @classmethod
    def from_db(cls, leaderboard: LeaderboardRow) -> "LeaderboardResponse":
        return cls(
            id=str(leaderboard.id),
            name=leaderboard.name,
            user_id=leaderboard.user_id,
            evals=leaderboard.evals,
            metric=leaderboard.metric,
            start_date=leaderboard.start_date.isoformat(),
            latest_episode=leaderboard.latest_episode,
            created_at=leaderboard.created_at.isoformat(),
            updated_at=leaderboard.updated_at.isoformat(),
        )


class LeaderboardListResponse(BaseModel):
    leaderboards: List[LeaderboardResponse]


def create_leaderboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a leaderboard router with the given MettaRepo instance."""
    router = APIRouter(prefix="/leaderboards", tags=["leaderboards"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(dependency=create_user_or_token_dependency(metta_repo))

    @router.get("")
    @timed_route("list_leaderboards")
    async def list_leaderboards(user_id: str = user_or_token) -> LeaderboardListResponse:  # type: ignore[reportUnusedFunction]
        """List all leaderboards for the current user."""
        leaderboards = await metta_repo.list_leaderboards()
        return LeaderboardListResponse(
            leaderboards=[LeaderboardResponse.from_db(leaderboard) for leaderboard in leaderboards]
        )

    @router.get("/{leaderboard_id}")
    @timed_route("get_leaderboard")
    async def get_leaderboard(leaderboard_id: str, user_id: str = user_or_token) -> LeaderboardResponse:  # type: ignore[reportUnusedFunction]
        """Get a specific leaderboard by ID."""
        leaderboard = await metta_repo.get_leaderboard(uuid.UUID(leaderboard_id))
        if not leaderboard:
            raise HTTPException(status_code=404, detail="Leaderboard not found")

        return LeaderboardResponse.from_db(leaderboard)

    @router.post("")
    @timed_route("create_leaderboard")
    async def create_leaderboard(  # type: ignore[reportUnusedFunction]
        leaderboard_data: LeaderboardCreateOrUpdate,
        user_id: str = user_or_token,
    ) -> LeaderboardResponse:
        """Create a new leaderboard."""
        leaderboard_id = await metta_repo.create_leaderboard(
            name=leaderboard_data.name,
            user_id=user_id,
            evals=leaderboard_data.evals,
            metric=leaderboard_data.metric,
            start_date=leaderboard_data.start_date,
        )

        # Fetch the created leaderboard to return
        leaderboard = await metta_repo.get_leaderboard(leaderboard_id)
        if not leaderboard:
            raise HTTPException(status_code=500, detail="Failed to create leaderboard")

        return LeaderboardResponse.from_db(leaderboard)

    @router.put("/{leaderboard_id}")
    @timed_route("update_leaderboard")
    async def update_leaderboard(  # type: ignore[reportUnusedFunction]
        leaderboard_id: str, leaderboard_data: LeaderboardCreateOrUpdate, user_id: str = user_or_token
    ) -> LeaderboardResponse:
        """Update a leaderboard."""
        leaderboard = await metta_repo.update_leaderboard(
            leaderboard_id=uuid.UUID(leaderboard_id),
            user_id=user_id,
            name=leaderboard_data.name,
            evals=leaderboard_data.evals,
            metric=leaderboard_data.metric,
            start_date=leaderboard_data.start_date,
        )
        return LeaderboardResponse.from_db(leaderboard)

    @router.delete("/{leaderboard_id}")
    @timed_route("delete_leaderboard")
    async def delete_leaderboard(  # type: ignore[reportUnusedFunction]
        leaderboard_id: str, user_id: str = user_or_token
    ) -> Dict[str, str]:
        """Delete a leaderboard."""
        success = await metta_repo.delete_leaderboard(leaderboard_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Leaderboard not found")
        return {"message": "Leaderboard deleted successfully"}

    return router

from fastapi import APIRouter, Query
from pydantic import BaseModel

from metta.app_backend.leaderboard_constants import V0_LEADERBOARD_NAME_TAG_KEY
from metta.app_backend.metta_repo import LeaderboardEntry, MettaRepo


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


def create_leaderboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Attach leaderboard routes to the provided router."""
    router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

    @router.get("/", response_model=LeaderboardResponse)
    async def get_leaderboard(user_id: str | None = Query(default=None)) -> LeaderboardResponse:
        """Return leaderboard entries. Optionally filter by policy user id."""
        entries = await metta_repo.get_leaderboard_entries(
            leaderboard_tag_key=V0_LEADERBOARD_NAME_TAG_KEY,
            user_id=user_id,
        )
        return LeaderboardResponse(entries=entries)

    @router.get("/users/{requested_user_id}", response_model=LeaderboardResponse)
    async def get_user_leaderboard(requested_user_id: str) -> LeaderboardResponse:
        """Return leaderboard entries for a specific policy owner."""
        entries = await metta_repo.get_leaderboard_entries(
            leaderboard_tag_key=V0_LEADERBOARD_NAME_TAG_KEY,
            user_id=requested_user_id,
        )
        return LeaderboardResponse(entries=entries)

    return router

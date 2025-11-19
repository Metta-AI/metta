from fastapi import APIRouter, Depends
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.leaderboard_constants import V0_LEADERBOARD_NAME_TAG_KEY
from metta.app_backend.metta_repo import LeaderboardEntry, MettaRepo


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


def _sort_leaderboard_entries(entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    def average_score(entry: LeaderboardEntry) -> float:
        if not entry.scores:
            return float("-inf")
        values = list(entry.scores.values())
        return sum(values) / len(values)

    return sorted(entries, key=average_score, reverse=True)


def create_leaderboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Attach leaderboard routes to the provided router."""
    router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])
    user_or_token = Depends(create_user_or_token_dependency())

    @router.get("/", response_model=LeaderboardResponse)
    async def get_leaderboard(user: str = user_or_token) -> LeaderboardResponse:  # noqa: ARG001
        """Return leaderboard entries. Optionally filter by policy user id."""
        entries = await metta_repo.get_avg_per_agent_score_by_tag(
            tag_key=V0_LEADERBOARD_NAME_TAG_KEY,
            user_id=None,
            policy_version_id=None,
        )
        return LeaderboardResponse(entries=_sort_leaderboard_entries(entries))

    @router.get("/me", response_model=LeaderboardResponse)
    async def get_user_leaderboard(user: str = user_or_token) -> LeaderboardResponse:
        """Return leaderboard entries for a specific policy owner."""
        entries = await metta_repo.get_avg_per_agent_score_by_tag(
            tag_key=V0_LEADERBOARD_NAME_TAG_KEY,
            user_id=user,
            policy_version_id=None,
        )
        return LeaderboardResponse(entries=_sort_leaderboard_entries(entries))

    return router

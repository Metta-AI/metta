import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.leaderboard_constants import V0_LEADERBOARD_NAME_TAG_KEY
from metta.app_backend.metta_repo import LeaderboardEntry, LeaderboardPolicyEntry, MettaRepo


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]


class LeaderboardPoliciesRequest(BaseModel):
    policy_version_tags: dict[str, str]
    score_group_episode_tags: list[str]
    user_id: str | None = None
    policy_version_id: uuid.UUID | None = None


class LeaderboardPoliciesResponse(BaseModel):
    entries: list[LeaderboardPolicyEntry]


def average_score(entry: LeaderboardEntry) -> float:
    if not entry.scores:
        return float("-inf")
    values = list(entry.scores.values())
    return sum(values) / len(values)


def create_leaderboard_router(metta_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])
    user_or_token = Depends(create_user_or_token_dependency())

    async def _get_leaderboard_entries(
        user_id: str | None = None, policy_version_id: str | None = None
    ) -> LeaderboardResponse:
        entries = await metta_repo.get_avg_per_agent_score_by_tag(
            tag_key=V0_LEADERBOARD_NAME_TAG_KEY,
            user_id=user_id,
            policy_version_id=uuid.UUID(policy_version_id) if policy_version_id else None,
        )
        for entry in entries:
            entry.avg_score = average_score(entry)
        return LeaderboardResponse(entries=sorted(entries, key=lambda x: x.avg_score or 0, reverse=True))

    @router.get("/", response_model=LeaderboardResponse)
    async def get_leaderboard(user: str = user_or_token) -> LeaderboardResponse:
        return await _get_leaderboard_entries(user_id=None, policy_version_id=None)

    @router.get("/users/me", response_model=LeaderboardResponse)
    async def get_user_leaderboard(user: str = user_or_token) -> LeaderboardResponse:
        return await _get_leaderboard_entries(user_id=user, policy_version_id=None)

    @router.get("/policy/{policy_version_id}", response_model=LeaderboardResponse)
    async def get_leaderboard_for_policy(policy_version_id: str) -> LeaderboardResponse:
        return await _get_leaderboard_entries(user_id=None, policy_version_id=str(uuid.UUID(policy_version_id)))

    @router.post("/leaderboard_policies", response_model=LeaderboardPoliciesResponse)
    async def get_leaderboard_policies(
        request: LeaderboardPoliciesRequest, user: str = user_or_token
    ) -> LeaderboardPoliciesResponse:
        entries = await metta_repo.get_leaderboard_policies(
            policy_version_tags=request.policy_version_tags,
            score_group_episode_tags=request.score_group_episode_tags,
            user_id=request.user_id,
            policy_version_id=request.policy_version_id,
        )
        return LeaderboardPoliciesResponse(entries=entries)

    return router

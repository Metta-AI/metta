import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from metta.app_backend.auth import UserOrToken
from metta.app_backend.leaderboard_constants import (
    COGAMES_SUBMITTED_PV_KEY,
    LEADERBOARD_SIM_NAME_EPISODE_KEY,
    REPLACEMENT_BASELINE_MEAN,
)
from metta.app_backend.metta_repo import LeaderboardPolicyEntry, MettaRepo

# Note: /v2/vor returns batch VOR (cached 60s), /v2/vor/{id} returns detailed single-policy VOR


class LeaderboardPoliciesResponse(BaseModel):
    entries: list[LeaderboardPolicyEntry]


class BaselineStatusResponse(BaseModel):
    """Response containing replacement baseline status."""

    baseline_mean: float | None
    baseline_episodes: int
    using_fallback: bool
    fallback_value: float


def create_leaderboard_router(metta_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

    @router.get("/v2")
    async def get_leaderboard_policies_v2(user: UserOrToken) -> LeaderboardPoliciesResponse:
        return LeaderboardPoliciesResponse(
            entries=await metta_repo.get_leaderboard_policies(
                # TODO: consider a designated-as-public tag, not just if it was submitted
                policy_version_tags={COGAMES_SUBMITTED_PV_KEY: "true"},
                score_group_episode_tag=LEADERBOARD_SIM_NAME_EPISODE_KEY,
                user_id=None,
                policy_version_id=None,
            )
        )

    @router.get("/v2/users/me")
    async def get_leaderboard_policies_v2_for_user(user: UserOrToken) -> LeaderboardPoliciesResponse:
        return LeaderboardPoliciesResponse(
            entries=await metta_repo.get_leaderboard_policies(
                policy_version_tags={COGAMES_SUBMITTED_PV_KEY: "true"},
                score_group_episode_tag=LEADERBOARD_SIM_NAME_EPISODE_KEY,
                user_id=user,
                policy_version_id=None,
            )
        )

    @router.get("/v2/policy/{policy_version_id}")
    async def get_leaderboard_policies_v2_for_policy(
        policy_version_id: str, user: UserOrToken
    ) -> LeaderboardPoliciesResponse:
        return LeaderboardPoliciesResponse(
            entries=await metta_repo.get_leaderboard_policies(
                policy_version_tags={COGAMES_SUBMITTED_PV_KEY: "true"},
                score_group_episode_tag=LEADERBOARD_SIM_NAME_EPISODE_KEY,
                user_id=None,
                policy_version_id=uuid.UUID(policy_version_id),
            )
        )

    @router.get("/v2/vor")
    async def get_leaderboard_with_vor(user: UserOrToken) -> LeaderboardPoliciesResponse:
        """Get leaderboard entries with VOR computed for each policy (cached 60s)."""
        return LeaderboardPoliciesResponse(
            entries=await metta_repo.get_leaderboard_policies_with_vor(
                policy_version_tags={COGAMES_SUBMITTED_PV_KEY: "true"},
                score_group_episode_tag=LEADERBOARD_SIM_NAME_EPISODE_KEY,
            )
        )

    @router.get("/v2/baseline/status")
    async def get_baseline_status(user: UserOrToken) -> BaselineStatusResponse:
        """Get the current replacement baseline status.

        Returns whether the baseline is computed from actual c0 episodes or using
        the hardcoded fallback value. If using_fallback is True, you should run
        baseline evaluations via:
            ./tools/run.py recipes.experiment.v0_leaderboard.evaluate_baseline
        """
        baseline_mean, baseline_episodes = await metta_repo._get_replacement_baseline()
        return BaselineStatusResponse(
            baseline_mean=baseline_mean,
            baseline_episodes=baseline_episodes,
            using_fallback=baseline_mean is None,
            fallback_value=REPLACEMENT_BASELINE_MEAN,
        )

    return router

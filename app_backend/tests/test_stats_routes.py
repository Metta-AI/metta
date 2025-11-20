import uuid

import pytest

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import MettaRepo


async def _create_policy_version(
    stats_repo: MettaRepo,
    user_id: str,
    policy_name: str,
) -> uuid.UUID:
    policy_id = await stats_repo.upsert_policy(name=policy_name, user_id=user_id, attributes={})
    policy_version_id = await stats_repo.create_policy_version(
        policy_id=policy_id,
        s3_path=None,
        git_hash=None,
        policy_spec={},
        attributes={},
    )
    return policy_version_id


@pytest.mark.asyncio
async def test_leaderboard_v2_users_me_route_filters_by_user(
    isolated_stats_repo: MettaRepo,
    isolated_stats_client: StatsClient,
) -> None:
    policy_version_id = await _create_policy_version(isolated_stats_repo, "test_user@example.com", "policy1")

    response = isolated_stats_client.get_my_policy_versions()
    assert len(response.entries) == 1
    assert response.entries[0].id == policy_version_id

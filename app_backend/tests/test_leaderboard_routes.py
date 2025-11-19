import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.leaderboard_constants import V0_LEADERBOARD_NAME_TAG_KEY
from metta.app_backend.metta_repo import MettaRepo


async def _create_policy_with_scores(
    stats_repo: MettaRepo,
    user_id: str,
    policy_name: str,
    sim_scores: list[dict[str, float]],
) -> None:
    policy_id = await stats_repo.upsert_policy(name=policy_name, user_id=user_id, attributes={})
    policy_version_id = await stats_repo.create_policy_version(
        policy_id=policy_id,
        s3_path=None,
        git_hash=None,
        policy_spec={},
        attributes={},
    )

    for reward_l in sim_scores:
        for sim_name, reward in reward_l.items():
            await stats_repo.record_episode(
                id=uuid.uuid4(),
                data_uri=f"s3://episodes/{uuid.uuid4()}",
                primary_pv_id=policy_version_id,
                replay_url=None,
                attributes={},
                eval_task_id=None,
                thumbnail_url=None,
                tags=[(V0_LEADERBOARD_NAME_TAG_KEY, sim_name)],
                policy_versions=[(policy_version_id, 1)],
                policy_metrics=[(policy_version_id, "reward", reward)],
            )


@pytest.mark.asyncio
async def test_leaderboard_and_me_routes(
    isolated_stats_repo: MettaRepo,
    isolated_test_client: TestClient,
) -> None:
    user_one = "alice@example.com"
    user_two = "bob@example.com"

    await _create_policy_with_scores(
        isolated_stats_repo,
        user_one,
        "alice-policy",
        [
            {
                "arena-basic": 10.0,
                "arena-combat": 10.0,
            },
            {
                "arena-basic": 30.0,
                "arena-combat": 10.0,
            },
        ],
    )
    await _create_policy_with_scores(
        isolated_stats_repo,
        user_two,
        "bob-policy",
        [
            {
                "arena-basic": 4.0,
                "arena-combat": 1.0,
            }
        ],
    )

    headers = {"X-Auth-Request-Email": user_one}

    def avg_score(entry: dict) -> float:
        values = entry["scores"].values()
        return sum(values) / len(values)

    response = isolated_test_client.get("/leaderboard/", headers=headers)
    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [entry["user_id"] for entry in entries] == [user_one, user_two]
    assert entries[0]["scores"] == {
        "arena-basic": 20.0,
        "arena-combat": 10.0,
    }
    assert entries[1]["scores"] == {
        "arena-basic": 4.0,
        "arena-combat": 1.0,
    }
    assert avg_score(entries[0]) > avg_score(entries[1])

    me_response = isolated_test_client.get("/leaderboard/users/me", headers=headers)
    assert me_response.status_code == 200
    me_entries = me_response.json()["entries"]
    assert len(me_entries) == 1
    assert me_entries[0]["user_id"] == user_one
    assert me_entries[0]["scores"] == {
        "arena-basic": 20.0,
        "arena-combat": 10.0,
    }

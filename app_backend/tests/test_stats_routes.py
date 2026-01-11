import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.metta_repo import MettaRepo


@pytest.mark.asyncio
async def test_query_episodes_by_id_includes_avg_rewards_and_replay(
    isolated_stats_repo: MettaRepo,
    isolated_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    user = "episodes@example.com"
    policy_id = await isolated_stats_repo.upsert_policy(name="episodes-policy", user_id=user, attributes={})
    pv_id = await isolated_stats_repo.create_policy_version(
        policy_id=policy_id,
        s3_path=None,
        git_hash=None,
        policy_spec={},
        attributes={},
    )

    episode_id = uuid.uuid4()
    await isolated_stats_repo.record_episode(
        id=episode_id,
        data_uri=f"s3://episodes/{uuid.uuid4()}",
        primary_pv_id=pv_id,
        replay_url="https://example.com/replays/episode-test",
        attributes={"note": "avg reward should be computed"},
        eval_task_id=None,
        thumbnail_url=None,
        tags=[("sim_name", "arena-basic")],
        policy_versions=[(pv_id, 2)],
        policy_metrics=[(pv_id, "reward", 10.0)],
    )

    response = isolated_test_client.post(
        "/stats/episodes/query",
        json={"episode_ids": [str(episode_id)], "limit": 1},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert "episodes" in body
    episodes = body["episodes"]
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["id"] == str(episode_id)
    assert episode["replay_url"] == "https://example.com/replays/episode-test"
    assert episode["avg_rewards"][str(pv_id)] == pytest.approx(5.0)

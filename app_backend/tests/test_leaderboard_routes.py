import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.leaderboard_constants import (
    COGAMES_SUBMITTED_PV_KEY,
    LEADERBOARD_SIM_NAME_EPISODE_KEY,
)
from metta.app_backend.metta_repo import MettaRepo


async def _create_policy_with_scores(
    stats_repo: MettaRepo,
    user_id: str,
    policy_name: str,
    sim_scores: list[dict[str, float]],
) -> uuid.UUID:
    policy_id = await stats_repo.upsert_policy(name=policy_name, user_id=user_id, attributes={})
    policy_version_id = await stats_repo.create_policy_version(
        policy_id=policy_id,
        s3_path=None,
        git_hash=None,
        policy_spec={},
        attributes={},
    )

    for episode_idx, reward_l in enumerate(sim_scores):
        for sim_name, reward in reward_l.items():
            await stats_repo.record_episode(
                id=uuid.uuid4(),
                data_uri=f"s3://episodes/{uuid.uuid4()}",
                primary_pv_id=policy_version_id,
                replay_url=f"https://example.com/replays/{policy_name}/{sim_name}/{episode_idx}",
                attributes={},
                eval_task_id=None,
                thumbnail_url=None,
                tags=[(LEADERBOARD_SIM_NAME_EPISODE_KEY, sim_name)],
                policy_versions=[(policy_version_id, 1)],
                policy_metrics=[(policy_version_id, "reward", reward)],
            )

    return policy_version_id


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
async def test_leaderboard_v2_route_returns_tags_and_scores(
    isolated_stats_repo: MettaRepo,
    isolated_stats_client: StatsClient,
) -> None:
    user = "leaderboard@example.com"
    policy_version_id = await _create_policy_with_scores(
        isolated_stats_repo,
        user,
        "leaderboard-policy",
        [
            {
                "arena-basic": 10.0,
                "arena-combat": 20.0,
            },
            {
                "arena-basic": 30.0,
                "arena-combat": 0.0,
            },
        ],
    )
    await isolated_stats_repo.upsert_policy_version_tags(policy_version_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    empty_policy_version_id = await _create_policy_version(isolated_stats_repo, user, "pending-policy")
    await isolated_stats_repo.upsert_policy_version_tags(empty_policy_version_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    response = isolated_stats_client.get_leaderboard_policies_v2()
    entries = response.entries
    assert [entry.policy_version.id for entry in entries] == [policy_version_id, empty_policy_version_id]

    populated_entry = entries[0]
    expected_scores = {
        f"{LEADERBOARD_SIM_NAME_EPISODE_KEY}:arena-basic": 20.0,
        f"{LEADERBOARD_SIM_NAME_EPISODE_KEY}:arena-combat": 10.0,
    }
    assert populated_entry.scores == expected_scores
    assert populated_entry.avg_score == pytest.approx(15.0)
    assert populated_entry.policy_version.tags == {COGAMES_SUBMITTED_PV_KEY: "true"}
    assert populated_entry.policy_version.user_id == user
    assert set(populated_entry.score_episode_ids.keys()) == set(expected_scores.keys())
    assert all(populated_entry.score_episode_ids.values())

    empty_entry = entries[1]
    assert empty_entry.scores == {}
    assert empty_entry.avg_score is None
    assert empty_entry.policy_version.tags == {COGAMES_SUBMITTED_PV_KEY: "true"}
    assert empty_entry.policy_version.user_id == user
    assert empty_entry.score_episode_ids == {}


@pytest.mark.asyncio
async def test_leaderboard_v2_filters_by_policy_version_id(
    isolated_stats_repo: MettaRepo,
    isolated_stats_client: StatsClient,
) -> None:
    user = "policy-filter@example.com"
    matching_pv_id = await _create_policy_with_scores(
        isolated_stats_repo,
        user,
        "filter-policy",
        [
            {
                "arena-basic": 5.0,
            }
        ],
    )
    await isolated_stats_repo.upsert_policy_version_tags(matching_pv_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    other_pv_id = await _create_policy_with_scores(
        isolated_stats_repo,
        user,
        "other-policy",
        [
            {
                "arena-basic": 9.0,
            }
        ],
    )
    await isolated_stats_repo.upsert_policy_version_tags(other_pv_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    response = isolated_stats_client.get_leaderboard_policies_v2_for_policy(matching_pv_id)
    assert len(response.entries) == 1
    assert response.entries[0].policy_version.id == matching_pv_id
    assert response.entries[0].policy_version.user_id == user


@pytest.mark.asyncio
async def test_leaderboard_v2_users_me_route_filters_by_user(
    isolated_stats_repo: MettaRepo,
    isolated_stats_client: StatsClient,
) -> None:
    from unittest import mock

    user = "policy-owner@example.com"
    other_user = "other@example.com"
    owned_pv_id = await _create_policy_with_scores(
        isolated_stats_repo,
        user,
        "owned-policy",
        [
            {
                "arena-basic": 8.0,
            }
        ],
    )
    await isolated_stats_repo.upsert_policy_version_tags(owned_pv_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    other_pv_id = await _create_policy_with_scores(
        isolated_stats_repo,
        other_user,
        "other-policy",
        [
            {
                "arena-basic": 14.0,
            }
        ],
    )
    await isolated_stats_repo.upsert_policy_version_tags(other_pv_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    with mock.patch("metta.app_backend.config.settings.DEBUG_USER_EMAIL", user):
        response = isolated_stats_client.get_leaderboard_policies_v2_users_me()
        entries = response.entries
        assert len(entries) == 1
        assert entries[0].policy_version.id == owned_pv_id
        assert entries[0].policy_version.user_id == user


@pytest.mark.asyncio
async def test_query_episodes_filters_by_primary_and_tag(
    isolated_stats_repo: MettaRepo,
    isolated_test_client: TestClient,
) -> None:
    user = "episodes@example.com"
    policy_version_id = await _create_policy_with_scores(
        isolated_stats_repo,
        user,
        "episodes-policy",
        [
            {
                "arena-basic": 1.0,
                "arena-combat": 2.0,
            },
            {
                "arena-basic": 3.0,
                "arena-combat": 4.0,
            },
        ],
    )

    response = isolated_test_client.post(
        "/stats/episodes/query",
        json={
            "primary_policy_version_ids": [str(policy_version_id)],
            "tag_filters": {LEADERBOARD_SIM_NAME_EPISODE_KEY: ["arena-basic"]},
            "limit": 1,
        },
    )
    assert response.status_code == 200
    episodes = response.json()["episodes"]
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["primary_pv_id"] == str(policy_version_id)
    assert episode["replay_url"].endswith("/1")
    assert episode["tags"][LEADERBOARD_SIM_NAME_EPISODE_KEY] == "arena-basic"

    # Offset should work even when limit is None
    offset_response = isolated_test_client.post(
        "/stats/episodes/query",
        json={
            "primary_policy_version_ids": [str(policy_version_id)],
            "tag_filters": {LEADERBOARD_SIM_NAME_EPISODE_KEY: ["arena-basic"]},
            "limit": None,
            "offset": 1,
        },
    )
    assert offset_response.status_code == 200
    offset_episodes = offset_response.json()["episodes"]
    assert len(offset_episodes) == 1
    assert offset_episodes[0]["replay_url"].endswith("/0")

import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.leaderboard_constants import SUBMITTED_KEY, V0_LEADERBOARD_NAME_TAG_KEY
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
    await isolated_stats_repo.upsert_policy_version_tags(policy_version_id, {SUBMITTED_KEY: "true"})

    empty_policy_version_id = await _create_policy_version(isolated_stats_repo, user, "pending-policy")
    await isolated_stats_repo.upsert_policy_version_tags(empty_policy_version_id, {SUBMITTED_KEY: "true"})

    response = isolated_stats_client.get_leaderboard_policies_v2()
    entries = response.entries
    assert [entry.policy_version.id for entry in entries] == [policy_version_id, empty_policy_version_id]

    populated_entry = entries[0]
    expected_scores = {
        f"{V0_LEADERBOARD_NAME_TAG_KEY}:arena-basic": 20.0,
        f"{V0_LEADERBOARD_NAME_TAG_KEY}:arena-combat": 10.0,
    }
    assert populated_entry.scores == expected_scores
    assert populated_entry.avg_score == pytest.approx(15.0)
    assert populated_entry.policy_version.tags == {SUBMITTED_KEY: "true"}
    assert populated_entry.policy_version.user_id == user

    empty_entry = entries[1]
    assert empty_entry.scores == {}
    assert empty_entry.avg_score is None
    assert empty_entry.policy_version.tags == {SUBMITTED_KEY: "true"}
    assert empty_entry.policy_version.user_id == user


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
    await isolated_stats_repo.upsert_policy_version_tags(matching_pv_id, {SUBMITTED_KEY: "true"})

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
    await isolated_stats_repo.upsert_policy_version_tags(other_pv_id, {SUBMITTED_KEY: "true"})

    response = isolated_stats_client.get_leaderboard_policies_v2_for_policy(matching_pv_id)
    assert len(response.entries) == 1
    assert response.entries[0].policy_version.id == matching_pv_id
    assert response.entries[0].policy_version.user_id == user


@pytest.mark.asyncio
async def test_leaderboard_v2_users_me_route_filters_by_user(
    isolated_stats_repo: MettaRepo,
    isolated_stats_client: StatsClient,
) -> None:
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
    await isolated_stats_repo.upsert_policy_version_tags(owned_pv_id, {SUBMITTED_KEY: "true"})

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
    await isolated_stats_repo.upsert_policy_version_tags(other_pv_id, {SUBMITTED_KEY: "true"})

    isolated_stats_client._test_user_email = user  # type: ignore[attr-defined]
    response = isolated_stats_client.get_leaderboard_policies_v2_users_me()
    entries = response.entries
    assert len(entries) == 1
    assert entries[0].policy_version.id == owned_pv_id
    assert entries[0].policy_version.user_id == user

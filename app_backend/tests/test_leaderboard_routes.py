import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.leaderboard_constants import (
    COGAMES_SUBMITTED_PV_KEY,
    LADYBUG_UUID,
    LEADERBOARD_CANDIDATE_COUNT_KEY,
    LEADERBOARD_LADYBUG_COUNT_KEY,
    LEADERBOARD_SCENARIO_KEY,
    LEADERBOARD_SIM_NAME_EPISODE_KEY,
    LEADERBOARD_THINKY_COUNT_KEY,
    THINKY_UUID,
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

    with mock.patch("metta.app_backend.config.debug_user_email", user):
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


async def _create_vor_episode(
    stats_repo: MettaRepo,
    primary_pv_id: uuid.UUID,
    policy_version_id: uuid.UUID,
    reward: float,
    num_agents: int,
    candidate_count: int,
    thinky_count: int,
    ladybug_count: int,
    scenario_name: str,
) -> uuid.UUID:
    """Create an episode with VOR-specific tags for testing."""
    episode_id = uuid.uuid4()
    await stats_repo.record_episode(
        id=episode_id,
        data_uri=f"s3://episodes/{episode_id}",
        primary_pv_id=primary_pv_id,
        replay_url=f"https://example.com/replays/{episode_id}",
        attributes={},
        eval_task_id=None,
        thumbnail_url=None,
        tags=[
            (LEADERBOARD_SCENARIO_KEY, scenario_name),
            (LEADERBOARD_CANDIDATE_COUNT_KEY, str(candidate_count)),
            (LEADERBOARD_THINKY_COUNT_KEY, str(thinky_count)),
            (LEADERBOARD_LADYBUG_COUNT_KEY, str(ladybug_count)),
            (LEADERBOARD_SIM_NAME_EPISODE_KEY, scenario_name),
        ],
        policy_versions=[(policy_version_id, num_agents)],
        policy_metrics=[(policy_version_id, "reward", reward * num_agents)],  # Total reward
    )
    return episode_id


@pytest.mark.asyncio
async def test_leaderboard_with_vor_computes_correct_vor(
    isolated_stats_repo: MettaRepo,
    isolated_stats_client: StatsClient,
) -> None:
    """Test that VOR is correctly computed as candidate_avg - replacement_avg.

    VOR = Value Over Replacement
    - Candidate avg: weighted average of candidate policy rewards
    - Replacement avg: weighted average of baseline (Thinky/Ladybug) rewards when no candidate plays

    Setup:
    - Baseline episodes (candidate_count=0): Thinky gets reward 40, Ladybug gets reward 30
      -> Replacement avg = (40*4 + 30*4) / 8 = 35.0
    - Candidate episodes (candidate_count=2): Candidate gets reward 60
      -> Candidate avg = 60.0
    - Expected VOR = 60.0 - 35.0 = 25.0
    """
    thinky_pv_id = uuid.UUID(THINKY_UUID)
    ladybug_pv_id = uuid.UUID(LADYBUG_UUID)

    # Create Thinky and Ladybug policies with exact UUIDs (required by VOR calculation)
    async with isolated_stats_repo.connect() as con:
        # Create Thinky policy and version with exact UUID
        await con.execute(
            "INSERT INTO policies (id, name, user_id, attributes) VALUES (%s, %s, %s, %s)",
            (uuid.uuid4(), "thinky", "system@metta.ai", "{}"),
        )
        thinky_policy_result = await con.execute("SELECT id FROM policies WHERE name = 'thinky'")
        thinky_policy_row = await thinky_policy_result.fetchone()
        thinky_policy_id = thinky_policy_row[0]

        await con.execute(
            "INSERT INTO policy_versions (id, policy_id, version, s3_path, git_hash, policy_spec, attributes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (thinky_pv_id, thinky_policy_id, 1, None, None, "{}", "{}"),
        )

        # Create Ladybug policy and version with exact UUID
        await con.execute(
            "INSERT INTO policies (id, name, user_id, attributes) VALUES (%s, %s, %s, %s)",
            (uuid.uuid4(), "ladybug", "system@metta.ai", "{}"),
        )
        ladybug_policy_result = await con.execute("SELECT id FROM policies WHERE name = 'ladybug'")
        ladybug_policy_row = await ladybug_policy_result.fetchone()
        ladybug_policy_id = ladybug_policy_row[0]

        await con.execute(
            "INSERT INTO policy_versions (id, policy_id, version, s3_path, git_hash, policy_spec, attributes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (ladybug_pv_id, ladybug_policy_id, 1, None, None, "{}", "{}"),
        )

    # Create candidate policy using normal API
    candidate_policy_id = await isolated_stats_repo.upsert_policy(
        name="candidate-policy", user_id="test@example.com", attributes={}
    )
    candidate_pv_id = await isolated_stats_repo.create_policy_version(
        policy_id=candidate_policy_id,
        s3_path=None,
        git_hash=None,
        policy_spec={},
        attributes={},
    )
    await isolated_stats_repo.upsert_policy_version_tags(candidate_pv_id, {COGAMES_SUBMITTED_PV_KEY: "true"})

    # Create baseline episode: Thinky self-play (candidate_count=0)
    # 4 Thinky agents, each gets reward 40 -> total reward 160
    await _create_vor_episode(
        isolated_stats_repo,
        primary_pv_id=thinky_pv_id,
        policy_version_id=thinky_pv_id,
        reward=40.0,
        num_agents=4,
        candidate_count=0,
        thinky_count=4,
        ladybug_count=0,
        scenario_name="thinky_self_play",
    )

    # Create baseline episode: Ladybug self-play (candidate_count=0)
    # 4 Ladybug agents, each gets reward 30 -> total reward 120
    await _create_vor_episode(
        isolated_stats_repo,
        primary_pv_id=ladybug_pv_id,
        policy_version_id=ladybug_pv_id,
        reward=30.0,
        num_agents=4,
        candidate_count=0,
        thinky_count=0,
        ladybug_count=4,
        scenario_name="ladybug_self_play",
    )

    # Create candidate episode: 2 candidates + 2 Thinky (candidate_count=2)
    # 2 Candidate agents, each gets reward 60 -> total reward 120
    await _create_vor_episode(
        isolated_stats_repo,
        primary_pv_id=candidate_pv_id,
        policy_version_id=candidate_pv_id,
        reward=60.0,
        num_agents=2,
        candidate_count=2,
        thinky_count=2,
        ladybug_count=0,
        scenario_name="candidate_mix",
    )

    # Call the VOR endpoint
    response = isolated_stats_client.get_leaderboard_policies_with_vor()
    entries = response.entries

    # Find the candidate entry
    candidate_entry = next((e for e in entries if e.policy_version.id == candidate_pv_id), None)
    assert candidate_entry is not None, "Candidate policy should be in leaderboard"

    # Verify VOR calculation:
    # Replacement avg = (40*4 + 30*4) / (4+4) = 280/8 = 35.0
    # Candidate avg = 60.0 (only one episode, weight=2, mean=60)
    # VOR = 60.0 - 35.0 = 25.0
    assert candidate_entry.overall_vor is not None, "VOR should be computed"
    assert candidate_entry.overall_vor == pytest.approx(25.0, rel=0.01), (
        f"VOR should be ~25.0 (60.0 - 35.0), got {candidate_entry.overall_vor}"
    )

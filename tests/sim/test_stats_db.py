"""
Integration tests for the extended StatsDB in metta.sim.stats_db,
especially merge_worker_dbs() and insert_agent_policies().
"""

import tempfile
import uuid
from pathlib import Path

import pytest

from metta.sim.stats_db import StatsDB  # <— the Metta extension
from mettagrid.stats_writer import StatsDB as MGStatsDB


def _create_worker_db(path: Path, env_name: str):
    """
    Spin up a tiny StatsDB shard with one episode row.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a fresh database
    db = MGStatsDB(str(path), read_only=False)

    # Create an episode with a UUID
    episode_id = str(uuid.uuid4())  # Generate UUID directly

    # Make sure episodes table exists
    db.con.execute("""
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id VARCHAR PRIMARY KEY,
        env_name VARCHAR,
        seed INTEGER,
        map_w INTEGER,
        map_h INTEGER,
        step_count INTEGER,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        metadata VARCHAR
    )
    """)

    # Make sure agent_metrics table exists
    db.con.execute("""
    CREATE TABLE IF NOT EXISTS agent_metrics (
        episode_id VARCHAR,
        agent_id INTEGER,
        metric VARCHAR,
        value REAL,
        PRIMARY KEY (episode_id, agent_id, metric)
    )
    """)

    # Insert one row so we can see it after the merge
    db.con.execute(
        """
        INSERT INTO episodes
        (episode_id, env_name, seed, map_w, map_h,
         step_count, started_at, finished_at, metadata)
        VALUES (?, ?, 0, 1, 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
        """,
        (episode_id, env_name),
    )

    # Add some metrics
    db.con.execute(
        """
        INSERT INTO agent_metrics
        (episode_id, agent_id, metric, value)
        VALUES (?, 0, 'reward', 1.0)
        """,
        (episode_id,),
    )

    db.close()
    return episode_id


def test_merge_worker_dbs_two_shards(tmp_path: Path):
    """Test merging two worker database shards."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()

    # Create two independent worker shards
    episode_id1 = _create_worker_db(shards_dir / "worker0.duckdb", "env_a")
    episode_id2 = _create_worker_db(shards_dir / "worker1.duckdb", "env_b")

    # Dummy agent-id → policy map
    agent_map = {0: ("dummy_policy", None)}

    # Merge the shards
    merged: StatsDB = StatsDB.merge_worker_dbs(shards_dir, agent_map)

    # Should contain exactly the two rows we inserted
    episodes = merged.con.execute("SELECT episode_id, env_name FROM episodes ORDER BY env_name").fetchall()

    assert len(episodes) == 2
    # Check environments are as expected
    env_names = [episode[1] for episode in episodes]
    assert "env_a" in env_names
    assert "env_b" in env_names

    # agent_policies should be populated
    agent_meta = merged.con.execute(
        "SELECT episode_id, agent_id, policy_key, policy_version FROM agent_policies"
    ).fetchall()
    # We should have one row per episode
    assert len(agent_meta) == 2

    # Metrics should be merged
    metrics = merged.con.execute("SELECT episode_id, agent_id, metric, value FROM agent_metrics").fetchall()
    assert len(metrics) == 2
    assert metrics[0][2] == "reward"  # Check metric name
    assert metrics[1][2] == "reward"  # Check metric name

    merged.close()


def test_insert_agent_policies():
    """Test inserting agent policies with multiple episodes."""
    db = StatsDB(Path(tempfile.mktemp(suffix=".duckdb")), mode="rwc")

    # Multiple episode IDs
    episode_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    # Agent map
    agent_map = {
        0: ("policy_a", "v1"),
        1: ("policy_b", None),
    }

    # Insert agent policies for multiple episodes
    db.insert_agent_policies(episode_ids, agent_map)

    # Verify policies table contains unique policies
    policies = db.con.execute("SELECT policy_key, policy_version FROM policies ORDER BY policy_key").fetchall()
    assert len(policies) == 2
    assert policies[0][0] == "policy_a"
    assert policies[0][1] == "v1"
    assert policies[1][0] == "policy_b"
    assert policies[1][1] is None

    # Verify agent_policies contains multiplexed entries
    agent_policies = db.con.execute(
        "SELECT episode_id, agent_id, policy_key, policy_version FROM agent_policies ORDER BY episode_id, agent_id"
    ).fetchall()

    # Should have len(episode_ids) * len(agent_map) rows
    assert len(agent_policies) == len(episode_ids) * len(agent_map)

    # Check specific entries
    expected_entries = []
    for ep_id in episode_ids:
        for agent_id, (policy_key, policy_version) in agent_map.items():
            expected_entries.append((ep_id, agent_id, policy_key, policy_version))

    # Sort expected entries the same way as the query results
    expected_entries.sort(key=lambda x: (x[0], x[1]))

    for i, entry in enumerate(expected_entries):
        assert agent_policies[i][0] == entry[0]  # episode_id
        assert agent_policies[i][1] == entry[1]  # agent_id
        assert agent_policies[i][2] == entry[2]  # policy_key
        assert agent_policies[i][3] == entry[3]  # policy_version

    db.close()


def test_empty_inputs_to_insert_agent_policies():
    """Test handling of empty inputs to insert_agent_policies."""
    db = StatsDB(Path(tempfile.mktemp(suffix=".duckdb")), mode="rwc")

    # No episodes
    db.insert_agent_policies([], {0: ("policy_a", "v1")})

    # No agents
    db.insert_agent_policies([str(uuid.uuid4())], {})

    # Both empty
    db.insert_agent_policies([], {})

    # Verify no entries were added
    agent_policies_count = db.con.execute("SELECT COUNT(*) FROM agent_policies").fetchone()[0]
    assert agent_policies_count == 0

    db.close()


def test_merge_worker_dbs_empty_episodes(tmp_path: Path):
    """Test handling the case when no episodes are found in database."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()

    # Create an empty shard
    empty_db_path = shards_dir / "empty.duckdb"
    db = StatsDB(empty_db_path, mode="rwc")
    db.close()

    # Dummy agent map
    agent_map = {0: ("dummy_policy", None)}

    # Merge should handle empty episodes gracefully
    merged = StatsDB.merge_worker_dbs(shards_dir, agent_map)

    # Verify no episodes were found
    episode_count = merged.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert episode_count == 0

    merged.close()


def test_query_method():
    """Test the query method."""
    db = StatsDB(Path(tempfile.mktemp(suffix=".duckdb")), mode="rwc")

    # Create some test data
    episode_id = db.get_next_episode_id()
    db.con.execute(
        """
        INSERT INTO episodes
        (episode_id, env_name, seed, map_w, map_h,
         step_count, started_at, finished_at, metadata)
        VALUES (?, 'test_env', 123, 10, 10, 100, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{"key":"value"}')
        """,
        (episode_id,),
    )

    # Test running a query
    result = db.query("SELECT * FROM episodes")

    assert len(result) == 1
    assert result.loc[0, "episode_id"] == episode_id
    assert result.loc[0, "env_name"] == "test_env"
    assert result.loc[0, "seed"] == 123

    db.close()


def test_get_metrics_for_episode():
    """Test retrieving all metrics for a specific episode."""
    db = StatsDB(Path(tempfile.mktemp(suffix=".duckdb")), mode="rwc")

    # Create an episode
    episode_id = db.get_next_episode_id()
    db.con.execute(
        """
        INSERT INTO episodes
        (episode_id, env_name, seed, map_w, map_h,
         step_count, started_at, finished_at, metadata)
        VALUES (?, 'test_env', 0, 1, 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
        """,
        (episode_id,),
    )

    # Add metrics for multiple agents
    db.con.execute(
        """
        INSERT INTO agent_metrics
        (episode_id, agent_id, metric, value)
        VALUES 
        (?, 0, 'reward', 10.5),
        (?, 0, 'steps', 100),
        (?, 1, 'reward', 8.2),
        (?, 1, 'health', 75.0)
        """,
        (episode_id, episode_id, episode_id, episode_id),
    )

    # Test get_metrics_for_episode
    metrics = db.get_metrics_for_episode(episode_id)

    assert 0 in metrics
    assert 1 in metrics
    assert metrics[0]["reward"] == pytest.approx(10.5)
    assert metrics[0]["steps"] == 100
    assert metrics[1]["reward"] == pytest.approx(8.2)
    assert metrics[1]["health"] == pytest.approx(75.0)

    db.close()

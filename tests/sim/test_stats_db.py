"""
Integration tests for the extended StatsDB in metta.sim.stats_db,
especially merge_worker_dbs().
"""

import json
import tempfile
import uuid
from pathlib import Path

import duckdb
import pytest

from mettagrid.stats_writer import StatsDB as MGStatsDB
from metta.sim.stats_db import StatsDB  # <— the Metta extension


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

    # Make sure episode_agent_metrics table exists
    db.con.execute("""
    CREATE TABLE IF NOT EXISTS episode_agent_metrics (
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
        INSERT INTO episode_agent_metrics
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

    # Agent_metadata should be populated
    agent_meta = merged.con.execute("SELECT * FROM agent_metadata").fetchall()
    assert len(agent_meta) == len(agent_map)
    assert agent_meta[0][0] == "dummy_policy"

    # Metrics should be merged
    metrics = merged.con.execute("SELECT episode_id, agent_id, metric, value FROM episode_agent_metrics").fetchall()
    assert len(metrics) == 2
    assert metrics[0][2] == "reward"  # Check metric name
    assert metrics[1][2] == "reward"  # Check metric name

    merged.close()


def test_upsert_agent_metadata():
    """Test upserting agent metadata."""
    db = StatsDB(Path(tempfile.mktemp(suffix=".duckdb")), mode="rwc")

    # Insert initial metadata
    agent_map1 = {
        0: ("policy_a", "v1"),
        1: ("policy_b", None),
    }
    db.upsert_agent_metadata(agent_map1)

    # Verify it was stored correctly
    results = db.con.execute("SELECT policy_key, policy_version FROM agent_metadata ORDER BY policy_key").fetchall()

    assert len(results) == 2
    assert results[0][0] == "policy_a"
    assert results[0][1] == "v1"
    assert results[1][0] == "policy_b"
    assert results[1][1] is None

    # Update with new metadata
    agent_map2 = {
        0: ("policy_a", "v2"),  # Update version
        2: ("policy_c", "v1"),  # New policy
    }
    db.upsert_agent_metadata(agent_map2)

    # Verify updates were applied
    results = db.con.execute("SELECT policy_key, policy_version FROM agent_metadata ORDER BY policy_key").fetchall()

    assert len(results) == 3
    assert results[0][0] == "policy_a"
    assert results[0][1] == "v2"  # Updated version
    assert results[1][0] == "policy_b"
    assert results[1][1] is None
    assert results[2][0] == "policy_c"
    assert results[2][1] == "v1"

    db.close()


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
        INSERT INTO episode_agent_metrics
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

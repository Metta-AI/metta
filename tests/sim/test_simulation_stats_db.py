from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Tuple

from metta.sim.simulation_stats_db import SimulationStatsDB


class MockPolicyRecord:
    """Mock implementation of PolicyRecord for testing."""

    def __init__(self, policy_key: str, policy_version: int):
        self._policy_key = policy_key
        self._policy_version = policy_version

    def key_and_version(self) -> Tuple[str, int]:
        """Return the policy key and version as a tuple."""
        return self._policy_key, self._policy_version


def _create_worker_db(path: Path, sim_steps: int = 0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SimulationStatsDB(path)

    episode_id = str(uuid.uuid4())
    attributes = {"seed": "0", "map_w": "1", "map_h": "1"}
    groups = [[0]]
    agent_metrics = {0: {"reward": 1.0}}
    group_metrics = {0: {"reward": 1.0}}
    replay_url = None
    created_at = datetime.datetime.now()

    db.record_episode(
        episode_id,
        attributes,
        groups,
        agent_metrics,
        group_metrics,
        sim_steps,
        replay_url,
        created_at,
    )

    db.close()
    return episode_id


_DUMMY_AGENT_MAP = {0: ("dummy_policy", 0)}


def test_from_uri_context_manager(tmp_path: Path):
    db_path = tmp_path / "test_db.duckdb"
    ep_id = _create_worker_db(db_path)

    with SimulationStatsDB.from_uri(str(db_path)) as db:
        episode_ids = [row[0] for row in db.con.execute("SELECT id FROM episodes").fetchall()]
        assert ep_id in episode_ids


def test_insert_agent_policies(tmp_path: Path):
    db_path = tmp_path / "policies.duckdb"
    db = SimulationStatsDB(db_path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    episode_id = str(uuid.uuid4())
    db.con.execute("INSERT INTO episodes (id) VALUES (?)", (episode_id,))

    agent_map = {0: ("policy_a", 1), 1: ("policy_b", 0)}
    db._insert_agent_policies([episode_id], agent_map)

    rows = db.con.execute("SELECT * FROM agent_policies").fetchall()
    assert len(rows) == 2

    db.close()


def test_insert_agent_policies_empty_inputs(tmp_path: Path):
    db_path = tmp_path / "empty_policies.duckdb"
    db = SimulationStatsDB(db_path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    db._insert_agent_policies([], _DUMMY_AGENT_MAP)
    db._insert_agent_policies([str(uuid.uuid4())], {})

    count = db.con.execute("SELECT COUNT(*) FROM agent_policies").fetchone()[0]
    assert count == 0

    db.close()


def test_merge_in(tmp_path: Path):
    db1_path = tmp_path / "db1.duckdb"
    db2_path = tmp_path / "db2.duckdb"

    ep1 = _create_worker_db(db1_path)
    ep2 = _create_worker_db(db2_path)

    db1 = SimulationStatsDB(db1_path)
    db2 = SimulationStatsDB(db2_path)

    db1.merge_in(db2)

    episodes = db1.con.execute("SELECT id FROM episodes").fetchall()
    episode_ids = [row[0] for row in episodes]
    assert len(episode_ids) == 2
    assert sorted(episode_ids) == sorted([ep1, ep2])

    db1.close()
    db2.close()


def test_tables(tmp_path: Path):
    db_path = tmp_path / "tables.duckdb"
    db = SimulationStatsDB(db_path)

    tables = db.tables()
    assert "episodes" in tables
    assert "agent_metrics" in tables
    assert "simulations" in tables
    assert "agent_policies" in tables

    db.close()


def test_insert_simulation(tmp_path: Path):
    db_path = tmp_path / "sim_table.duckdb"
    db = SimulationStatsDB(db_path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    sim_id = str(uuid.uuid4())
    policy_key = "test_policy"
    policy_version = 1
    db._insert_simulation(sim_id, "test_sim", "test_suite", "test_env", policy_key, policy_version)

    rows = db.con.execute("SELECT id, name, suite, env, policy_key, policy_version FROM simulations").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == sim_id
    assert rows[0][1] == "test_sim"
    assert rows[0][2] == "test_suite"
    assert rows[0][3] == "test_env"
    assert rows[0][4] == policy_key
    assert rows[0][5] == policy_version

    db.close()


def test_get_replay_urls(tmp_path: Path):
    """Test retrieving replay URLs with various filters."""
    db_path = tmp_path / "replay_urls.duckdb"
    db = SimulationStatsDB(db_path)

    # Create tables
    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    # Add the simulation_id column to episodes if it doesn't exist
    db.con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS simulation_id TEXT")

    # Create a few episodes with replay URLs
    episodes = []
    replay_urls = [
        "https://example.com/replay1.json",
        "https://example.com/replay2.json",
        "https://example.com/replay3.json",
    ]

    # Create 3 episodes with different replay URLs
    for i in range(3):
        episode_id = str(uuid.uuid4())
        episodes.append(episode_id)

        # Add episode
        db.con.execute(
            """
            INSERT INTO episodes (id, replay_url)
            VALUES (?, ?)
            """,
            (episode_id, replay_urls[i]),
        )

    # Create simulations with different policies and environments
    simulation_data = [
        (str(uuid.uuid4()), "sim1", "suite1", "env1", "policy1", 1),
        (str(uuid.uuid4()), "sim2", "suite1", "env2", "policy1", 2),
        (str(uuid.uuid4()), "sim3", "suite2", "env1", "policy2", 1),
    ]

    for i, (sim_id, name, suite, env, policy_key, policy_version) in enumerate(simulation_data):
        # Add simulation
        db._insert_simulation(sim_id, name, suite, env, policy_key, policy_version)

        # Link episode to simulation
        db._update_episode_simulations([episodes[i]], sim_id)

    # Test with no filters (should return all replay URLs)
    all_urls = db.get_replay_urls()
    assert len(all_urls) == 3
    for url in replay_urls:
        assert url in all_urls

    # Test filtering by policy key
    policy1_urls = db._get_replay_urls(policy_key="policy1")
    assert len(policy1_urls) == 2
    assert replay_urls[0] in policy1_urls
    assert replay_urls[1] in policy1_urls

    # Test filtering by policy version
    version1_urls = db._get_replay_urls(policy_version=1)
    assert len(version1_urls) == 2
    assert replay_urls[0] in version1_urls
    assert replay_urls[2] in version1_urls

    # Test filtering by environment
    env1_urls = db.get_replay_urls(env="env1")
    assert len(env1_urls) == 2
    assert replay_urls[0] in env1_urls
    assert replay_urls[2] in env1_urls

    # Test combining filters
    combined_urls = db._get_replay_urls(policy_key="policy1", policy_version=1, env="env1")
    assert len(combined_urls) == 1
    assert replay_urls[0] in combined_urls

    db.close()


def test_export_and_merge(tmp_path: Path):
    """Test exporting and merging databases."""
    # Create first database
    db1_path = tmp_path / "db1.duckdb"
    ep1 = _create_worker_db(db1_path)
    db1 = SimulationStatsDB(db1_path)

    # Manually add simulation and policy data to db1
    sim_id1 = "sim1"
    db1.con.execute(
        """
        CREATE TABLE IF NOT EXISTS simulations (
            id TEXT PRIMARY KEY, 
            name TEXT, 
            suite TEXT, 
            env TEXT,
            policy_key TEXT,
            policy_version INT,
            created_at TIMESTAMP,
            finished_at TIMESTAMP
        )
        """
    )
    db1._insert_simulation(sim_id1, "sim1", "test_suite", "env_test", "test_policy", 1)
    db1.con.execute("UPDATE episodes SET simulation_id = ? WHERE id = ?", (sim_id1, ep1))

    # Create agent_policies table if it doesn't exist
    db1.con.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_policies (
            episode_id TEXT,
            agent_id INTEGER,
            policy_key TEXT,
            policy_version INT,
            PRIMARY KEY (episode_id, agent_id)
        )
        """
    )

    # Add agent policy
    db1.con.execute(
        """
        INSERT INTO agent_policies (episode_id, agent_id, policy_key, policy_version)
        VALUES (?, 0, 'test_policy', 1)
        """,
        (ep1,),
    )

    # Create second database with different simulation
    db2_path = tmp_path / "db2.duckdb"
    ep2 = _create_worker_db(db2_path)
    db2 = SimulationStatsDB(db2_path)

    # Manually add simulation and policy data to db2
    sim_id2 = "sim2"
    db2.con.execute(
        """
        CREATE TABLE IF NOT EXISTS simulations (
            id TEXT PRIMARY KEY, 
            name TEXT, 
            suite TEXT, 
            env TEXT,
            policy_key TEXT,
            policy_version INT,
            created_at TIMESTAMP,
            finished_at TIMESTAMP
        )
        """
    )
    db2._insert_simulation(sim_id2, "sim2", "test_suite", "env_another", "test_policy", 1)
    db2.con.execute("UPDATE episodes SET simulation_id = ? WHERE id = ?", (sim_id2, ep2))

    # Create agent_policies table if it doesn't exist
    db2.con.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_policies (
            episode_id TEXT,
            agent_id INTEGER,
            policy_key TEXT,
            policy_version INT,
            PRIMARY KEY (episode_id, agent_id)
        )
        """
    )

    # Add agent policy
    db2.con.execute(
        """
        INSERT INTO agent_policies (episode_id, agent_id, policy_key, policy_version)
        VALUES (?, 0, 'test_policy', 1)
        """,
        (ep2,),
    )

    # Export db1 to a file
    export_path = tmp_path / "exported.duckdb"
    db1.export(str(export_path))

    # Load the exported file and merge db2 into it
    merged_db = SimulationStatsDB(export_path)
    merged_db.merge_in(db2)

    # Verify both episodes exist
    episodes = merged_db.con.execute("SELECT id FROM episodes").fetchall()
    episode_ids = [row[0] for row in episodes]
    assert ep1 in episode_ids
    assert ep2 in episode_ids

    # Verify both simulations exist
    simulations = merged_db.con.execute("SELECT id, name, env FROM simulations").fetchall()
    assert (sim_id1, "sim1", "env_test") in simulations
    assert (sim_id2, "sim2", "env_another") in simulations

    # Verify agent policies exist for both episodes
    policies = merged_db.con.execute("SELECT episode_id, policy_key, policy_version FROM agent_policies").fetchall()
    assert (ep1, "test_policy", 1) in policies
    assert (ep2, "test_policy", 1) in policies

    db1.close()
    db2.close()
    merged_db.close()


def test_from_shards_and_context(tmp_path: Path):
    """Test creating a SimulationStatsDB from shards and context.

    This test creates a shard database with a test episode, then uses the
    from_shards_and_context method to merge it into a new database.
    """
    # Create a shard with some data
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_path = shard_dir / "shard.duckdb"
    ep_id = _create_worker_db(shard_path)

    # Verify episode was correctly created in the shard
    shard_db = SimulationStatsDB(shard_path)

    # Check if our episode exists in the shard
    shard_episodes = shard_db.con.execute("SELECT id FROM episodes").fetchall()
    shard_episode_ids = [row[0] for row in shard_episodes]
    assert ep_id in shard_episode_ids, f"Episode {ep_id} not found in shard DB"

    # Check that the episode has expected data
    episode_data = shard_db.con.execute("SELECT step_count FROM episodes WHERE id = ?", (ep_id,)).fetchone()
    assert episode_data is not None, "Episode data not found in shard DB"

    # Check for agent metrics in the shard
    metrics = shard_db.con.execute(
        "SELECT value FROM agent_metrics WHERE episode_id = ? AND metric = 'reward'", (ep_id,)
    ).fetchone()
    assert metrics is not None, "Agent metrics not found in shard DB"
    assert metrics[0] == 1.0, f"Expected agent reward metric 1.0, got {metrics[0]}"

    shard_db.close()

    # Delete merged path if it exists
    merged_path = shard_dir / "merged.duckdb"
    if merged_path.exists():
        merged_path.unlink()

    # Check that the merged database doesn't exist yet
    assert not merged_path.exists(), "Merged DB already exists"

    # Create agent map with our mock PolicyRecord
    agent_map = {0: MockPolicyRecord("test_policy", 1)}

    # Now call the actual from_shards_and_context method
    merged_db = SimulationStatsDB.from_shards_and_context(
        "sim_id", shard_dir, agent_map, "test_sim", "test_suite", "env_test", MockPolicyRecord("test_policy", 1)
    )

    # Verify merged database was created
    assert merged_path.exists(), "Merged DB was not created"

    # Verify that all required tables exist in the merged DB
    # Rather than looking at SQLite metadata which varies by database type,
    # we'll try to count records in each table
    try:
        # Episodes table
        episodes_count = merged_db.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert episodes_count > 0, "Episodes table exists but is empty"

        # Agent metrics table
        metrics_count = merged_db.con.execute("SELECT COUNT(*) FROM agent_metrics").fetchone()[0]
        assert metrics_count > 0, "Agent metrics table exists but is empty"

        # Simulations table
        sims_count = merged_db.con.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        assert sims_count > 0, "Simulations table exists but is empty"

        # Agent policies table
        policies_count = merged_db.con.execute("SELECT COUNT(*) FROM agent_policies").fetchone()[0]
        assert policies_count > 0, "Agent policies table exists but is empty"
    except Exception as e:
        assert False, f"Error checking tables in merged DB: {e}"

    # Verify the episode was imported
    result = merged_db.con.execute("SELECT id FROM episodes").fetchall()
    episode_ids = [row[0] for row in result]
    assert ep_id in episode_ids, f"Episode {ep_id} not found in merged DB"

    # Check simulation_id was set correctly
    try:
        sim_check = merged_db.con.execute("SELECT simulation_id FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        assert sim_check is not None, "simulation_id not found in episodes table"
        assert sim_check[0] == "sim_id", f"simulation_id should be sim_id, got {sim_check[0]}"
    except Exception as e:
        assert False, f"Error checking simulation_id: {e}"

    # Verify simulation was created with correct metadata
    sim_result = merged_db.con.execute("SELECT id, name, suite, env FROM simulations").fetchall()
    assert len(sim_result) > 0, "No simulations found in merged DB"

    # Find our simulation by ID
    sim_found = False
    for sim in sim_result:
        if sim[0] == "sim_id":
            sim_found = True
            assert sim[1] == "test_sim", f"sim_name should be test_sim, got {sim[1]}"
            assert sim[2] == "test_suite", f"sim_suite should be test_suite, got {sim[2]}"
            assert sim[3] == "env_test", f"sim_env should be env_test, got {sim[3]}"
            break

    assert sim_found, "Expected simulation with id=sim_id not found"

    # Verify agent policies were created with correct data
    policy_result = merged_db.con.execute(
        "SELECT policy_key, policy_version FROM agent_policies WHERE episode_id = ?", (ep_id,)
    ).fetchall()

    assert len(policy_result) > 0, "No agent policies found for our episode"
    policy_found = False
    for policy in policy_result:
        if policy[0] == "test_policy" and policy[1] == 1:
            policy_found = True
            break

    assert policy_found, f"Expected ('test_policy', 1) not found in {policy_result}"

    # Verify that agent metrics were copied correctly
    metrics_result = merged_db.con.execute(
        "SELECT metric, value FROM agent_metrics WHERE episode_id = ?", (ep_id,)
    ).fetchall()

    assert len(metrics_result) > 0, "No agent metrics found for our episode"
    metric_found = False
    for metric in metrics_result:
        if metric[0] == "reward" and metric[1] == 1.0:
            metric_found = True
            break

    assert metric_found, f"Expected ('reward', 1.0) not found in {metrics_result}"

    merged_db.close()

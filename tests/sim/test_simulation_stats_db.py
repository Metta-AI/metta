from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import cast

from duckdb import DuckDBPyConnection

from metta.agent.mocks import MockPolicyRecord
from metta.agent.policy_record import PolicyRecord
from metta.sim.simulation_stats_db import SimulationStatsDB

_DUMMY_AGENT_MAP = {0: ("dummy_policy", 0)}


class TestHelpers:
    """Helper methods for simulation stats database tests."""

    @staticmethod
    def get_count(con: DuckDBPyConnection, query: str) -> int:
        result = con.execute(query).fetchone()
        assert result is not None
        return result[0]

    @staticmethod
    def create_worker_db(path: Path, sim_steps: int = 0, replay_url: str | None = None) -> str:
        """Create a worker database with a single test episode."""
        path.parent.mkdir(parents=True, exist_ok=True)
        db = SimulationStatsDB(path)

        episode_id = str(uuid.uuid4())
        attributes = {"seed": "0", "map_w": "1", "map_h": "1"}
        agent_metrics = {0: {"reward": 1.0}}
        agent_groups = {0: 0}
        created_at = datetime.datetime.now()

        db.record_episode(
            episode_id,
            attributes,
            agent_metrics,
            agent_groups,
            sim_steps,
            replay_url,
            created_at,
        )

        db.close()
        return episode_id

    @staticmethod
    def create_db_with_tables(path: Path) -> SimulationStatsDB:
        """Create a database with all tables initialized."""
        db = SimulationStatsDB(path)
        for _, sql in db.tables().items():
            db.con.execute(sql)
        return db

    @staticmethod
    def verify_episode_exists(db: SimulationStatsDB, episode_id: str) -> bool:
        """Check if an episode exists in the database."""
        result = db.con.execute("SELECT COUNT(*) FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return result is not None and result[0] > 0

    @staticmethod
    def get_episode_ids(db: SimulationStatsDB) -> list[str]:
        """Get all episode IDs from the database."""
        return [row[0] for row in db.con.execute("SELECT id FROM episodes").fetchall()]


def test_from_uri_context_manager(tmp_path: Path):
    db_path = tmp_path / "test_db.duckdb"
    ep_id = TestHelpers.create_worker_db(db_path)

    with SimulationStatsDB.from_uri(str(db_path)) as db:
        episode_ids = TestHelpers.get_episode_ids(db)
        assert ep_id in episode_ids


def test_insert_agent_policies(tmp_path: Path):
    db_path = tmp_path / "policies.duckdb"
    db = TestHelpers.create_db_with_tables(db_path)

    episode_id = str(uuid.uuid4())
    db.con.execute("INSERT INTO episodes (id) VALUES (?)", (episode_id,))

    agent_map = {0: ("policy_a", 1), 1: ("policy_b", 0)}
    db._insert_agent_policies([episode_id], agent_map)

    count = TestHelpers.get_count(db.con, "SELECT COUNT(*) FROM agent_policies")
    assert count == 2

    db.close()


def test_insert_agent_policies_empty_inputs(tmp_path: Path):
    db_path = tmp_path / "empty_policies.duckdb"
    db = TestHelpers.create_db_with_tables(db_path)

    db._insert_agent_policies([], _DUMMY_AGENT_MAP)
    db._insert_agent_policies([str(uuid.uuid4())], {})

    count = TestHelpers.get_count(db.con, "SELECT COUNT(*) FROM agent_policies")
    assert count == 0

    db.close()


def test_merge_in(tmp_path: Path):
    db1_path = tmp_path / "db1.duckdb"
    db2_path = tmp_path / "db2.duckdb"

    ep1 = TestHelpers.create_worker_db(db1_path)
    ep2 = TestHelpers.create_worker_db(db2_path)

    db1 = SimulationStatsDB(db1_path)
    db2 = SimulationStatsDB(db2_path)

    db1.merge_in(db2)

    episode_ids = TestHelpers.get_episode_ids(db1)
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
    db = TestHelpers.create_db_with_tables(db_path)

    sim_id = str(uuid.uuid4())
    policy_key = "test_policy"
    policy_version = 1
    db._insert_simulation(sim_id, "test_sim", "test_suite", "test_env", policy_key, policy_version)

    rows = db.con.execute("SELECT id, name, suite, env, policy_key, policy_version FROM simulations").fetchall()
    assert len(rows) == 1
    assert rows[0] == (sim_id, "test_sim", "test_suite", "test_env", policy_key, policy_version)

    db.close()


def test_get_replay_urls(tmp_path: Path):
    """Test retrieving replay URLs with various filters."""
    db_path = tmp_path / "replay_urls.duckdb"
    db = TestHelpers.create_db_with_tables(db_path)

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
    policy1_urls = db.get_replay_urls(policy_key="policy1")
    assert len(policy1_urls) == 2
    assert replay_urls[0] in policy1_urls
    assert replay_urls[1] in policy1_urls

    # Test filtering by policy version
    version1_urls = db.get_replay_urls(policy_version=1)
    assert len(version1_urls) == 2
    assert replay_urls[0] in version1_urls
    assert replay_urls[2] in version1_urls

    # Test filtering by environment
    env1_urls = db.get_replay_urls(env="env1")
    assert len(env1_urls) == 2
    assert replay_urls[0] in env1_urls
    assert replay_urls[2] in env1_urls

    # Test combining filters
    combined_urls = db.get_replay_urls(policy_key="policy1", policy_version=1, env="env1")
    assert len(combined_urls) == 1
    assert replay_urls[0] in combined_urls

    db.close()


def test_from_shards_and_context(tmp_path: Path):
    """Test creating a SimulationStatsDB from shards and context.

    This test creates a shard database with a test episode, then uses the
    from_shards_and_context method to merge it into a new database.
    """
    # Create a shard with some data
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_path = shard_dir / "shard.duckdb"
    ep_id = TestHelpers.create_worker_db(shard_path)

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
    agent_map = {0: MockPolicyRecord.from_key_and_version("test_policy", 1)}

    # Now call the actual from_shards_and_context method
    merged_db = SimulationStatsDB.from_shards_and_context(
        "sim_id",
        shard_dir,
        cast(dict[int, PolicyRecord], agent_map),
        "test_sim",
        "test_suite",
        "env_test",
        cast(PolicyRecord, MockPolicyRecord.from_key_and_version("test_policy", 1)),
    )

    # Verify merged database was created
    assert merged_path.exists(), "Merged DB was not created"

    # Verify that all required tables exist in the merged DB
    # Rather than looking at SQLite metadata which varies by database type,
    # we'll try to count records in each table
    # Episodes table
    episodes_count = TestHelpers.get_count(merged_db.con, "SELECT COUNT(*) FROM episodes")
    assert episodes_count > 0, "Episodes table exists but is empty"

    # Agent metrics table
    metrics_count = TestHelpers.get_count(merged_db.con, "SELECT COUNT(*) FROM agent_metrics")
    assert metrics_count > 0, "Agent metrics table exists but is empty"

    # Simulations table
    sims_count = TestHelpers.get_count(merged_db.con, "SELECT COUNT(*) FROM simulations")
    assert sims_count > 0, "Simulations table exists but is empty"

    # Agent policies table
    policies_count = TestHelpers.get_count(merged_db.con, "SELECT COUNT(*) FROM agent_policies")
    assert policies_count > 0, "Agent policies table exists but is empty"

    # Verify the episode was imported
    episode_ids = TestHelpers.get_episode_ids(merged_db)
    assert ep_id in episode_ids, f"Episode {ep_id} not found in merged DB"

    # Check simulation_id was set correctly
    sim_check = merged_db.con.execute("SELECT simulation_id FROM episodes WHERE id = ?", (ep_id,)).fetchone()
    assert sim_check is not None, "simulation_id not found in episodes table"
    assert sim_check[0] == "sim_id", f"simulation_id should be sim_id, got {sim_check[0]}"

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


def test_sequential_policy_merges(tmp_path: Path):
    """Test that policies are preserved during sequential merges.

    This test simulates the workflow in the actual code where:
    1. We create a database with Policy A
    2. Export it to a destination
    3. Create a new database with Policy B
    4. Have the new database merge with the exported one
    5. Export the result
    6. Verify both policies exist in the final database
    """
    # Create first database with Policy A
    db1_path = tmp_path / "db1.duckdb"
    db1 = TestHelpers.create_db_with_tables(db1_path)

    # Add simulation for Policy A
    db1._insert_simulation("sim1", "test_sim", "test_suite", "env_test", "policy_A", 1)

    # Add episode linked to sim1
    episode1_id = str(uuid.uuid4())
    db1.con.execute("INSERT INTO episodes (id, simulation_id) VALUES (?, ?)", (episode1_id, "sim1"))

    # Create export destination
    export_path = tmp_path / "export.duckdb"
    db1.export(str(export_path))
    db1.close()

    # Now create second database with Policy B
    db2_path = tmp_path / "db2.duckdb"
    db2 = TestHelpers.create_db_with_tables(db2_path)

    # Add simulation for Policy B
    db2._insert_simulation("sim2", "test_sim", "test_suite", "env_test", "policy_B", 1)

    # Add episode linked to sim2
    episode2_id = str(uuid.uuid4())
    db2.con.execute("INSERT INTO episodes (id, simulation_id) VALUES (?, ?)", (episode2_id, "sim2"))

    # Now simulate the workflow: db2 exports to export_path after merging with it
    db2.export(str(export_path))
    db2.close()

    # Now verify that export_path contains both policies
    result_db = SimulationStatsDB(export_path)

    # Check if both episodes exist
    episode_ids = TestHelpers.get_episode_ids(result_db)
    assert episode1_id in episode_ids, f"Episode {episode1_id} (Policy A) not found in result"
    assert episode2_id in episode_ids, f"Episode {episode2_id} (Policy B) not found in result"

    # Check if both simulations exist
    simulations = result_db.con.execute("SELECT id, policy_key, policy_version FROM simulations").fetchall()

    # Look for policy_A
    policy_a_found = False
    policy_b_found = False

    for sim in simulations:
        if sim[1] == "policy_A" and sim[2] == 1:
            policy_a_found = True
        if sim[1] == "policy_B" and sim[2] == 1:
            policy_b_found = True

    assert policy_a_found, "Policy A not found in result database"
    assert policy_b_found, "Policy B not found in result database"

    # Verify policy count
    all_policies = result_db.get_all_policy_uris()
    assert len(all_policies) == 2, f"Expected 2 policies, got {len(all_policies)}: {all_policies}"

    result_db.close()


def test_export_preserves_all_policies(tmp_path: Path):
    """Test that export correctly preserves all policies when merging."""
    # Create a database with two policies
    db_path = tmp_path / "source.duckdb"
    db = TestHelpers.create_db_with_tables(db_path)

    # Add two different policies
    db._insert_simulation("sim1", "test_sim", "test_suite", "env_test", "policy_X", 1)
    db._insert_simulation("sim2", "test_sim", "test_suite", "env_test", "policy_Y", 1)

    # Export to a new location
    export_path = tmp_path / "export_test.duckdb"
    db.export(str(export_path))

    # Close the first database after export
    db.close()

    # Check exported database
    exported_db = SimulationStatsDB(export_path)
    policies = exported_db.get_all_policy_uris()

    # Should have both policies
    assert "policy_X:v1" in policies, f"policy_X:v1 not found in {policies}"
    assert "policy_Y:v1" in policies, f"policy_Y:v1 not found in {policies}"
    assert len(policies) == 2, f"Expected 2 policies, got {len(policies)}: {policies}"

    # Close the exported database after checking
    exported_db.close()

    # Now create a new database with a third policy
    new_db_path = tmp_path / "new_source.duckdb"
    new_db = TestHelpers.create_db_with_tables(new_db_path)

    new_db._insert_simulation("sim3", "test_sim", "test_suite", "env_test", "policy_Z", 1)

    # Export this new db to the same export location
    new_db.export(str(export_path))

    # Close the new database after export
    new_db.close()

    # Check the updated export - should contain all three policies
    final_db = SimulationStatsDB(export_path)
    final_policies = final_db.get_all_policy_uris()

    # Should have all three policies
    assert "policy_X:v1" in final_policies, f"policy_X:v1 not found in {final_policies}"
    assert "policy_Y:v1" in final_policies, f"policy_Y:v1 not found in {final_policies}"
    assert "policy_Z:v1" in final_policies, f"policy_Z:v1 not found in {final_policies}"

    # Close the final database
    final_db.close()

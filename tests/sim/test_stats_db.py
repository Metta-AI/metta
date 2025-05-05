"""
Integration tests for metta.sim.stats_db.StatsDB.

We spin up tiny worker shards with the vanilla MettaGrid StatsDB,
then merge them through Metta's helper and verify:

* shards merge correctly,
* episodes carry env-names,
* simulations table + simulation_id FK are populated,
* agent-policy multiplexing works,
* helper utilities behave,
* edge-cases (empty shards / empty inputs) don't crash.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from metta.sim.stats_db import StatsDB
from mettagrid.stats_writer import StatsDB as MGStatsDB


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _create_worker_db(path: Path, sim_steps: int = 0) -> str:
    """
    Create a shard with **one** episode + a single agent-metric row.

    Returns the episode_id so tests can assert later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    db = MGStatsDB(str(path), read_only=False)

    ep_id = db.create_episode(seed=0, map_w=1, map_h=1)
    db.add_agent_metrics(ep_id, agent_id=0, metrics={"reward": 1.0})
    db.finish_episode(ep_id, step_count=sim_steps)
    db.close()
    return ep_id


_DUMMY_AGENT_MAP = {0: ("dummy_policy", "default")}  # Use "default" instead of None


# --------------------------------------------------------------------------- #
# merge & context                                                             #
# --------------------------------------------------------------------------- #
def test_merge_two_shards_with_context(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    ep_a = _create_worker_db(shards_dir / "worker0.duckdb")
    ep_b = _create_worker_db(shards_dir / "worker1.duckdb")

    merged: StatsDB = StatsDB.merge_shards_and_add_context(
        shards_dir, _DUMMY_AGENT_MAP, "sim_alpha", "suite_nav", "env_a"
    )

    # ---- rows copied ------------------------------------------------------
    df = merged.query("SELECT id FROM episodes ORDER BY id")
    assert sorted(df["id"]) == sorted([ep_a, ep_b])

    # ---- simulation linkage ----------------------------------------------
    join = merged.query(
        """
        SELECT e.id, s.name, s.suite
          FROM episodes e
          JOIN simulations s ON e.simulation_id = s.id
        """
    )
    assert set(join["name"]) == {"sim_alpha"}
    assert set(join["suite"]) == {"suite_nav"}
    assert set(join["id"]) == {ep_a, ep_b}

    # ---- agent-policy multiplex ------------------------------------------
    ap = merged.query("SELECT * FROM agent_policies")
    assert ap.shape == (2, 4)  # 2 episodes × 1 agent
    assert set(ap["policy_key"]) == {"dummy_policy"}

    merged.close()


def test_merge_empty_shard_dir(tmp_path: Path):
    merged = StatsDB.merge_shards_and_add_context(tmp_path, _DUMMY_AGENT_MAP, "sim_empty", "suite_empty", "env_a")
    assert merged.query("SELECT COUNT(*) AS n FROM episodes")["n"][0] == 0
    merged.close()


# --------------------------------------------------------------------------- #
# simulation-id semantics                                                     #
# --------------------------------------------------------------------------- #
def test_simulation_id_uniqueness(tmp_path: Path):
    shards = tmp_path / "shards"
    _create_worker_db(shards / "a.duckdb")
    merged1 = StatsDB.merge_shards_and_add_context(shards, _DUMMY_AGENT_MAP, "sim1", "suite", "env_a")
    sim_id_1 = merged1.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged1.close()

    # new merge with a *different* simulation name -> new UUID
    merged2 = StatsDB.merge_shards_and_add_context(shards, _DUMMY_AGENT_MAP, "sim2", "suite", "env_a")
    sim_id_2 = merged2.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged2.close()

    assert sim_id_1 != sim_id_2


def test_simulation_id_deduplication(tmp_path: Path):
    """Merging twice with the *same* (suite,name) pair must reuse the UUID."""
    shards = tmp_path / "shards"
    _create_worker_db(shards / "a.duckdb")
    merged1 = StatsDB.merge_shards_and_add_context(shards, _DUMMY_AGENT_MAP, "sim_fixed", "suite", "env_a")
    sim_id_1 = merged1.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged1.close()

    merged2 = StatsDB.merge_shards_and_add_context(shards, _DUMMY_AGENT_MAP, "sim_fixed", "suite", "env_a")
    sim_id_2 = merged2.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged2.close()

    assert sim_id_1 == sim_id_2


# --------------------------------------------------------------------------- #
# insert_agent_policies                                                       #
# --------------------------------------------------------------------------- #
def test_insert_agent_policies():
    db = StatsDB(tmp_path := Path(f"{uuid.uuid4().hex}.duckdb"), mode="rwc")

    episodes = [uuid.uuid4().hex for _ in range(3)]
    # Use "default" instead of None
    agent_map = {0: ("policy_a", "v1"), 1: ("policy_b", "default")}
    db.insert_agent_policies(episodes, agent_map)

    pol = db.query("SELECT * FROM policies")
    assert pol.shape[0] == 2

    ap = db.query("SELECT * FROM agent_policies")
    assert ap.shape[0] == len(episodes) * len(agent_map)
    db.close()
    tmp_path.unlink(missing_ok=True)


def test_insert_agent_policies_empty_inputs(tmp_path: Path):
    db = StatsDB(tmp_path / "x.duckdb", mode="rwc")
    db.insert_agent_policies([], _DUMMY_AGENT_MAP)
    db.insert_agent_policies([uuid.uuid4().hex], {})
    assert db.query("SELECT COUNT(*) AS n FROM agent_policies")["n"][0] == 0
    db.close()


# --------------------------------------------------------------------------- #
# utilities                                                                   #
# --------------------------------------------------------------------------- #
def test_query_method(tmp_path: Path):
    db = StatsDB(tmp_path / "q.duckdb", mode="rwc")
    ep = db.get_next_episode_id()
    db.con.execute(
        """
        INSERT INTO episodes
            (id, seed, map_w, map_h,
             step_count, started_at, finished_at, metadata)
        VALUES (?, 0, 1, 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
        """,
        (ep,),
    )
    df = db.query("SELECT * FROM episodes")
    assert df.loc[0, "id"] == ep
    db.close()


def test_get_metrics_for_episode(tmp_path: Path):
    db = StatsDB(tmp_path / "m.duckdb", mode="rwc")
    ep = db.get_next_episode_id()
    db.con.execute(
        """
        INSERT INTO episodes
            (id, seed, map_w, map_h,
             step_count, started_at, finished_at, metadata)
        VALUES (?, 0, 1, 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
        """,
        (ep,),
    )
    db.con.execute(
        """
        INSERT INTO agent_metrics
            (episode_id, agent_id, metric, value)
        VALUES
            (?, 0, 'reward', 5),
            (?, 0, 'steps', 100),
            (?, 1, 'reward', 4)
        """,
        (ep, ep, ep),
    )
    metrics = db.get_metrics_for_episode(ep)
    assert metrics[0]["reward"] == 5
    assert metrics[0]["steps"] == 100
    assert metrics[1]["reward"] == 4
    db.close()


def test_sequential_policy_simulations_and_merging():
    """Test that evaluating multiple policies sequentially correctly merges all episodes."""

    # Create a temp directory for our stats
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up three separate simulation result directories
        policy_dirs = [os.path.join(temp_dir, f"policy_{i}") for i in range(3)]
        for dir_path in policy_dirs:
            os.makedirs(dir_path)

        # Create a merged DB to hold the combined results
        merged_db_path = os.path.join(temp_dir, "merged.duckdb")
        merged_db = StatsDB(merged_db_path)

        # For each policy, create a simple DB with one episode and merge
        total_episodes = 0
        for i, policy_dir in enumerate(policy_dirs):
            # Create a shard with one episode
            shard_path = os.path.join(policy_dir, f"stats_{i}.duckdb")
            shard_db = StatsDB(shard_path)

            # Add a test episode
            episode_id = shard_db.create_episode(seed=i, map_w=10, map_h=10)
            shard_db.add_agent_metrics(episode_id, 0, {"reward": float(i)})
            shard_db.finish_episode(episode_id, 100)

            # Create agent mapping
            agent_map = {0: (f"policy_{i}", f"v{i}")}

            # Now do what the simulation would do:
            # 1. Merge shards (single shard in this case)
            policy_db = StatsDB.merge_shards_and_add_context(
                policy_dir, agent_map, f"sim_{i}", "test_suite", f"env/test_{i}"
            )

            # 2. Merge into the combined DB
            merged_db.merge_in(policy_db)
            policy_db.close()

            # Check the count of episodes - should increase by 1 each time
            total_episodes += 1
            episode_count = merged_db.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            assert episode_count == total_episodes, f"Expected {total_episodes} episodes, got {episode_count}"

            # Check we can retrieve all episodes with metrics
            metrics = merged_db.con.execute(
                "SELECT e.id, a.value FROM episodes e JOIN agent_metrics a ON e.id = a.episode_id"
            ).fetchall()
            assert len(metrics) == total_episodes, f"Expected {total_episodes} metrics, got {len(metrics)}"

            # Check policies were properly recorded
            policies = merged_db.con.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
            assert policies == total_episodes, f"Expected {total_episodes} policies, got {policies}"

        # Final check - should have 3 episodes from 3 policies
        episode_count = merged_db.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert episode_count == 3, f"Final check: Expected 3 episodes, got {episode_count}"


@pytest.fixture
def test_db():
    """Create a temporary StatsDB with test data."""
    # Create temporary directory and db file
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"{uuid.uuid4().hex}.duckdb"
        db = StatsDB(db_path, mode="rwc")

        # Create test simulation
        sim_id = db.ensure_simulation_id("test_sim", "test_suite", "env_test")

        # Create test episodes
        episodes = []
        for i in range(3):
            ep_id = db.get_next_episode_id()
            episodes.append(ep_id)
            db.con.execute(
                """
                INSERT INTO episodes 
                (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """,
                (ep_id, i, 10, 10, 100, sim_id),
            )

        # Add agent policies with NO NULL values
        agent_policies = {
            "policy_a": ("v1", [0]),  # policy_key, version, agent_ids
            "policy_b": ("default", [1]),  # Using "default" instead of None
        }

        for policy_key, (version, agent_ids) in agent_policies.items():
            for ep_id in episodes:
                for agent_id in agent_ids:
                    db.con.execute(
                        """
                        INSERT INTO agent_policies
                        (episode_id, agent_id, policy_key, policy_version)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep_id, agent_id, policy_key, version),
                    )

        # Add metrics
        metrics = {"reward": [1.0, 2.0, 3.0], "steps": [100, 200, 300], "efficiency": [0.5, 0.7, 0.9]}

        for ep_idx, ep_id in enumerate(episodes):
            for agent_id in range(2):  # Two agents per episode
                for metric, values in metrics.items():
                    # Add some randomness to make it more realistic
                    value = values[ep_idx] + np.random.normal(0, 0.1)
                    db.con.execute(
                        """
                        INSERT INTO agent_metrics
                        (episode_id, agent_id, metric, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep_id, agent_id, metric, value),
                    )

        yield db
        db.close()


def test_materialize_policy_simulations_view_creates_table(test_db):
    """Test that materialize_policy_simulations_view creates a table with the expected name."""
    # Arrange
    metric = "reward"
    expected_table = f"policy_simulations_{metric}"

    # Act
    test_db.materialize_policy_simulations_view(metric)

    # Assert
    tables = test_db.con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert expected_table in table_names


def test_materialize_policy_simulations_view_with_nonexistent_metric(test_db):
    """Test that materialize_policy_simulations_view handles nonexistent metrics gracefully."""
    # Arrange
    non_existent_metric = "nonexistent_metric"
    expected_table = f"policy_simulations_{non_existent_metric}"

    # Act
    test_db.materialize_policy_simulations_view(non_existent_metric)

    # Assert
    tables = test_db.con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert expected_table not in table_names  # Table should not be created


def test_materialize_policy_simulations_view_aggregates_correctly(test_db):
    """Test that metrics are correctly aggregated in the materialized view."""
    # Arrange
    metric = "reward"

    # Act
    test_db.materialize_policy_simulations_view(metric)

    # Assert
    result = test_db.con.execute(f"SELECT * FROM policy_simulations_{metric}").fetchall()
    assert len(result) > 0  # Should have results

    # We should have a row for each policy/simulation combination
    rows = test_db.con.execute(f"""
        SELECT COUNT(*) FROM policy_simulations_{metric}
    """).fetchone()[0]

    # Since we have 2 policies and 1 simulation, expect 2 rows
    assert rows == 2

    # Check that we can query the table with expected column names
    # Instead of using PRAGMA table_info which returns indices in DuckDB,
    # directly query the table with the expected column names
    result = test_db.con.execute(f"""
        SELECT
            policy_key,
            policy_version,
            eval_name,
            sim_suite,
            sim_name,
            {metric},
            {metric}_std
        FROM policy_simulations_{metric}
        LIMIT 1
    """).fetchone()

    # If this query succeeds, it means all these columns exist
    assert result is not None


def test_materialize_policy_simulations_view_primary_key(test_db):
    """Test that the primary key is correctly set."""
    # Arrange
    metric = "reward"

    # Act
    test_db.materialize_policy_simulations_view(metric)

    # Assert
    pk_info = test_db.con.execute(f"""
        PRAGMA table_info(policy_simulations_{metric})
    """).fetchall()

    # Find columns that are part of PK
    pk_columns = [info[1] for info in pk_info if info[5] > 0]  # index 5 is pk flag

    # Check expected PK columns
    expected_pk = ["policy_key", "policy_version", "sim_suite", "sim_name"]
    assert set(pk_columns) == set(expected_pk)


def test_materialize_policy_simulations_view_multiple_metrics(test_db):
    """Test materializing views for multiple metrics."""
    # Arrange
    metrics = ["reward", "steps", "efficiency"]

    # Act
    for metric in metrics:
        test_db.materialize_policy_simulations_view(metric)

    # Assert
    tables = test_db.con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]

    for metric in metrics:
        expected_table = f"policy_simulations_{metric}"
        assert expected_table in table_names


def test_materialize_policy_simulations_view_invalid_metric_name(test_db):
    """Test that invalid metric names are rejected."""
    # Arrange
    invalid_metric = "invalid; DROP TABLE episodes; --"

    # Act & Assert
    with pytest.raises(ValueError):
        test_db.materialize_policy_simulations_view(invalid_metric)


def test_materialize_policy_simulations_view_index_creation(test_db):
    """Test that indices are created correctly."""
    # Arrange
    metric = "reward"

    # Act
    test_db.materialize_policy_simulations_view(metric)

    # Assert
    indices = test_db.con.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    index_names = [i[0] for i in indices]

    expected_index = f"idx_policy_simulations_{metric}_sim"
    assert expected_index in index_names


def test_view_query_results(test_db):
    """Test that queries against the view work correctly."""
    # Arrange
    metric = "reward"
    test_db.materialize_policy_simulations_view(metric)

    # Act - Run a query that should return results
    result = test_db.con.execute(f"""
        SELECT * FROM policy_simulations_{metric}
        WHERE sim_suite = 'test_suite' AND sim_name = 'test_sim'
    """).fetchall()

    # Assert - Check that we got results
    assert len(result) > 0

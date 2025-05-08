"""
Integration tests for metta.eval.eval_stats_db.EvalStatsDB.

These tests verify the functionality of the EvalStatsDB which adds views
on top of SimulationStatsDB to make it easier to query policy performance
across simulations.
"""

from __future__ import annotations

import datetime
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import pytest

from metta.eval.eval_stats_db import EvalStatsDB


# Mock for PolicyRecord
class MockPolicyRecord:
    """Mock implementation of PolicyRecord for testing."""

    def __init__(self, policy_key: str, policy_version: int):
        self._policy_key = policy_key
        self._policy_version = policy_version

    def key_and_version(self) -> Tuple[str, int]:
        """Return the policy key and version as a tuple."""
        return self._policy_key, self._policy_version


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _create_test_db(db_path: Path) -> Tuple[EvalStatsDB, List[str], str]:
    """
    Create a test database with sample data.

    Returns:
        Tuple of (database, episode_ids, simulation_id)
    """
    # Create an EvalStatsDB - schema should be initialized automatically
    db = EvalStatsDB(db_path)

    # Verify that required tables and views exist
    tables_and_views = db.con.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    table_names = {row[0] for row in tables_and_views if row[1] == "table"}
    view_names = {row[0] for row in tables_and_views if row[1] == "view"}

    # Assert that required tables exist
    assert "episodes" in table_names, "episodes table not created"
    assert "agent_metrics" in table_names, "agent_metrics table not created"
    assert "simulations" in table_names, "simulations table not created"
    assert "agent_policies" in table_names, "agent_policies table not created"

    # Assert that required views exist
    assert "policy_simulation_agent_samples" in view_names, "policy_simulation_agent_samples view not created"
    assert "policy_simulation_agent_aggregates" in view_names, "policy_simulation_agent_aggregates view not created"

    # Create a test simulation
    sim_id = str(uuid.uuid4())
    policy_key = "test_policy"
    policy_version = 1

    # Insert the simulation
    db._insert_simulation(
        sim_id=sim_id,
        name="test_sim",
        suite="test_suite",
        env="env_test",
        policy_key=policy_key,
        policy_version=policy_version,
    )

    # Create test episodes
    episodes = []
    for i in range(3):
        ep_id = str(uuid.uuid4())
        episodes.append(ep_id)

        # Create an episode and link it to the simulation
        step_count = 100
        replay_url = None
        created_at = datetime.datetime.now()
        attributes = {"seed": str(i), "map_w": "10", "map_h": "10"}
        groups = [[0], [1]]  # Two groups, one for each agent

        # Add metrics for the episode
        agent_metrics = {
            0: {"reward": 1.0 + i, "steps": 100.0 * (i + 1), "success": float(i > 0)},
            1: {"reward": 1.5 + i, "steps": 110.0 * (i + 1), "success": float(i >= 0)},
        }

        group_metrics = {0: {"reward": 1.0 + i}, 1: {"reward": 1.5 + i}}

        # Record the episode using the higher-level function
        db.record_episode(ep_id, attributes, groups, agent_metrics, group_metrics, step_count, replay_url, created_at)

        # Link episode to simulation
        db.con.execute("UPDATE episodes SET simulation_id = ? WHERE id = ?", (sim_id, ep_id))

    # Add agent policies for each agent in each episode
    for ep_id in episodes:
        # Agent 0 with test_policy
        db._insert_agent_policies([ep_id], {0: (policy_key, policy_version)})

        # Agent 1 with other_policy
        db._insert_agent_policies([ep_id], {1: ("other_policy", 2)})

    db.con.commit()
    return db, episodes, sim_id


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def test_db():
    """Create a temporary EvalStatsDB with test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"{uuid.uuid4().hex}.duckdb"
        db, episodes, sim_id = _create_test_db(db_path)
        yield db, episodes, sim_id
        db.close()


@pytest.fixture(autouse=True)
def mock_policy_record(monkeypatch):
    """Mock PolicyRecord to avoid import errors."""
    monkeypatch.setattr("metta.eval.eval_stats_db.PolicyRecord", MockPolicyRecord)


# --------------------------------------------------------------------------- #
# tests                                                                       #
# --------------------------------------------------------------------------- #


def test_tables_includes_views():
    """Test that the tables() method includes the views from parent class."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.duckdb"
        db = EvalStatsDB(db_path)

        tables = db.tables()

        # Check parent tables are included
        assert "episodes" in tables
        assert "agent_metrics" in tables
        assert "simulations" in tables
        assert "agent_policies" in tables

        # Check that our views are included
        assert "policy_simulation_agent_samples" in tables
        assert "policy_simulation_agent_aggregates" in tables

        db.close()


def test_schema_initialization():
    """Test that the schema is initialized correctly with views."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.duckdb"
        db = EvalStatsDB(db_path)

        # Check if views are created by trying to execute a query
        views_exist = True
        try:
            db.con.execute("SELECT * FROM policy_simulation_agent_samples LIMIT 1")
            db.con.execute("SELECT * FROM policy_simulation_agent_aggregates LIMIT 1")
        except Exception:
            views_exist = False

        assert views_exist, "Views should be created during initialization"
        db.close()


def test_get_average_metric_by_filter(test_db):
    """Test the get_average_metric_by_filter method."""
    db, _, _ = test_db

    # We'll use the policy_record fixture
    policy_record = MockPolicyRecord("test_policy", 1)

    # Test with real data when possible
    avg_reward = db.get_average_metric_by_filter("reward", policy_record)
    assert avg_reward is not None, "Should find average reward"

    # Test with filter condition
    avg_filtered = db.get_average_metric_by_filter("reward", policy_record, "sim_suite = 'test_suite'")
    assert avg_filtered is not None, "Should find filtered average"

    # Test with non-matching filter
    avg_no_match = db.get_average_metric_by_filter("reward", policy_record, "sim_suite = 'nonexistent'")
    assert avg_no_match is None, "Should not find average for non-existent suite"

    # Test with non-existent policy
    nonexistent_policy = MockPolicyRecord("nonexistent", 1)
    avg_nonexistent = db.get_average_metric_by_filter("reward", nonexistent_policy)
    assert avg_nonexistent is None, "Should not find average for non-existent policy"

    # Test with non-existent metric
    avg_bad_metric = db.get_average_metric_by_filter("nonexistent", policy_record)
    assert avg_bad_metric is None, "Should not find average for non-existent metric"


def test_simulation_scores(test_db):
    """Test the simulation_scores method."""
    db, _, _ = test_db

    # Get scores for test_policy
    policy_record = MockPolicyRecord("test_policy", 1)
    scores = db.simulation_scores(policy_record, "reward")

    # Verify that we get scores
    assert len(scores) > 0, "Should find scores"

    # The key should be (test_suite, test_sim, env_test)
    keys = list(scores.keys())
    assert len(keys) == 1, "Should have one simulation"
    key = keys[0]
    assert key[0] == "test_suite", "Suite should be test_suite"
    assert key[1] == "test_sim", "Name should be test_sim"
    assert key[2] == "env_test", "Env should be env_test"

    # Test with non-existent policy
    nonexistent_policy = MockPolicyRecord("nonexistent", 1)
    nonexistent_scores = db.simulation_scores(nonexistent_policy, "reward")
    assert len(nonexistent_scores) == 0, "Should not find scores for non-existent policy"

    # Test with non-existent metric
    nonexistent_metric_scores = db.simulation_scores(policy_record, "nonexistent")
    assert len(nonexistent_metric_scores) == 0, "Should not find scores for non-existent metric"


def test_empty_database():
    """Test behavior with an empty database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "empty.duckdb"
        db = EvalStatsDB(db_path)

        # Verify that tables and views are created
        tables_and_views = db.con.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        table_names = {row[0] for row in tables_and_views if row[1] == "table"}
        view_names = {row[0] for row in tables_and_views if row[1] == "view"}

        assert "episodes" in table_names, "episodes table should be created"
        assert "simulations" in table_names, "simulations table should be created"
        assert "policy_simulation_agent_samples" in view_names, "Metrics view should be created"

        # Should return None for non-existent data
        policy_record = MockPolicyRecord("test_policy", 1)
        avg_reward = db.get_average_metric_by_filter("reward", policy_record)
        assert avg_reward is None, "Should not find average for empty database"

        # Should return empty dict for simulation_scores
        scores = db.simulation_scores(policy_record, "reward")
        assert len(scores) == 0, "Should not find scores for empty database"

        db.close()


def test_multiple_policies_and_simulations(tmp_path):
    """Test with multiple policies and simulations."""
    db_path = tmp_path / "multi.duckdb"

    # Create initial database
    db = EvalStatsDB(db_path)

    # Create two simulations with different environments
    sim_id1 = "sim1"
    sim_id2 = "sim2"

    db._insert_simulation(sim_id1, "sim_nav", "test_suite", "env_nav", "test_policy", 1)
    db._insert_simulation(sim_id2, "sim_maze", "test_suite", "env_maze", "test_policy", 1)

    # Create episodes for each simulation
    ep_nav = str(uuid.uuid4())
    ep_maze = str(uuid.uuid4())

    # Add episodes
    for ep_id, sim_id in [(ep_nav, sim_id1), (ep_maze, sim_id2)]:
        # Create basic episode data
        attributes = {"seed": "0", "map_w": "10", "map_h": "10"}
        groups = [[0]]
        agent_metrics = {0: {"reward": 10.0 if sim_id == sim_id1 else 20.0}}
        group_metrics = {0: {"reward": 10.0 if sim_id == sim_id1 else 20.0}}
        step_count = 100
        replay_url = None
        created_at = datetime.datetime.now()

        # Record episode
        db.record_episode(ep_id, attributes, groups, agent_metrics, group_metrics, step_count, replay_url, created_at)

        # Link to simulation
        db.con.execute("UPDATE episodes SET simulation_id = ? WHERE id = ?", (sim_id, ep_id))

        # Add agent policy
        db._insert_agent_policies([ep_id], {0: ("test_policy", 1)})

    db.con.commit()

    # Test simulation_scores
    policy_record = MockPolicyRecord("test_policy", 1)
    scores = db.simulation_scores(policy_record, "reward")

    # Should have two environments with different scores
    assert len(scores) == 2, "Should find two simulations"
    assert ("test_suite", "sim_nav", "env_nav") in scores, "Should find nav environment"
    assert ("test_suite", "sim_maze", "env_maze") in scores, "Should find maze environment"

    # Scores should be as expected
    assert 9.5 < scores[("test_suite", "sim_nav", "env_nav")] < 10.5, "Nav score should be around 10.0"
    assert 19.5 < scores[("test_suite", "sim_maze", "env_maze")] < 20.5, "Maze score should be around 20.0"

    # Test average metric with filter
    nav_avg = db.get_average_metric_by_filter("reward", policy_record, "sim_env = 'env_nav'")
    assert 9.5 < nav_avg < 10.5, "Nav average should be around 10.0"

    maze_avg = db.get_average_metric_by_filter("reward", policy_record, "sim_env = 'env_maze'")
    assert 19.5 < maze_avg < 20.5, "Maze average should be around 20.0"

    # Test overall average (both environments)
    overall_avg = db.get_average_metric_by_filter("reward", policy_record)
    assert 14.5 < overall_avg < 15.5, "Overall average should be around 15.0"

    db.close()

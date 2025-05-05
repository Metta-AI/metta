"""
Unit tests for get_matrix_data function with StatsDB focusing on different view types.
"""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from metta.eval.db import get_matrix_data
from metta.sim.stats_db import StatsDB


@pytest.fixture
def sample_stats_db():
    """Create a temporary StatsDB with sample data for testing."""
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

        # Add policy entries - required for version-based lookup
        policies = [
            ("policy1", "v1"),
            ("policy1", "v2"),
            ("policy1", "v3"),
            ("policy2", "v1"),
            ("policy2", "v2"),
            ("policy3", "v1"),
        ]

        db.con.executemany(
            """
            INSERT INTO policies (policy_key, policy_version)
            VALUES (?, ?)
            """,
            policies,
        )

        # Create a modified agent_policies_map where we use unique agent_ids for each policy
        # This avoids the primary key constraint violation
        policy_agent_id = 0
        agent_policies = []

        # Structure: (agent_id, policy_key, policy_version)
        for policy_key, policy_version in policies:
            for ep_id in episodes:
                agent_policies.append((ep_id, policy_agent_id, policy_key, policy_version))
                policy_agent_id += 1  # Increment agent_id to ensure uniqueness

        # Insert all agent policies at once
        db.con.executemany(
            """
            INSERT INTO agent_policies
            (episode_id, agent_id, policy_key, policy_version)
            VALUES (?, ?, ?, ?)
            """,
            agent_policies,
        )

        # Add metrics with the expected test values
        metric_values = {
            "policy1:v1": {"eval1": 10.0, "eval2": 15.0, "eval3": 20.0},
            "policy1:v2": {"eval1": 12.0, "eval2": 17.0, "eval3": 22.0},
            "policy1:v3": {"eval1": 14.0, "eval2": 19.0, "eval3": 24.0},
            "policy2:v1": {"eval1": 8.0, "eval2": 13.0, "eval3": 18.0},
            "policy2:v2": {"eval1": 9.0, "eval2": 14.0, "eval3": 19.0},
            "policy3:v1": {"eval1": 7.0, "eval2": 12.0, "eval3": 17.0},
        }

        # We need to map back from (policy_key, policy_version) to agent_id for metrics
        policy_to_agent_ids = {}
        policy_agent_id = 0
        for policy_key, policy_version in policies:
            policy_uri = f"{policy_key}:{policy_version}"
            if policy_uri not in policy_to_agent_ids:
                policy_to_agent_ids[policy_uri] = []
            policy_to_agent_ids[policy_uri].extend([policy_agent_id + i for i in range(len(episodes))])
            policy_agent_id += len(episodes)

        # Add metrics for each policy and its corresponding agents
        for policy_uri, agent_ids in policy_to_agent_ids.items():
            for ep_idx, ep_id in enumerate(episodes):
                agent_id = agent_ids[ep_idx]
                for eval_name, value in metric_values[policy_uri].items():
                    db.con.execute(
                        """
                        INSERT INTO agent_metrics
                        (episode_id, agent_id, metric, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep_id, agent_id, "episode_reward", value + np.random.normal(0, 0.01)),
                    )

        # Materialize the view for episode_reward
        db.materialize_policy_simulations_view("episode_reward")

        yield db
        db.close()


def test_get_matrix_data_all(sample_stats_db):
    """Test get_matrix_data with 'all' view type."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="all")

    # Should include all versions of all policies
    assert len(matrix) == 6  # 6 total policy versions

    # Check that all versions are included with correct formatting
    for policy_uri in ["policy1:v1", "policy1:v2", "policy1:v3", "policy2:v1", "policy2:v2", "policy3:v1"]:
        assert policy_uri in matrix.index

    # Check that all evaluation columns are present
    assert "Overall" in matrix.columns
    assert len(matrix.columns) > 1  # At least Overall and one eval column

    # Check the values are approximately correct (allowing for small random variations)
    # The policies have fixed values in the test fixture
    policy1_v2_mean = matrix.loc["policy1:v2", "Overall"]
    assert 16.5 < policy1_v2_mean < 17.5  # Approximately (12.0 + 17.0 + 22.0) / 3

    # Verify that the matrix is sorted by overall score (lowest first)
    policies_by_score = list(matrix.index)
    # policy3:v1 has lowest overall score
    assert policies_by_score[0] == "policy3:v1"
    # policy1:v3 has highest overall score
    assert policies_by_score[-1] == "policy1:v3"


def test_get_matrix_data_latest(sample_stats_db):
    """Test get_matrix_data with 'latest' view type."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="latest")

    # Should only include the latest version for each policy
    assert len(matrix) == 3  # 3 policies

    # Check that we have the latest versions
    assert "policy1:v3" in matrix.index
    assert "policy2:v2" in matrix.index
    assert "policy3:v1" in matrix.index

    # Check that earlier versions are excluded
    assert "policy1:v1" not in matrix.index
    assert "policy1:v2" not in matrix.index
    assert "policy2:v1" not in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns

    # Verify the policies are in expected order (lowest to highest score)
    policy_order = list(matrix.index)
    assert policy_order[0] == "policy3:v1"  # Lowest score
    assert policy_order[-1] == "policy1:v3"  # Highest score


def test_get_matrix_data_policy_versions(sample_stats_db):
    """Test get_matrix_data with 'policy_versions' view type."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="policy_versions", policy_uri="policy1")

    # Should include all versions of policy1
    assert len(matrix) == 3  # 3 versions of policy1

    # Check that all versions of policy1 are included
    for policy_uri in ["policy1:v1", "policy1:v2", "policy1:v3"]:
        assert policy_uri in matrix.index

    # Check that other policies are excluded
    assert "policy2:v1" not in matrix.index
    assert "policy2:v2" not in matrix.index
    assert "policy3:v1" not in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns

    # Check that the matrix is ordered by version
    assert list(matrix.index) == ["policy1:v1", "policy1:v2", "policy1:v3"]


def test_get_matrix_data_chronological(sample_stats_db):
    """Test get_matrix_data with 'chronological' view type."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="chronological")

    # Should include all versions of all policies
    assert len(matrix) == 6  # 6 total policy versions

    # Check that all versions are included
    for policy_uri in ["policy1:v1", "policy1:v2", "policy1:v3", "policy2:v1", "policy2:v2", "policy3:v1"]:
        assert policy_uri in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns


def test_get_matrix_data_num_output_policies(sample_stats_db):
    """Test get_matrix_data with 'num_output_policies' parameter."""
    all_matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="all")
    limited_matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="all", num_output_policies=2)

    # Check that the limited matrix has only the requested number of policies
    assert len(limited_matrix) == 2
    assert len(all_matrix) == 6

    # The limited matrix should contain the last 2 rows from the all_matrix
    # (highest scoring policies at the end after sorting)
    assert limited_matrix.index.tolist() == all_matrix.index.tolist()[-2:]


def test_get_matrix_data_empty_result(sample_stats_db):
    """Test get_matrix_data when no data is found for the specified metric."""
    matrix = get_matrix_data(sample_stats_db, "nonexistent_metric")

    # Should return an empty DataFrame
    assert matrix.empty


@patch("metta.eval.db.logger")
def test_get_matrix_data_logs_warning_for_empty_result(mock_logger, sample_stats_db):
    """Test that get_matrix_data logs a warning when no data is found."""
    get_matrix_data(sample_stats_db, "nonexistent_metric")

    # Check that a warning was logged
    mock_logger.warning.assert_called_once_with("No data found for metric nonexistent_metric")


def test_get_matrix_data_handles_policy_uri_with_prefix(sample_stats_db):
    """Test that get_matrix_data handles policy URIs with prefixes correctly."""
    # Test with a URI that includes a wandb prefix
    matrix = get_matrix_data(
        sample_stats_db, "episode_reward", view_type="policy_versions", policy_uri="wandb://run/policy1:v2"
    )

    # Should include all versions of policy1
    assert len(matrix) == 3
    assert "policy1:v1" in matrix.index
    assert "policy1:v2" in matrix.index
    assert "policy1:v3" in matrix.index

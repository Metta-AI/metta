"""
Unit tests for get_matrix_data function with StatsDB focusing on different view types.
"""

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from metta.eval.heatmap import get_matrix_data
from metta.sim.stats_db import StatsDB


@pytest.fixture
def sample_stats_db():
    """Create a temporary StatsDB with sample data for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"{uuid.uuid4().hex}.duckdb"
        db = StatsDB(db_path, mode="rwc")

        # Create a simulation for each eval type
        sim_ids = {}
        for eval_name in ["eval1", "eval2", "eval3"]:
            sim_ids[eval_name] = db.ensure_simulation_id(f"test_{eval_name}", "test_suite", f"env_{eval_name}")

        # Create test episodes - one for each simulation (eval type)
        episodes_by_sim = {}
        for eval_name, sim_id in sim_ids.items():
            episodes_by_sim[eval_name] = []
            for i in range(3):  # 3 episodes per simulation
                ep_id = db.get_next_episode_id()
                episodes_by_sim[eval_name].append(ep_id)
                db.con.execute(
                    """
                    INSERT INTO episodes 
                    (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                    """,
                    (ep_id, i, 10, 10, 100, sim_id),
                )

        # Add policy entries - using integer versions instead of string versions
        policies = [
            ("policy1", 1),  # Changed from "v1" to 1
            ("policy1", 2),  # Changed from "v2" to 2
            ("policy1", 3),  # Changed from "v3" to 3
            ("policy2", 1),  # Changed from "v1" to 1
            ("policy2", 2),  # Changed from "v2" to 2
            ("policy3", 1),  # Changed from "v1" to 1
        ]

        db.con.executemany(
            """
            INSERT INTO policies (policy_key, policy_version)
            VALUES (?, ?)
            """,
            policies,
        )

        # Create agent policies - one unique agent ID per policy
        agent_policies = []
        all_episodes = []
        for _, episodes in episodes_by_sim.items():
            all_episodes.extend(episodes)

        # Assign each policy to all episodes
        for policy_idx, (policy_key, policy_version) in enumerate(policies):
            agent_id = policy_idx  # Use policy index as agent ID
            for ep_id in all_episodes:
                agent_policies.append((ep_id, agent_id, policy_key, policy_version))

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
            "policy1:1": {"eval1": 10.0, "eval2": 15.0, "eval3": 20.0},
            "policy1:2": {"eval1": 12.0, "eval2": 17.0, "eval3": 22.0},
            "policy1:3": {"eval1": 14.0, "eval2": 19.0, "eval3": 24.0},
            "policy2:1": {"eval1": 8.0, "eval2": 13.0, "eval3": 18.0},
            "policy2:2": {"eval1": 9.0, "eval2": 14.0, "eval3": 19.0},
            "policy3:1": {"eval1": 7.0, "eval2": 12.0, "eval3": 17.0},
        }

        # Add agent metrics for each policy, eval type, and episode
        for policy_idx, (policy_key, policy_version) in enumerate(policies):
            agent_id = policy_idx
            policy_uri = f"{policy_key}:{policy_version}"

            for eval_name, value in metric_values[policy_uri].items():
                for ep_id in episodes_by_sim[eval_name]:
                    # Add a small random noise to the value
                    metric_value = value + np.random.normal(0, 0.01)

                    db.con.execute(
                        """
                        INSERT INTO agent_metrics
                        (episode_id, agent_id, metric, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep_id, agent_id, "episode_reward", metric_value),
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
    for policy_uri in ["policy1:1", "policy1:2", "policy1:3", "policy2:1", "policy2:2", "policy3:1"]:
        assert policy_uri in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns

    # Verify that the matrix is sorted by overall score (lowest first)
    policies_by_score = list(matrix.index)
    # policy3:1 has lowest overall score
    assert policies_by_score[0] == "policy3:1"
    # policy1:3 has highest overall score
    assert policies_by_score[-1] == "policy1:3"


def test_get_matrix_data_latest(sample_stats_db):
    """Test get_matrix_data with 'latest' view type."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="latest")

    # Should only include the latest version for each policy
    assert len(matrix) == 3  # 3 policies

    # Check that we have the latest versions
    assert "policy1:3" in matrix.index
    assert "policy2:2" in matrix.index
    assert "policy3:1" in matrix.index

    # Check that earlier versions are excluded
    assert "policy1:1" not in matrix.index
    assert "policy1:2" not in matrix.index
    assert "policy2:1" not in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns

    # Verify the policies are in expected order (lowest to highest score)
    policy_order = list(matrix.index)
    assert policy_order[0] == "policy3:1"  # Lowest score
    assert policy_order[-1] == "policy1:3"  # Highest score


def test_get_matrix_data_with_policy_filter_all(sample_stats_db):
    """Test get_matrix_data with 'all' view type and policy filter."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="all", policy_uri="policy1")

    # Should include all versions of policy1
    assert len(matrix) == 3  # 3 versions of policy1

    # Check that all versions of policy1 are included
    for policy_uri in ["policy1:1", "policy1:2", "policy1:3"]:
        assert policy_uri in matrix.index

    # Check that other policies are excluded
    assert "policy2:1" not in matrix.index
    assert "policy2:2" not in matrix.index
    assert "policy3:1" not in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns

    # Check that the matrix is sorted by overall score (lowest to highest)
    policy_order = list(matrix.index)
    assert policy_order[0] == "policy1:1"  # Lowest score among policy1 versions
    assert policy_order[-1] == "policy1:3"  # Highest score among policy1 versions


def test_get_matrix_data_with_policy_filter_latest(sample_stats_db):
    """Test get_matrix_data with 'latest' view type and policy filter."""
    matrix = get_matrix_data(sample_stats_db, "episode_reward", view_type="latest", policy_uri="policy1")

    # Should include only the latest version of policy1
    assert len(matrix) == 1

    # Check that only the latest version is included
    assert "policy1:3" in matrix.index
    assert "policy1:1" not in matrix.index
    assert "policy1:2" not in matrix.index


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
    with pytest.raises(ValueError):
        _ = get_matrix_data(sample_stats_db, "nonexistent_metric")

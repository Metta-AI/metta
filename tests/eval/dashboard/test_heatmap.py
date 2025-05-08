"""
Unit tests for get_heatmap_matrix function with StatsDB focusing on different view types.
"""

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metta.eval.dashboard.heatmap import get_heatmap_matrix
from metta.eval.eval_stats_db import EvalStatsDB


@pytest.fixture
def sample_stats_db():
    """Create a temporary StatsDB with sample data for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"{uuid.uuid4().hex}.duckdb"
        db = EvalStatsDB(db_path)

        # Create a simulation for each eval type
        sim_ids = {}
        for eval_name in ["eval1", "eval2", "eval3"]:
            sim_id = str(uuid.uuid4())
            sim_ids[eval_name] = sim_id
            db.con.execute(
                """
                INSERT INTO simulations 
                (id, name, suite, env, policy_key, policy_version, created_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (sim_id, f"test_{eval_name}", "test_suite", eval_name, "dummy_policy", 1),
            )

        # Create test episodes - one for each simulation (eval type)
        episodes_by_sim = {}
        for eval_name, sim_id in sim_ids.items():
            episodes_by_sim[eval_name] = []
            for _ in range(3):  # 3 episodes per simulation
                ep_id = str(uuid.uuid4())
                episodes_by_sim[eval_name].append(ep_id)
                db.con.execute(
                    """
                    INSERT INTO episodes 
                    (id, step_count, created_at, completed_at, simulation_id, replay_url)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    """,
                    (ep_id, 100, sim_id, None),
                )

        # Add policy entries
        policies = [
            ("policy1", 1),
            ("policy1", 2),
            ("policy1", 3),
            ("policy2", 1),
            ("policy2", 2),
            ("policy3", 1),
        ]

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
            "policy1:1": {"eval1": 0.10, "eval2": 0.15, "eval3": 0.20},
            "policy1:2": {"eval1": 0.12, "eval2": 0.17, "eval3": 0.22},
            "policy1:3": {"eval1": 0.14, "eval2": 0.19, "eval3": 0.24},
            "policy2:1": {"eval1": 0.08, "eval2": 0.13, "eval3": 0.18},
            "policy2:2": {"eval1": 0.09, "eval2": 0.14, "eval3": 0.19},
            "policy3:1": {"eval1": 0.07, "eval2": 0.12, "eval3": 0.17},
        }

        # Add agent metrics for each policy, eval type, and episode
        for policy_idx, (policy_key, policy_version) in enumerate(policies):
            agent_id = policy_idx
            policy_uri = f"{policy_key}:{policy_version}"

            for eval_name, value in metric_values[policy_uri].items():
                for ep_id in episodes_by_sim[eval_name]:
                    # Add a small random noise to the value
                    metric_value = value + np.random.normal(0, 0.001)

                    db.con.execute(
                        """
                        INSERT INTO agent_metrics
                        (episode_id, agent_id, metric, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep_id, agent_id, "episode_reward", metric_value),
                    )

        # Add some replay URLs
        replay_urls = []
        for _, episodes in episodes_by_sim.items():
            for ep_id in episodes:
                replay_url = f"https://example.com/replay/{ep_id}.json"
                replay_urls.append((ep_id, replay_url))

        db.con.executemany(
            """
            UPDATE episodes
            SET replay_url = ?
            WHERE id = ?
            """,
            [(url, ep_id) for ep_id, url in replay_urls],
        )

        # Mock the get_replay_urls method
        def mock_get_replay_urls(policy_key=None, policy_version=None, env=None):
            if policy_key and policy_version and env:
                return [f"https://example.com/replay/{policy_key}/{policy_version}/{env}/replay.json"]
            return []

        # Add the mock method to the database instance
        db.get_replay_urls = mock_get_replay_urls

        db.con.commit()  # Make sure all data is committed
        yield db
        db.close()


def test_get_heatmap_matrix(sample_stats_db):
    matrix = get_heatmap_matrix(sample_stats_db, "episode_reward")

    # Should include all versions of all policies
    assert len(matrix) == 6  # 6 total policy versions

    # Check that all versions are included with correct formatting
    for policy_uri in ["policy1:1", "policy1:2", "policy1:3", "policy2:1", "policy2:2", "policy3:1"]:
        assert policy_uri in matrix.index

    # Check that the matrix includes the Overall column
    assert "Overall" in matrix.columns

    # Check that metrics columns exist
    for col in ["eval1", "eval2", "eval3"]:
        assert col in matrix.columns

    # Check replay URL map
    assert hasattr(matrix, "replay_url_map")
    assert isinstance(matrix.replay_url_map, dict)

    # Verify that the replay URL map has entries for each policy+eval combination
    for policy_uri in matrix.index:
        policy_key, policy_version = policy_uri.split(":")
        for eval_name in ["eval1", "eval2", "eval3"]:
            key = f"{policy_key}|{policy_version}|{eval_name}"
            assert key in matrix.replay_url_map
            assert matrix.replay_url_map[key].startswith("https://example.com/replay/")

    # All policies should be included since our mock doesn't actually filter
    assert len(matrix) == 6
    assert "Overall" in matrix.columns


def test_get_heatmap_matrix_num_output_policies(sample_stats_db):
    """Test get_heatmap_matrix with 'num_output_policies' parameter."""
    all_matrix = get_heatmap_matrix(sample_stats_db, "episode_reward")
    limited_matrix = get_heatmap_matrix(sample_stats_db, "episode_reward", num_output_policies=2)

    # Check that the limited matrix has only the requested number of policies
    assert len(limited_matrix) == 2
    assert len(all_matrix) == 6

    # The limited matrix should contain the 2 policies with highest scores
    # (due to tail() behavior)
    assert limited_matrix.index.tolist() == all_matrix.index.tolist()[-2:]


def test_get_heatmap_matrix_empty_result(sample_stats_db):
    """Test get_heatmap_matrix when no data is found for the specified metric."""
    # Using a non-existent metric should return an empty DataFrame, not raise an error
    matrix = get_heatmap_matrix(sample_stats_db, "nonexistent_metric")
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.empty

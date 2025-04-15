"""
Unit tests for PolicyEvalDB focusing on different view types.
"""

import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from metta.eval.db import PolicyEvalDB

@pytest.fixture
def mock_db_connection():
    """Create a mock database connection for testing."""
    # Use an in-memory SQLite database for testing
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture
def sample_db():
    """Create a temporary database with sample data for testing."""
    # Create a temporary file for the database
    temp_fd, temp_path = tempfile.mkstemp()
    os.close(temp_fd)

    # Create database and initialize schema
    db = PolicyEvalDB(temp_path)

    # Insert test data
    with db.conn:
        # Add policies - WITHOUT the "wandb://run/" prefix to match production behavior
        db.conn.executemany(
            "INSERT INTO policies (uri, version) VALUES (?, ?)",
            [
                ("policy1", "v1"),
                ("policy1", "v2"),
                ("policy1", "v3"),
                ("policy2", "v1"),
                ("policy2", "v2"),
                ("policy3", "v1"),
            ],
        )

        # Add evaluations
        db.conn.executemany(
            "INSERT INTO evaluations (name, metric) VALUES (?, ?)",
            [
                ("eval1", "episode_reward"),
                ("eval2", "episode_reward"),
                ("eval3", "episode_reward"),
            ],
        )

        # Add policy evaluations - also WITHOUT the "wandb://run/" prefix
        db.conn.executemany(
            "INSERT INTO policy_evaluations "
            "(policy_uri, policy_version, evaluation_name, metric, mean, stdev) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                # policy1:v1
                ("policy1", "v1", "eval1", "episode_reward", 10.0, 1.0),
                ("policy1", "v1", "eval2", "episode_reward", 15.0, 1.5),
                ("policy1", "v1", "eval3", "episode_reward", 20.0, 2.0),
                # policy1:v2
                ("policy1", "v2", "eval1", "episode_reward", 12.0, 1.0),
                ("policy1", "v2", "eval2", "episode_reward", 17.0, 1.5),
                ("policy1", "v2", "eval3", "episode_reward", 22.0, 2.0),
                # policy1:v3
                ("policy1", "v3", "eval1", "episode_reward", 14.0, 1.0),
                ("policy1", "v3", "eval2", "episode_reward", 19.0, 1.5),
                ("policy1", "v3", "eval3", "episode_reward", 24.0, 2.0),
                # policy2:v1
                ("policy2", "v1", "eval1", "episode_reward", 8.0, 1.0),
                ("policy2", "v1", "eval2", "episode_reward", 13.0, 1.5),
                ("policy2", "v1", "eval3", "episode_reward", 18.0, 2.0),
                # policy2:v2
                ("policy2", "v2", "eval1", "episode_reward", 9.0, 1.0),
                ("policy2", "v2", "eval2", "episode_reward", 14.0, 1.5),
                ("policy2", "v2", "eval3", "episode_reward", 19.0, 2.0),
                # policy3:v1
                ("policy3", "v1", "eval1", "episode_reward", 7.0, 1.0),
                ("policy3", "v1", "eval2", "episode_reward", 12.0, 1.5),
                ("policy3", "v1", "eval3", "episode_reward", 17.0, 2.0),
            ],
        )

    yield db

    # Clean up
    db.conn.close()
    os.unlink(temp_path)


def test_get_matrix_data_all(sample_db):
    """Test get_matrix_data with 'all' view type."""
    matrix = sample_db.get_matrix_data("episode_reward", view_type="all")

    # Should include all versions of all policies
    assert len(matrix) == 6  # 6 total policy versions

    # Check that all versions are included with correct formatting
    for policy_uri in ["policy1:v1", "policy1:v2", "policy1:v3", "policy2:v1", "policy2:v2", "policy3:v1"]:
        assert policy_uri in matrix.index

    # Check that all evaluation columns are present
    for eval_name in ["eval1", "eval2", "eval3", "Overall"]:
        assert eval_name in matrix.columns

    # Check that the values are correct for a sample entry
    assert matrix.loc["policy1:v2", "eval1"] == 12.0
    assert matrix.loc["policy1:v2", "eval2"] == 17.0
    assert matrix.loc["policy1:v2", "eval3"] == 22.0
    assert matrix.loc["policy1:v2", "Overall"] == (12.0 + 17.0 + 22.0) / 3

    # Check that the matrix is sorted by overall score (lowest first)
    assert list(matrix.index)[0] == "policy3:v1"  # Should be lowest score
    assert list(matrix.index)[-1] == "policy1:v3"  # Should be highest score


def test_get_matrix_data_latest(sample_db):
    """Test get_matrix_data with 'latest' view type."""
    matrix = sample_db.get_matrix_data("episode_reward", view_type="latest")

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

    # Check that all evaluation columns are present
    for eval_name in ["eval1", "eval2", "eval3", "Overall"]:
        assert eval_name in matrix.columns

    # Check that the values are correct
    assert matrix.loc["policy1:v3", "eval1"] == 14.0
    assert matrix.loc["policy1:v3", "eval2"] == 19.0
    assert matrix.loc["policy1:v3", "eval3"] == 24.0
    assert matrix.loc["policy1:v3", "Overall"] == (14.0 + 19.0 + 24.0) / 3

    # Check that the matrix is sorted by overall score (lowest first)
    assert list(matrix.index) == [
        "policy3:v1",  # Average: 12.0
        "policy2:v2",  # Average: 14.0
        "policy1:v3",  # Average: 19.0
    ]


def test_get_matrix_data_sorting(sample_db):
    """Test the sorting behavior of get_matrix_data."""
    # Test with 'latest' view type
    latest_matrix = sample_db.get_matrix_data("episode_reward", view_type="latest")

    # Calculate the overall scores manually
    policy1_v3_avg = (14.0 + 19.0 + 24.0) / 3  # 19.0
    policy2_v2_avg = (9.0 + 14.0 + 19.0) / 3  # 14.0
    policy3_v1_avg = (7.0 + 12.0 + 17.0) / 3  # 12.0

    # Verify overall scores in the matrix
    assert abs(latest_matrix.loc["policy1:v3", "Overall"] - policy1_v3_avg) < 0.01
    assert abs(latest_matrix.loc["policy2:v2", "Overall"] - policy2_v2_avg) < 0.01
    assert abs(latest_matrix.loc["policy3:v1", "Overall"] - policy3_v1_avg) < 0.01

    # Check actual order (should be lowest to highest score)
    actual_order = list(latest_matrix.index)
    expected_order = [
        "policy3:v1",  # Average: 12.0
        "policy2:v2",  # Average: 14.0
        "policy1:v3",  # Average: 19.0
    ]

    assert actual_order == expected_order

    # Test with 'all' view type
    all_matrix = sample_db.get_matrix_data("episode_reward", view_type="all")

    # The expected order by score (lowest to highest) would be:
    # 1. policy3:v1 (12.0)
    # 2. policy2:v1 (13.0)
    # 3. policy2:v2 (14.0)
    # 4. policy1:v1 (15.0)
    # 5. policy1:v2 (17.0)
    # 6. policy1:v3 (19.0)

    all_order = list(all_matrix.index)

    # Check that the lowest scoring policy is first
    assert all_order[0] == "policy3:v1"

    # Check that policy1 versions are ordered by score (lowest to highest)
    policy1_versions = [uri for uri in all_order if uri.startswith("policy1")]
    expected_policy1_order = [
        "policy1:v1",  # 15.0
        "policy1:v2",  # 17.0
        "policy1:v3",  # 19.0
    ]
    assert policy1_versions == expected_policy1_order


def test_get_matrix_data_policy_versions(sample_db):
    """Test get_matrix_data with 'policy_versions' view type."""
    # The policy_versions query returns raw data that needs further processing
    # Let's modify the test to get the raw query result
    sql = """
    SELECT
        p.uri || ':' || p.version as policy_uri,
        pe.evaluation_name,
        pe.mean as value
    FROM policy_evaluations pe
    JOIN policies p ON pe.policy_uri = p.uri AND pe.policy_version = p.version
    WHERE pe.metric = ? AND p.uri = ?
    ORDER BY p.version ASC
    """
    raw_data = sample_db.query(sql, ("episode_reward", "policy1"))

    # Check that we get 9 rows (3 versions x 3 evaluations)
    assert len(raw_data) == 9

    # Now test the get_matrix_data function with the correct URI format
    matrix = sample_db.get_matrix_data("episode_reward", view_type="policy_versions", policy_uri="policy1")

    # Should include all versions of policy1
    assert len(matrix) == 3  # 3 versions of policy1

    # Check that all versions of policy1 are included
    for policy_uri in [
        "policy1:v1",
        "policy1:v2",
        "policy1:v3",
    ]:
        assert policy_uri in matrix.index

    # Check that other policies are excluded
    assert "policy2:v1" not in matrix.index
    assert "policy2:v2" not in matrix.index
    assert "policy3:v1" not in matrix.index

    # Check that all evaluation columns are present
    for eval_name in ["eval1", "eval2", "eval3", "Overall"]:
        assert eval_name in matrix.columns

    # Check that the values are correct
    assert matrix.loc["policy1:v1", "eval1"] == 10.0
    assert matrix.loc["policy1:v2", "eval1"] == 12.0
    assert matrix.loc["policy1:v3", "eval1"] == 14.0

    # Check that the matrix is ordered by version
    assert list(matrix.index) == [
        "policy1:v1",
        "policy1:v2",
        "policy1:v3",
    ]


def test_get_matrix_data_chronological(sample_db):
    """Test get_matrix_data with 'chronological' view type."""
    # Since created_at is set automatically and can't be easily mocked,
    # we'll just verify that the function runs without errors and returns
    # a non-empty dataframe with the expected columns
    matrix = sample_db.get_matrix_data("episode_reward", view_type="chronological")

    # Should include all versions of all policies
    assert len(matrix) == 6  # 6 total policy versions

    # Check that all versions are included
    for policy_uri in ["policy1:v1", "policy1:v2", "policy1:v3", "policy2:v1", "policy2:v2", "policy3:v1"]:
        assert policy_uri in matrix.index

    # Check that all evaluation columns are present
    for eval_name in ["eval1", "eval2", "eval3", "Overall"]:
        assert eval_name in matrix.columns


def test_get_matrix_data_empty_result(sample_db):
    """Test get_matrix_data when no data is found for the specified metric."""
    matrix = sample_db.get_matrix_data("nonexistent_metric")

    # Should return an empty DataFrame
    assert matrix.empty


@patch("metta.eval.db.logger")
def test_get_matrix_data_logs_warning_for_empty_result(mock_logger, sample_db):
    """Test that get_matrix_data logs a warning when no data is found."""
    sample_db.get_matrix_data("nonexistent_metric")

    # Check that a warning was logged
    mock_logger.warning.assert_called_once_with("No data found for metric nonexistent_metric")


def test_parse_versioned_uri():
    """Test the parse_versioned_uri method."""
    db = PolicyEvalDB(":memory:")  # In-memory database for simple test

    # Test with full URI
    uri, version = db.parse_versioned_uri("wandb://run/policy1:v10")
    assert uri == "policy1"
    assert version == "v10"

    # Test with partial URI
    uri, version = db.parse_versioned_uri("policy1:v20")
    assert uri == "policy1"
    assert version == "v20"

    # Test with no version
    uri, version = db.parse_versioned_uri("policy2")
    assert uri == "policy2"
    assert version == "latest"

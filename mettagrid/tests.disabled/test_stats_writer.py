"""
Unit tests for the StatsWriter functionality in mettagrid.stats_writer.
"""

import datetime
import tempfile
import uuid
from pathlib import Path

import pytest

from metta.mettagrid.episode_stats_db import EpisodeStatsDB
from metta.mettagrid.stats_writer import StatsWriter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


def test_stats_writer_initialization(temp_dir):
    """Test that StatsWriter initializes correctly."""
    writer = StatsWriter(temp_dir)
    assert writer.dir == temp_dir
    assert writer.db is None  # Database should be created on demand


def test_ensure_db(temp_dir):
    """Test that _ensure_db creates a database when needed."""
    writer = StatsWriter(temp_dir)
    writer._ensure_db()
    assert writer.db is not None
    assert isinstance(writer.db, EpisodeStatsDB)

    # Check that the database file exists
    db_files = list(temp_dir.glob("*.duckdb"))
    assert len(db_files) == 1

    writer.close()


def test_episode_lifecycle(temp_dir):
    """Test the full lifecycle of an episode."""
    writer = StatsWriter(temp_dir)

    # Create episode ID
    episode_id = str(uuid.uuid4())

    # Episode attributes
    attributes = {"seed": "12345", "map_w": "10", "map_h": "20", "meta": '{"key": "value"}'}

    # Metrics
    agent_metrics = {0: {"reward": 10.5, "steps": 50.0}, 1: {"reward": 8.2, "steps": 45.0}}
    agent_groups = {0: 0, 1: 1}

    # Step count and timestamps
    step_count = 100
    created_at = datetime.datetime.now()
    replay_url = "https://example.com/replay.json"

    # Record the complete episode
    writer.record_episode(episode_id, attributes, agent_metrics, agent_groups, step_count, replay_url, created_at)

    # Verify data in database
    assert writer.db is not None
    db = writer.db

    # Check episode exists
    result = db.con.execute("SELECT id FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    assert result is not None
    assert result[0] == episode_id

    # Check episode attributes
    for attr, value in attributes.items():
        result = db.con.execute(
            "SELECT value FROM episode_attributes WHERE episode_id = ? AND attribute = ?", (episode_id, attr)
        ).fetchone()
        assert result is not None
        assert result[0] == value

    # Check agent metrics
    for agent_id, metrics in agent_metrics.items():
        for metric, value in metrics.items():
            result = db.con.execute(
                "SELECT value FROM agent_metrics WHERE episode_id = ? AND agent_id = ? AND metric = ?",
                (episode_id, agent_id, metric),
            ).fetchone()
            assert result is not None
            assert abs(result[0] - value) < 1e-6  # Compare floats with tolerance

    # Check step count
    result = db.con.execute("SELECT step_count FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    assert result is not None
    assert result[0] == step_count

    # Check replay URL
    result = db.con.execute("SELECT replay_url FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    assert result is not None
    assert result[0] == replay_url

    writer.close()


def test_close_without_db(temp_dir):
    """Test calling close() on a StatsWriter that hasn't created a DB yet."""
    writer = StatsWriter(temp_dir)
    assert writer.db is None

    # This should not raise an error
    writer.close()
    assert writer.db is None

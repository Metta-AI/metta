"""
Unit-tests for the StatsDB functionality in mettagrid.stats_writer.
"""

import tempfile
import uuid
from pathlib import Path

# The updated StatsDB with UUID-based IDs
from mettagrid.stats_writer import StatsDB


def make_tmp_db() -> StatsDB:
    """Return a brand-new writable StatsDB in a temp file."""
    tmp = Path(tempfile.mktemp(suffix=".duckdb"))
    return StatsDB(tmp, read_only=False)  # read_only==False → create schema


def test_uuid_generation():
    """Test that get_next_episode_id generates valid UUIDs."""
    db = make_tmp_db()

    # Generate multiple UUIDs to ensure they're unique
    id1 = db.get_next_episode_id()
    id2 = db.get_next_episode_id()

    # Verify they're valid UUIDs
    assert uuid.UUID(id1)
    assert uuid.UUID(id2)

    # Verify they're unique
    assert id1 != id2

    db.close()


def test_create_and_get_episode():
    """Test creating an episode and retrieving it."""
    db = make_tmp_db()

    # Create an episode
    seed = 12345
    map_w = 10
    map_h = 20
    metadata = {"key": "value"}

    episode_id = db.create_episode(seed, map_w, map_h, metadata)

    # Verify it's a valid UUID
    assert uuid.UUID(episode_id)

    # Verify it was stored correctly
    result = db.con.execute("SELECT seed, map_w, map_h, metadata FROM episodes WHERE id = ?", (episode_id,)).fetchone()

    assert result[0] == seed
    assert result[1] == map_w
    assert result[2] == map_h
    assert "key" in result[3]  # Basic check on metadata

    db.close()


def test_finish_episode():
    """Test marking an episode as finished."""
    db = make_tmp_db()

    # Create an episode
    episode_id = db.create_episode(0, 1, 1)

    # Mark it as finished
    step_count = 100
    db.finish_episode(episode_id, step_count)

    # Verify it was updated correctly
    result = db.con.execute("SELECT step_count, finished_at FROM episodes WHERE id = ?", (episode_id,)).fetchone()

    assert result[0] == step_count
    assert result[1] is not None  # finished_at should be set

    db.close()


def test_add_agent_metrics():
    """Test adding agent metrics."""
    db = make_tmp_db()

    # Create an episode
    episode_id = db.create_episode(0, 1, 1)

    # Add metrics for an agent
    agent_id = 0
    metrics = {"reward": 10.5, "steps": 50}

    db.add_agent_metrics(episode_id, agent_id, metrics)

    # Verify metrics were stored correctly
    for metric, value in metrics.items():
        result = db.con.execute(
            """
            SELECT value FROM agent_metrics 
            WHERE episode_id = ? AND agent_id = ? AND metric = ?
            """,
            (episode_id, agent_id, metric),
        ).fetchone()

        assert result[0] == value

    db.close()

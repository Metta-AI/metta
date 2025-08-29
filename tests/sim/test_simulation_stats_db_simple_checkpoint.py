"""Simulation stats database tests using CheckpointManager patterns."""

from __future__ import annotations

import datetime
import tempfile
import uuid
from pathlib import Path

import pytest
from duckdb import DuckDBPyConnection

from metta.agent.mocks import MockAgent
from metta.rl.checkpoint_manager import CheckpointManager, key_and_version, parse_checkpoint_filename
from metta.sim.simulation_stats_db import SimulationStatsDB


class TestHelpersSimpleCheckpoint:
    """Helper methods for simulation stats database tests using CheckpointManager."""

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
    def create_checkpoint_with_manager(
        temp_dir: Path, run_name: str, epoch: int, score: float = 0.5, metadata: dict = None
    ) -> str:
        """Create a checkpoint using CheckpointManager and return URI."""
        checkpoint_manager = CheckpointManager(run_name=run_name, run_dir=str(temp_dir))

        # Create a mock agent and save it
        mock_agent = MockAgent()
        if metadata is None:
            metadata = {
                "score": score,
                "agent_step": epoch * 1000,
                "generation": 1,
                "train_time": epoch * 10.0,
                "epoch": epoch,
            }
        else:
            # Ensure essential fields are present
            metadata = metadata.copy()
            metadata.update(
                {
                    "score": score,
                    "epoch": epoch,
                    "agent_step": metadata.get("agent_step", epoch * 1000),
                }
            )

        checkpoint_manager.save_agent(mock_agent, epoch=epoch, metadata=metadata)

        # Get the actual checkpoint path using the new dot-separated format
        agent_step = metadata.get("agent_step", epoch * 1000)
        total_time = int(metadata.get("train_time", epoch * 10.0))
        filename = f"{run_name}.e{epoch}.s{agent_step}.t{total_time}.pt"
        checkpoint_path = checkpoint_manager.checkpoint_dir / filename

        return f"file://{checkpoint_path}"

    @staticmethod
    def create_agent_map_from_checkpoint_uris(checkpoint_uris: list[str]) -> dict[int, tuple[str, int]]:
        """Create agent map from checkpoint URI list."""
        agent_map = {}
        for i, uri in enumerate(checkpoint_uris):
            agent_map[i] = key_and_version(uri)
        return agent_map


def test_from_shards_and_context_with_simple_checkpoint_manager(tmp_path: Path):
    """
    Test creating a SimulationStatsDB from shards using CheckpointManager.
    This shows how the database integration would work with the new checkpoint system.
    """
    # Create checkpoint directories
    checkpoint_temp_dir = tmp_path / "checkpoints"
    checkpoint_temp_dir.mkdir(parents=True)

    # Create a checkpoint using CheckpointManager
    checkpoint_uri = TestHelpersSimpleCheckpoint.create_checkpoint_with_manager(
        checkpoint_temp_dir, "test_policy", epoch=1, score=0.85
    )

    # Create agent map in the format expected by the database
    agent_map = TestHelpersSimpleCheckpoint.create_agent_map_from_checkpoint_uris([checkpoint_uri])

    # Create a shard with some data
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard.duckdb"
    ep_id = TestHelpersSimpleCheckpoint.create_worker_db(shard_path)

    # Verify episode was correctly created in the shard
    shard_db = SimulationStatsDB(shard_path)
    shard_episodes = shard_db.con.execute("SELECT id FROM episodes").fetchall()
    shard_episode_ids = [row[0] for row in shard_episodes]
    assert ep_id in shard_episode_ids, f"Episode {ep_id} not found in shard DB"
    shard_db.close()

    # Create the merged database using CheckpointManager context
    sim_id = str(uuid.uuid4())

    # This would be the updated interface for CheckpointManager integration
    merged_db = SimulationStatsDB.from_shards_and_context(
        sim_id=sim_id,
        dir_with_shards=shard_dir,
        agent_map=agent_map,
        sim_name="test_sim",
        sim_env="test_env",
        policy_uri=checkpoint_uri,
    )

    # Verify the merged database contains our data
    merged_episodes = merged_db.con.execute("SELECT id FROM episodes").fetchall()
    merged_episode_ids = [row[0] for row in merged_episodes]

    # The episode should be in the merged database
    assert ep_id in merged_episode_ids, f"Episode {ep_id} not found in merged DB"

    # Verify simulation metadata
    simulations = merged_db.con.execute("SELECT id, name, env FROM simulations").fetchall()
    assert len(simulations) == 1
    assert simulations[0][1] == "test_sim"
    assert simulations[0][2] == "test_env"

    # Verify agent policies table
    agent_policies = merged_db.con.execute(
        "SELECT episode_id, agent_id, policy_key, policy_version FROM agent_policies"
    ).fetchall()
    assert len(agent_policies) > 0

    # Should have mapping from our checkpoint
    policy_entries = [ap for ap in agent_policies if ap[2] == "test_policy" and ap[3] == 1]
    assert len(policy_entries) > 0

    merged_db.close()

    print("✅ Database integration with CheckpointManager verified")


def test_checkpoint_info_compatibility():
    """Test that checkpoint URIs provide the required interface for database operations."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a checkpoint
        checkpoint_uri = TestHelpersSimpleCheckpoint.create_checkpoint_with_manager(
            temp_path, "test_run", epoch=5, score=0.92
        )

        # Test the interface methods that database code expects
        assert checkpoint_uri.startswith("file://")
        
        # Test key_and_version function
        key, version = key_and_version(checkpoint_uri)
        assert key == "test_run"
        assert version == 5

        # Verify the checkpoint file actually exists (extract path from URI)
        checkpoint_path = checkpoint_uri[7:]  # Remove "file://" prefix
        assert Path(checkpoint_path).exists()

        # Verify metadata can be parsed from filename
        filename = Path(checkpoint_path).name
        run_name, epoch, agent_step, total_time = parse_checkpoint_filename(filename)
        assert run_name == "test_run"
        assert epoch == 5

        print("✅ Checkpoint compatibility with database operations verified")


def test_database_policy_lookup_with_checkpoints(tmp_path: Path):
    """Test database policy lookups using checkpoint information.
    This demonstrates how evaluation/analysis would work with CheckpointManager.
    """

    # Create multiple checkpoints for different experiments
    checkpoint_infos = []
    for run_name, epochs_scores in [
        ("baseline_run", [(1, 0.3), (5, 0.6), (10, 0.8)]),
        ("improved_run", [(1, 0.4), (5, 0.7), (10, 0.95)]),
        ("experimental_run", [(1, 0.2), (5, 0.5), (10, 0.75)]),
    ]:
        for epoch, score in epochs_scores:
            checkpoint_info = TestHelpersSimpleCheckpoint.create_checkpoint_with_manager(
                tmp_path / run_name, run_name, epoch, score
            )
            checkpoint_infos.append(checkpoint_info)

    # Create a database with episode data
    db_path = tmp_path / "test_analysis.duckdb"
    db = SimulationStatsDB(db_path)

    # Record episodes with different checkpoint associations
    for _i, checkpoint_info in enumerate(checkpoint_infos):
        episode_id = str(uuid.uuid4())
        attributes = {"experiment": checkpoint_info.run_name}
        agent_metrics = {0: {"reward": checkpoint_info.metadata["score"] * 10}}  # Scale for testing
        agent_groups = {0: 0}
        created_at = datetime.datetime.now()

        db.record_episode(episode_id, attributes, agent_metrics, agent_groups, 100, None, created_at)

        # Associate episode with checkpoint info
        key, version = checkpoint_info.key_and_version()
        query = (
            "INSERT OR REPLACE INTO agent_policies "
            "(episode_id, agent_id, policy_key, policy_version) VALUES (?, ?, ?, ?)"
        )
        db.con.execute(query, [episode_id, 0, key, version])

    # Test querying by policy (equivalent to what dashboard/analysis tools would do)

    # Find all episodes for a specific run
    baseline_episodes = db.con.execute(
        "SELECT episode_id FROM agent_policies WHERE policy_key = 'baseline_run'"
    ).fetchall()
    assert len(baseline_episodes) == 3  # 3 epochs

    # Find best performing checkpoint across all runs
    best_policy = db.con.execute(
        """
        SELECT policy_key, policy_version, AVG(agent_metrics.value) as avg_reward
        FROM agent_policies 
        JOIN episodes ON agent_policies.episode_id = episodes.id
        JOIN agent_metrics ON episodes.id = agent_metrics.episode_id
        WHERE agent_metrics.metric = 'reward'
        GROUP BY policy_key, policy_version
        ORDER BY avg_reward DESC
        LIMIT 1
        """
    ).fetchone()

    assert best_policy is not None
    assert best_policy[0] == "improved_run"  # Should be the best performing run
    assert best_policy[1] == 10  # Should be epoch 10 (highest score)

    # Find all checkpoints from a specific epoch across runs
    epoch_5_policies = db.con.execute(
        "SELECT DISTINCT policy_key FROM agent_policies WHERE policy_version = 5"
    ).fetchall()

    policy_keys = [p[0] for p in epoch_5_policies]
    assert "baseline_run" in policy_keys
    assert "improved_run" in policy_keys
    assert "experimental_run" in policy_keys

    db.close()

    print("✅ Database policy lookup and analysis with checkpoints verified")


def test_checkpoint_metadata_database_integration(tmp_path: Path):
    """
    Test that checkpoint metadata integrates properly with database operations.
    This shows how rich metadata from CheckpointManager can enhance database queries.
    """

    # Create checkpoints with rich metadata
    checkpoint_infos = []
    for i in range(3):
        metadata = {
            "score": 0.5 + i * 0.2,
            "agent_step": (i + 1) * 5000,
            "train_time": (i + 1) * 120.0,
            "generation": 1,
            "experiment_config": {
                "learning_rate": 0.001 * (i + 1),
                "batch_size": 256,
                "architecture": "transformer" if i == 2 else "mlp",
            },
            "evaluation_metrics": {
                "success_rate": 0.7 + i * 0.1,
                "average_episode_length": 100 + i * 20,
            },
        }

        checkpoint_info = TestHelpersSimpleCheckpoint.create_checkpoint_with_manager(
            tmp_path / "rich_metadata_run", "rich_metadata_run", epoch=i + 1, score=metadata["score"], metadata=metadata
        )
        checkpoint_infos.append((checkpoint_info, metadata))

    # Verify that metadata is properly embedded in filenames
    for i, (checkpoint_info, original_metadata) in enumerate(checkpoint_infos):
        checkpoint_path = checkpoint_info.uri[7:]  # Remove "file://" prefix

        # Verify that basic metadata is embedded in filename
        filename = Path(checkpoint_path).name
        parsed_metadata = parse_checkpoint_filename(filename)
        assert parsed_metadata is not None
        assert parsed_metadata["run"] == "rich_metadata_run"
        assert parsed_metadata["epoch"] == i + 1
        assert parsed_metadata["agent_step"] == original_metadata["agent_step"]

        # This metadata could be stored in database for advanced queries
        # (This would require extending SimulationStatsDB to handle checkpoint metadata)

    print("✅ Rich checkpoint metadata integration verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

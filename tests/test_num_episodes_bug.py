#!/usr/bin/env python3

import tempfile
import uuid
from pathlib import Path

from metta.eval.eval_stats_db import EvalStatsDB
from softmax.training.rl.checkpoint_manager import CheckpointManager


def create_test_database(
    db_path: Path,
    num_episodes_requested: int,
    num_episodes_completed: int,
    num_agents: int = 2,
    checkpoint_filename: str = "test_policy/checkpoints/test_policy:v1.pt",
):
    """Create a test database that simulates the bug scenario."""
    db = EvalStatsDB(db_path)

    sim_id = uuid.uuid4().hex[:8]
    metadata = CheckpointManager.get_policy_metadata(f"file:///tmp/{checkpoint_filename}")
    policy_key, policy_version = metadata["run_name"], metadata["epoch"]

    db.con.execute(
        "INSERT INTO simulations (id, name, env, policy_key, policy_version) VALUES (?, ?, ?, ?, ?)",
        (sim_id, "test_sim", "test_env", policy_key, policy_version),
    )

    episode_ids = [f"episode_{i}" for i in range(num_episodes_requested)]
    for episode_id in episode_ids:
        db.con.execute(
            "INSERT INTO episodes (id, simulation_id, step_count) VALUES (?, ?, ?)",
            (episode_id, sim_id, 100),
        )

    for episode_id in episode_ids:
        for agent_id in range(num_agents):
            db.con.execute(
                "INSERT INTO agent_policies (episode_id, agent_id, policy_key, policy_version) VALUES (?, ?, ?, ?)",
                (episode_id, agent_id, policy_key, policy_version),
            )

    for i in range(num_episodes_completed):
        episode_id = f"episode_{i}"
        for agent_id in range(num_agents):
            db.con.execute(
                "INSERT INTO agent_metrics (episode_id, agent_id, metric, value) VALUES (?, ?, ?, ?)",
                (episode_id, agent_id, "reward", 1.0),
            )

    db.con.commit()
    return db


def test_normalization_bug():
    """Test normalization bug: rewards should not be diluted by incomplete episodes."""
    checkpoint_filename = "test_policy/checkpoints/test_policy:v1.pt"

    with tempfile.TemporaryDirectory() as tmpdir:
        db1 = create_test_database(Path(tmpdir) / "test1.duckdb", num_episodes_requested=1, num_episodes_completed=1)
        db2 = create_test_database(Path(tmpdir) / "test2.duckdb", num_episodes_requested=5, num_episodes_completed=2)

        policy_uri = CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}")
        avg_reward_complete = db1.get_average_metric("reward", policy_uri)
        avg_reward_partial = db2.get_average_metric("reward", policy_uri)

        if avg_reward_complete and avg_reward_partial:
            ratio = avg_reward_partial / avg_reward_complete
            assert ratio >= 0.99, f"Normalization bug detected: {ratio:.3f} ratio (should be ~1.0)"

        db1.close()
        db2.close()


if __name__ == "__main__":
    test_normalization_bug()

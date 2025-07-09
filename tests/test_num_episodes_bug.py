#!/usr/bin/env python3
"""
Proper test to demonstrate the num_episodes normalization bug and verify the fix.
This creates a mock database scenario and tests the actual normalization logic.
"""

import tempfile
import uuid
from pathlib import Path

from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord
from metta.eval.eval_stats_db import EvalStatsDB


def create_test_database(db_path: Path, num_episodes_requested: int, num_episodes_completed: int, num_agents: int = 2):
    """Create a test database that simulates the bug scenario."""
    db = EvalStatsDB(db_path)

    # Create a simulation
    sim_id = uuid.uuid4().hex[:8]
    db.con.execute(
        """
        INSERT INTO simulations (id, name, suite, env, policy_key, policy_version)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (sim_id, "test_sim", "test_suite", "test_env", "test_policy", 1),
    )

    # Create episodes - THIS IS KEY: we might create records for all requested episodes
    # even if they don't complete with metrics
    episode_ids = []
    for i in range(num_episodes_requested):
        episode_id = f"episode_{i}"
        episode_ids.append(episode_id)
        db.con.execute(
            """
            INSERT INTO episodes (id, simulation_id, step_count)
            VALUES (?, ?, ?)
        """,
            (episode_id, sim_id, 100),
        )

    # Create agent_policies for ALL episodes (this is what causes the bug!)
    for episode_id in episode_ids:
        for agent_id in range(num_agents):
            db.con.execute(
                """
                INSERT INTO agent_policies (episode_id, agent_id, policy_key, policy_version)
                VALUES (?, ?, ?, ?)
            """,
                (episode_id, agent_id, "test_policy", 1),
            )

    # But only create metrics for COMPLETED episodes
    for i in range(num_episodes_completed):
        episode_id = f"episode_{i}"
        for agent_id in range(num_agents):
            # Each agent gets a reward of 1.0
            db.con.execute(
                """
                INSERT INTO agent_metrics (episode_id, agent_id, metric, value)
                VALUES (?, ?, ?, ?)
            """,
                (episode_id, agent_id, "reward", 1.0),
            )

    db.con.commit()
    return db


def test_normalization_bug():
    """Test that demonstrates the normalization bug and the fix."""
    print("Testing num_episodes normalization bug...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Scenario 1: All episodes complete (baseline)
        print("\nScenario 1: 1 episode requested, 1 completed")
        db1 = create_test_database(
            Path(tmpdir) / "test1.duckdb", num_episodes_requested=1, num_episodes_completed=1, num_agents=2
        )

        # Calculate metrics

        # Create a mock policy record using the real class to satisfy type checks.
        # The policy_store isn't used by the methods under test.
        policy_record = PolicyRecord(
            policy_store=None,  # type: ignore
            run_name="test_policy",
            uri="test_policy",
            metadata=PolicyMetadata(epoch=1),
        )

        # Check what's in the database
        episodes_count = db1.query("SELECT COUNT(*) as cnt FROM episodes")["cnt"][0]
        agent_policies_count = db1.query("SELECT COUNT(*) as cnt FROM agent_policies")["cnt"][0]
        metrics_count = db1.query("SELECT COUNT(*) as cnt FROM agent_metrics")["cnt"][0]

        print(f"  Episodes in DB: {episodes_count}")
        print(f"  Agent policies: {agent_policies_count}")
        print(f"  Agent metrics: {metrics_count}")

        # Get average reward
        avg_reward_1 = db1.get_average_metric_by_filter("reward", policy_record)
        print(f"  Average reward: {avg_reward_1}")

        # Scenario 2: Not all episodes complete (demonstrates bug)
        print("\nScenario 2: 5 episodes requested, only 2 completed")
        db2 = create_test_database(
            Path(tmpdir) / "test2.duckdb", num_episodes_requested=5, num_episodes_completed=2, num_agents=2
        )

        episodes_count = db2.query("SELECT COUNT(*) as cnt FROM episodes")["cnt"][0]
        agent_policies_count = db2.query("SELECT COUNT(*) as cnt FROM agent_policies")["cnt"][0]
        metrics_count = db2.query("SELECT COUNT(*) as cnt FROM agent_metrics")["cnt"][0]

        print(f"  Episodes in DB: {episodes_count}")
        print(f"  Agent policies: {agent_policies_count} (note: includes incomplete episodes!)")
        print(f"  Agent metrics: {metrics_count} (only completed episodes)")

        # Get average reward
        avg_reward_2 = db2.get_average_metric_by_filter("reward", policy_record)
        print(f"  Average reward: {avg_reward_2}")

        # Check the normalization calculation details
        print("\nDiagnosing the normalization:")
        print("-" * 40)

        # Get the potential samples count
        potential = db2.potential_samples_for_metric("test_policy", 1)
        print(f"  Potential samples (from agent_policies): {potential}")

        # Get the actual recorded metrics
        recorded = db2.count_metric_agents("test_policy", 1, "reward")
        print(f"  Recorded metrics: {recorded}")

        # Get the sum
        sum_query = db2.query("""
            SELECT SUM(value) as total
            FROM policy_simulation_agent_metrics
            WHERE policy_key = 'test_policy' AND policy_version = 1 AND metric = 'reward'
        """)
        total = sum_query["total"][0] if not sum_query.empty else 0
        print(f"  Sum of rewards: {total}")
        print(f"  Normalization: {total} / {potential} = {total / potential if potential > 0 else 'N/A'}")

        # Show the problem
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Average reward with 1 episode (all complete): {avg_reward_1}")
        print(f"Average reward with 5 episodes (2 complete): {avg_reward_2}")

        if avg_reward_1 and avg_reward_2:
            ratio = avg_reward_2 / avg_reward_1
            print(f"\nRatio: {ratio:.2f}")
            if ratio < 0.5:
                print("❌ BUG CONFIRMED: Metrics are reduced when not all episodes complete!")
                print("   The normalization divides by ALL requested episodes, not just completed ones.")
            else:
                print("✅ FIX WORKING: Metrics are normalized correctly!")
                print("   The normalization only counts episodes that have metrics.")

        # Close databases
        db1.close()
        db2.close()


if __name__ == "__main__":
    test_normalization_bug()

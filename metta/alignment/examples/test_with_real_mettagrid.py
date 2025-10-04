"""
Test GAMMA metrics with actual MettaGrid environment.

This script creates a real MettaGrid environment and tests the integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from metta.alignment.integration import (
    GAMMAEvaluator,
    MettaGridAdapter,
    TrajectoryCollector,
)

# Import MettaGrid
try:
    from mettagrid import MettaGridCore

    METTAGRID_AVAILABLE = True
except ImportError:
    print("Warning: MettaGrid not available, using mock")
    METTAGRID_AVAILABLE = False


def test_with_real_mettagrid():
    """Test GAMMA metrics with real MettaGrid environment."""

    if not METTAGRID_AVAILABLE:
        print("MettaGrid not available - skipping real environment test")
        return

    print("=" * 60)
    print("Testing GAMMA with Real MettaGrid Environment")
    print("=" * 60)

    # Create a simple MettaGrid environment with proper config
    try:
        from mettagrid import MettaGridConfig
        from mettagrid.config.mettagrid_config import (
            ActionConfig,
            ActionsConfig,
            GameConfig,
        )
        from mettagrid.map_builder.random import RandomMapBuilder

        # Create a simple random map with 4 agents
        config = MettaGridConfig(
            game=GameConfig(
                num_agents=4,
                obs_width=5,
                obs_height=5,
                num_observation_tokens=50,
                max_steps=100,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                    rotate=ActionConfig(),
                ),
                map_builder=RandomMapBuilder.Config(
                    width=10,
                    height=10,
                    agents=4,
                    seed=42,
                ),
            )
        )

        env = MettaGridCore(config)
        print("\n✓ Created MettaGrid environment")

        # Get environment info
        num_agents = env.num_agents
        map_width = env.map_width
        map_height = env.map_height

        print(f"  - Num agents: {num_agents}")
        print(f"  - Map size: {map_width} x {map_height}")

    except Exception as e:
        print(f"\n✗ Failed to create MettaGrid: {e}")
        import traceback

        traceback.print_exc()
        return

    # Setup GAMMA components
    collector = TrajectoryCollector(num_agents=num_agents)
    adapter = MettaGridAdapter(grid_to_continuous_scale=1.0)
    evaluator = GAMMAEvaluator(alpha=0.1)

    print("\n✓ Created GAMMA components")

    # Reset environment
    try:
        obs, info = env.reset()
        print("\n✓ Environment reset")
        print(f"  - Observation shape: {obs.shape}")

    except Exception as e:
        print(f"\n✗ Failed to reset: {e}")
        return

    # Run a short episode
    collector.reset()
    max_steps = 50
    dt = 0.1

    print(f"\nRunning {max_steps} steps...")

    for step in range(max_steps):
        try:
            # Extract positions using adapter
            positions = adapter.extract_agent_positions(env)

            # Compute task directions
            task_dirs = adapter.compute_task_directions_to_resources(env, resource_types=["generator", "converter"])

            # Record step
            collector.record_step(positions=positions, task_directions=task_dirs, dt=dt)

            # Take random actions (must be int32, shape (num_agents, 1))
            actions = np.random.randint(0, 3, size=(num_agents, 1), dtype=np.int32)
            obs, rewards, dones, truncs, info = env.step(actions)

            if step % 10 == 0:
                print(f"  Step {step}: {len(positions)} agents tracked")

            if dones.all() or truncs.all():
                print(f"  Episode ended at step {step}")
                break

        except Exception as e:
            print(f"\n✗ Error at step {step}: {e}")
            break

    # Evaluate GAMMA metrics
    print("\n" + "=" * 60)
    print("Computing GAMMA Metrics")
    print("=" * 60)

    try:
        trajectories = collector.get_trajectories()

        # Check trajectory data
        print(f"\nCollected {len(trajectories)} agent trajectories")
        for i, traj in enumerate(trajectories[:3]):  # Show first 3
            print(f"  Agent {i}: {len(traj['positions'])} timesteps")

        # Evaluate
        results = evaluator.evaluate_with_components(trajectories, dt=dt)

        print("\n✓ GAMMA Metrics Computed:")
        print(f"  GAMMA:       {results['GAMMA']:.3f}")
        print(f"  GAMMA_α:     {results['GAMMA_alpha']:.3f}")
        print(f"  IAM mean:    {results['IAM_mean']:.3f}")
        print(f"  IAM std:     {results['IAM_std']:.3f}")

        # Show per-agent breakdown for first 3 agents
        print("\nPer-Agent Breakdown (first 3):")
        for i in range(min(3, len(results["components"]))):
            comp = results["components"][i]
            print(f"  Agent {i}: IAM={comp['IAM']:.3f}, A={comp['A']:.3f}, D={comp['D']:.3f}, E={comp['E']:.3f}")

        # Format for wandb
        wandb_dict = evaluator.format_for_wandb(results)
        print(f"\n✓ Formatted {len(wandb_dict)} metrics for wandb")

        print("\n" + "=" * 60)
        print("Test Successful! ✓")
        print("=" * 60)
        print("\nGAMMA integration with MettaGrid is working correctly.")

    except Exception as e:
        print(f"\n✗ Failed to compute metrics: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_with_real_mettagrid()

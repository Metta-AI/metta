"""
Test GAMMA metrics with actual MettaGrid environment.

This script creates a real MettaGrid environment and tests the integration.

Usage:
    uv run python metta/alignment/examples/test_with_real_mettagrid.py
    uv run python metta/alignment/examples/test_with_real_mettagrid.py --num_steps 100 --num_agents 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib.pyplot as plt
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
    print("Success: MettaGrid AVAILABLE")
except ImportError:
    print("Warning: MettaGrid not available, using mock")
    METTAGRID_AVAILABLE = False


def test_with_real_mettagrid(num_agents: int = 4, num_steps: int = 50, map_size: int = 10, seed: int = 42):
    """
    Test GAMMA metrics with real MettaGrid environment.

    Args:
        num_agents: Number of agents in the environment
        num_steps: Number of steps to run
        map_size: Size of the square map
        seed: Random seed for reproducibility
    """

    if not METTAGRID_AVAILABLE:
        print("MettaGrid not available - skipping real environment test")
        return

    print("=" * 60)
    print("Testing GAMMA with Real MettaGrid Environment")
    print("=" * 60)
    print(f"Config: {num_agents} agents, {num_steps} steps, {map_size}x{map_size} map, seed={seed}")

    # Create a simple MettaGrid environment with proper config
    try:
        from mettagrid import MettaGridConfig
        from mettagrid.config.mettagrid_config import (
            ActionConfig,
            ActionsConfig,
            GameConfig,
        )
        from mettagrid.map_builder.random import RandomMapBuilder

        # Create environment with specified parameters
        config = MettaGridConfig(
            game=GameConfig(
                num_agents=num_agents,
                obs_width=5,
                obs_height=5,
                num_observation_tokens=50,
                max_steps=num_steps * 2,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                    rotate=ActionConfig(),
                ),
                map_builder=RandomMapBuilder.Config(
                    width=map_size,
                    height=map_size,
                    agents=num_agents,
                    seed=seed,
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

    # Run episode
    collector.reset()
    dt = 0.1

    # Define goals for agents (center of map)
    center = np.array([map_width / 2, map_height / 2])
    goals = np.tile(center, (num_agents, 1))

    print(f"\nRunning {num_steps} steps...")
    print(f"  Goal: All agents move toward center at {center}")

    for step in range(num_steps):
        try:
            # Extract positions using adapter
            positions = adapter.extract_agent_positions(env)

            # Compute task directions toward center
            task_dirs = adapter.compute_task_directions_to_goal(positions, goals)

            # Debug: print first step info
            if step == 0:
                print(f"\n  Debug info:")
                print(f"    Positions shape: {positions.shape}")
                print(f"    Task dirs shape: {task_dirs.shape}")
                print(f"    Sample position: {positions[0]}")
                print(f"    Sample task dir: {task_dirs[0]}")
                print(f"    Task dir norm: {np.linalg.norm(task_dirs[0]):.3f}")

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

        # Evaluate (pass goals for proper goal attainment)
        results = evaluator.evaluate_with_components(trajectories, dt=dt, goals=list(goals))

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

        # Visualize trajectories
        print("\n" + "=" * 60)
        print("Generating Visualization")
        print("=" * 60)

        visualize_trajectories(trajectories, goals, map_width, map_height, results)

    except Exception as e:
        print(f"\n✗ Failed to compute metrics: {e}")
        import traceback

        traceback.print_exc()


def visualize_trajectories(trajectories, goals, map_width, map_height, results):
    """Visualize agent trajectories and alignment metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Trajectories on map
    ax1 = axes[0, 0]
    ax1.set_title("Agent Trajectories on Map", fontsize=14, fontweight="bold")
    ax1.set_xlim(-0.5, map_width - 0.5)
    ax1.set_ylim(-0.5, map_height - 0.5)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X (columns)")
    ax1.set_ylabel("Y (rows)")

    # Plot goal
    ax1.scatter(goals[0, 0], goals[0, 1], s=300, marker="*", color="gold",
               edgecolor="orange", linewidth=2, label="Goal", zorder=10)

    # Plot trajectories
    colors = ["red", "blue", "green", "purple"]
    for i, traj in enumerate(trajectories):
        if len(traj["positions"]) > 0:
            pos = traj["positions"]
            ax1.plot(pos[:, 0], pos[:, 1], "-", color=colors[i % len(colors)],
                    linewidth=2, alpha=0.7, label=f"Agent {i}")
            ax1.scatter(pos[0, 0], pos[0, 1], s=100, color=colors[i % len(colors)],
                       marker="o", edgecolor="black", zorder=5)
            ax1.scatter(pos[-1, 0], pos[-1, 1], s=100, color=colors[i % len(colors)],
                       marker="s", edgecolor="black", zorder=5)

    ax1.legend(loc="upper right")

    # Panel 2: Velocity vectors at key timesteps
    ax2 = axes[0, 1]
    ax2.set_title("Velocity Vectors (every 10 steps)", fontsize=14, fontweight="bold")
    ax2.set_xlim(-0.5, map_width - 0.5)
    ax2.set_ylim(-0.5, map_height - 0.5)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Plot goal
    ax2.scatter(goals[0, 0], goals[0, 1], s=300, marker="*", color="gold",
               edgecolor="orange", linewidth=2, zorder=10)

    for i, traj in enumerate(trajectories):
        if len(traj["positions"]) > 0:
            pos = traj["positions"]
            vel = traj["velocities"]
            task_dir = traj["task_directions"]

            # Plot every 10th step
            for t in range(0, len(pos), 10):
                # Velocity arrow
                ax2.arrow(pos[t, 0], pos[t, 1], vel[t, 0] * 2, vel[t, 1] * 2,
                         head_width=0.3, head_length=0.2, fc=colors[i % len(colors)],
                         ec=colors[i % len(colors)], alpha=0.6, linewidth=1.5)
                # Task direction (dashed)
                ax2.arrow(pos[t, 0], pos[t, 1], task_dir[t, 0] * 1.5, task_dir[t, 1] * 1.5,
                         head_width=0.2, head_length=0.15, fc="gray", ec="gray",
                         alpha=0.4, linestyle="--", linewidth=1)

    # Panel 3: Metric breakdown per agent
    ax3 = axes[1, 0]
    ax3.set_title("Alignment Metrics per Agent", fontsize=14, fontweight="bold")

    if "components" in results and len(results["components"]) > 0:
        agents = list(range(len(results["components"])))
        metrics = ["A", "D", "E", "T", "Y"]

        x = np.arange(len(agents))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [results["components"][j][metric] for j in agents]
            ax3.bar(x + i * width, values, width, label=metric, alpha=0.8)

        ax3.set_xlabel("Agent ID")
        ax3.set_ylabel("Metric Value")
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels(agents)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.set_ylim(0, 1.1)

    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("GAMMA Summary", fontsize=14, fontweight="bold")

    summary_text = [
        "Collective Metrics:",
        f"  GAMMA:       {results['GAMMA']:.3f}",
        f"  GAMMA_α:     {results['GAMMA_alpha']:.3f}",
        f"  IAM mean:    {results['IAM_mean']:.3f}",
        f"  IAM std:     {results['IAM_std']:.3f}",
        f"  CV:          {results['CV']:.3f}",
        "",
        "Individual Scores:",
    ]

    if "IAM_scores" in results:
        for i, score in enumerate(results["IAM_scores"]):
            summary_text.append(f"  Agent {i}: {score:.3f}")

    summary_text.extend([
        "",
        "Interpretation:",
        "• Low scores expected for random agents",
        "• D≈0: No directional intent",
        "• E<0.5: Inefficient wandering paths",
        "• A<0.1: Far from goal",
        "",
        "With trained agents, expect:",
        "• GAMMA > 0.7",
        "• D > 0.8, E > 0.7, A > 0.8",
    ])

    y_pos = 0.95
    for line in summary_text:
        if line.startswith("  "):
            ax4.text(0.1, y_pos, line, fontsize=10, family="monospace",
                    transform=ax4.transAxes, va="top")
        else:
            ax4.text(0.05, y_pos, line, fontsize=11, fontweight="bold" if ":" in line else "normal",
                    transform=ax4.transAxes, va="top")
        y_pos -= 0.04

    plt.tight_layout()
    plt.savefig("gamma_mettagrid_test.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved visualization to: gamma_mettagrid_test.png")
    plt.show()


def create_step_by_step_animation(trajectories, goals, map_width, map_height):
    """Create a step-by-step animation showing agent movement."""
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, map_width - 0.5)
    ax.set_ylim(-0.5, map_height - 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Agent Movement Over Time", fontsize=14, fontweight="bold")

    # Plot goal
    ax.scatter(goals[0, 0], goals[0, 1], s=300, marker="*", color="gold",
              edgecolor="orange", linewidth=2, label="Goal", zorder=10)

    colors = ["red", "blue", "green", "purple"]
    max_steps = max(len(traj["positions"]) for traj in trajectories if len(traj["positions"]) > 0)

    # Initialize agent markers and trails
    agent_markers = []
    agent_trails = []

    for i, traj in enumerate(trajectories):
        if len(traj["positions"]) > 0:
            marker, = ax.plot([], [], "o", color=colors[i % len(colors)],
                            markersize=15, label=f"Agent {i}")
            trail, = ax.plot([], [], "-", color=colors[i % len(colors)],
                           alpha=0.5, linewidth=2)
            agent_markers.append(marker)
            agent_trails.append(trail)
        else:
            agent_markers.append(None)
            agent_trails.append(None)

    ax.legend(loc="upper right")
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                       fontsize=12, verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def update(frame):
        step_text.set_text(f"Step: {frame}/{max_steps-1}")

        for i, traj in enumerate(trajectories):
            if len(traj["positions"]) > frame and agent_markers[i] is not None:
                pos = traj["positions"]
                # Update marker position
                agent_markers[i].set_data([pos[frame, 0]], [pos[frame, 1]])
                # Update trail
                agent_trails[i].set_data(pos[:frame+1, 0], pos[:frame+1, 1])

        return agent_markers + agent_trails + [step_text]

    anim = FuncAnimation(fig, update, frames=max_steps, interval=100, blit=True)
    anim.save("gamma_mettagrid_animation.gif", writer="pillow", fps=10)
    print("\n✓ Saved animation to: gamma_mettagrid_animation.gif")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GAMMA metrics with MettaGrid")
    parser.add_argument("--num_agents", type=int, default=4, help="Number of agents (default: 4)")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of steps (default: 50)")
    parser.add_argument("--map_size", type=int, default=10, help="Map size (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    test_with_real_mettagrid(
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        map_size=args.map_size,
        seed=args.seed
    )

"""
Demonstrate HIGH alignment scores with goal-directed agents.

This uses pure simulation (no MettaGrid) to show what well-aligned agents look like.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from metta.alignment.integration import GAMMAEvaluator, TrajectoryCollector


def simulate_aligned_agents(num_agents=4, num_steps=100, noise_level=0.05):
    """Simulate agents that move directly toward their goals."""

    print("=" * 60)
    print("GAMMA Demo: HIGH Alignment (Goal-Directed Agents)")
    print("=" * 60)
    print(f"Simulating {num_agents} agents over {num_steps} steps")
    print(f"Noise level: {noise_level}")

    # Setup
    collector = TrajectoryCollector(num_agents=num_agents)
    evaluator = GAMMAEvaluator(alpha=0.1)
    collector.reset()
    dt = 0.1

    # Initial positions (spread out)
    np.random.seed(42)
    positions = np.random.rand(num_agents, 2) * 10

    # Goals (center)
    center = np.array([5.0, 5.0])
    goals = np.tile(center, (num_agents, 1))

    print(f"\nGoal: All agents move to center at {center}")
    print("\nInitial distances:")
    for i in range(num_agents):
        dist = np.linalg.norm(positions[i] - center)
        print(f"  Agent {i}: {dist:.2f}")

    # Simulate
    for step in range(num_steps):
        # Compute task directions
        task_dirs = goals - positions
        task_norms = np.linalg.norm(task_dirs, axis=1, keepdims=True)
        task_dirs = task_dirs / np.maximum(task_norms, 1e-8)

        # Move toward goal (90% fidelity to task direction)
        velocities = task_dirs * 0.5

        # Add small noise
        if noise_level > 0:
            velocities += np.random.randn(num_agents, 2) * noise_level

        # Record
        collector.record_step(positions.copy(), task_directions=task_dirs, dt=dt)

        # Update positions
        positions = positions + velocities * dt

        if step % 25 == 0:
            avg_dist = np.mean(np.linalg.norm(positions - goals, axis=1))
            print(f"  Step {step:3d}: avg distance = {avg_dist:.2f}")

    print("\nFinal distances:")
    for i in range(num_agents):
        dist = np.linalg.norm(positions[i] - center)
        print(f"  Agent {i}: {dist:.2f}")

    # Evaluate
    print("\n" + "=" * 60)
    print("GAMMA Metrics")
    print("=" * 60)

    trajectories = collector.get_trajectories()
    results = evaluator.evaluate_with_components(trajectories, dt=dt, goals=list(goals))

    print("\n✓ Collective Metrics:")
    print(f"  GAMMA:       {results['GAMMA']:.3f} {'✓ HIGH' if results['GAMMA'] > 0.7 else '✗ LOW'}")
    print(f"  GAMMA_α:     {results['GAMMA_alpha']:.3f}")
    print(f"  IAM mean:    {results['IAM_mean']:.3f}")
    print(f"  IAM std:     {results['IAM_std']:.3f}")
    print(f"  CV:          {results['CV']:.3f}")

    print("\n✓ Component Averages:")
    avg_A = np.mean([c["A"] for c in results["components"]])
    avg_D = np.mean([c["D"] for c in results["components"]])
    avg_E = np.mean([c["E"] for c in results["components"]])
    avg_T = np.mean([c["T"] for c in results["components"]])
    avg_Y = np.mean([c["Y"] for c in results["components"]])

    print(f"  Goal Attainment (A):      {avg_A:.3f} {'✓' if avg_A > 0.7 else '✗'}")
    print(f"  Directional Intent (D):   {avg_D:.3f} {'✓' if avg_D > 0.7 else '✗'}")
    print(f"  Path Efficiency (E):      {avg_E:.3f} {'✓' if avg_E > 0.7 else '✗'}")
    print(f"  Time Efficiency (T):      {avg_T:.3f} {'✓' if avg_T > 0.7 else '✗'}")
    print(f"  Energy Proportionality (Y): {avg_Y:.3f}")

    print("\n✓ Per-Agent IAM Scores:")
    for i, score in enumerate(results["IAM_scores"]):
        print(f"  Agent {i}: {score:.3f}")

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison with Random Agents")
    print("=" * 60)
    print("Random agents:        GAMMA ≈ 0.001, D ≈ 0.04, E ≈ 0.2, A ≈ 0.01")
    print(f"Goal-directed agents: GAMMA = {results['GAMMA']:.3f}, D = {avg_D:.3f}, E = {avg_E:.3f}, A = {avg_A:.3f}")
    print(f"\nImprovement factor: {results['GAMMA'] / 0.001:.0f}x higher alignment!")

    # Visualize
    visualize_results(trajectories, goals, results)

    return results


def visualize_results(trajectories, goals, results):
    """Create visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Trajectories
    ax1 = axes[0, 0]
    ax1.set_title("Agent Trajectories (Goal-Directed)", fontsize=14, fontweight="bold")
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Goal
    ax1.scatter(
        goals[0, 0],
        goals[0, 1],
        s=500,
        marker="*",
        color="gold",
        edgecolor="orange",
        linewidth=3,
        label="Goal",
        zorder=10,
    )
    ax1.add_patch(plt.Circle(goals[0], 0.5, fill=False, edgecolor="orange", linestyle="--", linewidth=2, alpha=0.5))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
    for i, (traj, color) in enumerate(zip(trajectories, colors, strict=False)):
        pos = traj["positions"]
        ax1.plot(pos[:, 0], pos[:, 1], "-", color=color, linewidth=2.5, alpha=0.8)
        ax1.scatter(pos[0, 0], pos[0, 1], s=150, color=color, marker="o", edgecolor="black", linewidth=2, zorder=5)
        ax1.scatter(pos[-1, 0], pos[-1, 1], s=150, color=color, marker="s", edgecolor="black", linewidth=2, zorder=5)
        ax1.text(pos[0, 0], pos[0, 1] - 0.5, f"A{i}", fontsize=10, ha="center")

    ax1.legend(loc="upper left", fontsize=10)

    # Panel 2: Distance convergence
    ax2 = axes[0, 1]
    ax2.set_title("Convergence to Goal", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance to Goal")
    ax2.grid(True, alpha=0.3)

    for i, (traj, color) in enumerate(zip(trajectories, colors, strict=False)):
        pos = traj["positions"]
        distances = np.linalg.norm(pos - goals[i], axis=1)
        ax2.plot(distances, color=color, linewidth=2, alpha=0.8, label=f"Agent {i}")

    ax2.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Success threshold")
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Metrics
    ax3 = axes[1, 0]
    ax3.set_title("Alignment Metrics (All 5 Components)", fontsize=14, fontweight="bold")

    agents = list(range(len(results["components"])))
    metrics = ["A", "D", "E", "T", "Y"]
    metric_names = ["Goal\nAttain", "Direct\nIntent", "Path\nEffic", "Time\nEffic", "Energy\nProp"]

    x = np.arange(len(agents))
    width = 0.15

    for i, (metric, name) in enumerate(zip(metrics, metric_names, strict=False)):
        values = [results["components"][j][metric] for j in agents]
        ax3.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax3.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, linewidth=2)
    ax3.text(len(agents) - 0.5, 0.72, "Good threshold", fontsize=9, color="green")
    ax3.set_xlabel("Agent ID")
    ax3.set_ylabel("Metric Value")
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(agents)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0, 1.1)

    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    gamma_val = results["GAMMA"]
    if gamma_val > 0.7:
        status = "HIGH ALIGNMENT ✓"
        status_color = "green"
    elif gamma_val > 0.4:
        status = "MODERATE ALIGNMENT ○"
        status_color = "orange"
    else:
        status = "LOW ALIGNMENT ✗"
        status_color = "red"

    ax4.text(
        0.5,
        0.95,
        status,
        transform=ax4.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="center",
        color=status_color,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=status_color, linewidth=3),
    )

    summary = [
        "",
        f"GAMMA = {gamma_val:.3f}",
        f"IAM mean = {results['IAM_mean']:.3f}",
        "",
        "Component Averages:",
        f"  A (Goal Attainment) = {np.mean([c['A'] for c in results['components']]):.3f}",
        f"  D (Directional Intent) = {np.mean([c['D'] for c in results['components']]):.3f}",
        f"  E (Path Efficiency) = {np.mean([c['E'] for c in results['components']]):.3f}",
        f"  T (Time Efficiency) = {np.mean([c['T'] for c in results['components']]):.3f}",
        f"  Y (Energy Prop.) = {np.mean([c['Y'] for c in results['components']]):.3f}",
    ]

    y_pos = 0.75
    for line in summary:
        ax4.text(
            0.5,
            y_pos,
            line,
            transform=ax4.transAxes,
            fontsize=11,
            ha="center",
            va="top",
            family="monospace" if "=" in line else "sans-serif",
        )
        y_pos -= 0.06

    plt.tight_layout()
    plt.savefig("gamma_aligned_agents.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved visualization to: gamma_aligned_agents.png")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.05)

    args = parser.parse_args()

    simulate_aligned_agents(args.num_agents, args.num_steps, args.noise)

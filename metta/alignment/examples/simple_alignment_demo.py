"""
Simple demonstration of alignment metrics.

This script shows how to use the GAMMA framework to evaluate agent alignment
in a simple goal-reaching task.
"""

import matplotlib.pyplot as plt
import numpy as np

from metta.alignment.metrics import (
    DirectionalIntentMetric,
    EnergyProportionalityMetric,
    GoalAttainmentMetric,
    IndividualAlignmentMetric,
    PathEfficiencyMetric,
    TimeEfficiencyMetric,
)
from metta.alignment.metrics.gamma import GAMMAMetric
from metta.alignment.task_interfaces import SetpointTask


def simulate_agent(
    start: np.ndarray,
    goal: np.ndarray,
    T: int,
    dt: float,
    noise_level: float = 0.0,
    adversarial: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a simple agent moving toward a goal.

    Args:
        start: Starting position
        goal: Goal position
        T: Number of timesteps
        dt: Time step size
        noise_level: Amount of random noise to add
        adversarial: If True, agent moves in circles

    Returns:
        Tuple of (positions, velocities, task_directions)
    """
    positions = np.zeros((T, 2))
    velocities = np.zeros((T, 2))
    task_directions = np.zeros((T, 2))

    positions[0] = start

    task = SetpointTask(goal=goal, tolerance=0.1)

    for t in range(T):
        # Get task direction
        task_directions[t] = task.get_task_direction(positions[t], t * dt)

        if adversarial:
            # Move in circles instead of toward goal
            angle = 2 * np.pi * t / T
            velocities[t] = np.array([np.cos(angle), np.sin(angle)]) * 0.5
        else:
            # Move toward goal with noise
            velocities[t] = task_directions[t] * 0.5
            if noise_level > 0:
                velocities[t] += np.random.randn(2) * noise_level

        # Update position
        if t < T - 1:
            positions[t + 1] = positions[t] + velocities[t] * dt

    return positions, velocities, task_directions


def main() -> None:
    """Run alignment metric demonstration."""
    print("=" * 60)
    print("GAMMA Alignment Metrics Demo")
    print("=" * 60)

    # Simulation parameters
    T = 100
    dt = 0.1
    start = np.array([0.0, 0.0])
    goal = np.array([5.0, 5.0])

    # Create three agents with different behaviors
    print("\nSimulating three agents:")
    print("  1. Aligned agent (moves toward goal)")
    print("  2. Noisy agent (moves toward goal with noise)")
    print("  3. Adversarial agent (moves in circles)")

    # Simulate agents
    pos1, vel1, task1 = simulate_agent(start, goal, T, dt, noise_level=0.0)
    pos2, vel2, task2 = simulate_agent(start, goal, T, dt, noise_level=0.1)
    pos3, vel3, task3 = simulate_agent(start, goal, T, dt, adversarial=True)

    # Compute individual metrics
    print("\n" + "=" * 60)
    print("Individual Metrics")
    print("=" * 60)

    di_metric = DirectionalIntentMetric(tolerance=0.05)
    pe_metric = PathEfficiencyMetric()
    ga_metric = GoalAttainmentMetric(scale=1.0)
    te_metric = TimeEfficiencyMetric(baseline_speed=0.5)
    ep_metric = EnergyProportionalityMetric(beta=1.0)
    iam_metric = IndividualAlignmentMetric(scale=1.0, tolerance=0.05, baseline_speed=0.5, beta=1.0)

    agents = [
        ("Aligned", pos1, vel1, task1),
        ("Noisy", pos2, vel2, task2),
        ("Adversarial", pos3, vel3, task3),
    ]

    for name, pos, vel, task_dir in agents:
        print(f"\n{name} Agent:")

        D_i = di_metric.compute(pos, vel, task_dir, dt)
        E_i = pe_metric.compute(pos, vel, task_dir, dt)
        A_i = ga_metric.compute(pos, vel, task_dir, dt, goal=goal)
        T_i = te_metric.compute(pos, vel, task_dir, dt, goal=goal)
        Y_i = ep_metric.compute(pos, vel, task_dir, dt)
        IAM_i = iam_metric.compute(pos, vel, task_dir, dt, goal=goal)

        print(f"  Directional Intent (D):   {D_i:.3f}")
        print(f"  Path Efficiency (E):      {E_i:.3f}")
        print(f"  Goal Attainment (A):      {A_i:.3f}")
        print(f"  Time Efficiency (T):      {T_i:.3f}")
        print(f"  Energy Proportionality (Y): {Y_i:.3f}")
        print(f"  Individual Alignment (IAM): {IAM_i:.3f}")

        # Misalignment detectors
        anti_progress = di_metric.compute_anti_progress_mass(vel, task_dir, dt)
        loopiness = pe_metric.compute_loopiness(pos, vel, dt)
        print(f"  Anti-progress mass:       {anti_progress:.3f}")
        print(f"  Loopiness:                {loopiness:.3f}")

    # Compute collective GAMMA
    print("\n" + "=" * 60)
    print("Collective Metrics (GAMMA)")
    print("=" * 60)

    gamma_metric = GAMMAMetric(alpha=0.1)

    agent_trajectories = [
        {"positions": pos1, "velocities": vel1, "task_directions": task1},
        {"positions": pos2, "velocities": vel2, "task_directions": task2},
        {"positions": pos3, "velocities": vel3, "task_directions": task3},
    ]

    results = gamma_metric.compute(agent_trajectories, dt, goals=[goal, goal, goal])

    print(f"\nGAMMA (collective alignment):     {results['GAMMA']:.3f}")
    print(f"GAMMA_α (dispersion-penalized):   {results['GAMMA_alpha']:.3f}")
    print(f"Mean IAM:                         {results['IAM_mean']:.3f}")
    print(f"Std IAM:                          {results['IAM_std']:.3f}")
    print(f"Coefficient of Variation:         {results['CV']:.3f}")

    # Visualization
    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, pos, vel, _task_dir) in enumerate(agents):
        ax = axes[idx]

        # Plot trajectory
        ax.plot(pos[:, 0], pos[:, 1], "b-", linewidth=2, label="Trajectory")
        ax.scatter(start[0], start[1], color="green", s=100, marker="o", label="Start", zorder=5)
        ax.scatter(goal[0], goal[1], color="red", s=100, marker="*", label="Goal", zorder=5)
        ax.scatter(pos[-1, 0], pos[-1, 1], color="orange", s=100, marker="s", label="End", zorder=5)

        # Plot some velocity vectors
        skip = 10
        for t in range(0, len(pos), skip):
            ax.arrow(
                pos[t, 0],
                pos[t, 1],
                vel[t, 0] * 0.5,
                vel[t, 1] * 0.5,
                head_width=0.2,
                head_length=0.1,
                fc="gray",
                ec="gray",
                alpha=0.5,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{name} Agent")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("alignment_demo.png", dpi=150)
    print("\nSaved visualization to: alignment_demo.png")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

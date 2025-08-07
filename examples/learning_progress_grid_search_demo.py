#!/usr/bin/env python3
"""
Learning Progress Grid Search Demo

This script performs a grid search across the two dominant timescale hyperparameters
of learning progress curriculum:
1. ema_timescale: Controls the exponential moving average timescale for tracking learning progress
2. progress_smoothing: Controls the smoothing applied to progress values

The grid search explores 9 parameter combinations (3x3 grid) and compares performance.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the enhanced oracle
from metta.rl.enhanced_oracle import create_enhanced_oracle_from_demo_tasks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def create_dependency_graph() -> Tuple[nx.DiGraph, Dict[str, int]]:
    """Create a sequential chain dependency graph for curriculum tasks."""
    G = nx.DiGraph()

    # Define 10 tasks in a sequential chain: A -> B -> C -> D -> E -> F -> G -> H -> I -> J
    tasks = {
        "A": [],  # Root task
        "B": ["A"],  # Depends on A
        "C": ["B"],  # Depends on B
        "D": ["C"],  # Depends on C
        "E": ["D"],  # Depends on D
        "F": ["E"],  # Depends on E
        "G": ["F"],  # Depends on F
        "H": ["G"],  # Depends on G
        "I": ["H"],  # Depends on H
        "J": ["I"],  # Depends on I
    }

    # Add nodes and edges
    for task, deps in tasks.items():
        G.add_node(task)
        for dep in deps:
            G.add_edge(dep, task)

    # Calculate dependency depth for each task
    dependency_depths = {}
    for task in tasks:
        depth = len(list(nx.ancestors(G, task)))
        dependency_depths[task] = depth

    return G, dependency_depths


def simulate_learning_progress_curriculum(
    num_epochs: int = 150, ema_timescale: float = 0.001, progress_smoothing: float = 0.05
) -> Dict[str, List[float]]:
    """Simulate learning progress curriculum with specific hyperparameters."""
    tasks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    performance_data = {}

    # Create enhanced oracle to get realistic learning curves
    enhanced_oracle = create_enhanced_oracle_from_demo_tasks()
    learning_curves = enhanced_oracle.learning_curves

    # Initialize learning progress tracking
    task_success_rates = {task: [] for task in tasks}
    task_weights = {task: 1.0 / len(tasks) for task in tasks}  # Uniform initial weights

    # Track fast and slow moving averages for each task
    p_fast = {task: 0.0 for task in tasks}
    p_slow = {task: 0.0 for task in tasks}
    p_true = {task: 0.0 for task in tasks}

    for task in tasks:
        # Get learning curve for this task
        learning_curve = learning_curves[task]

        # Learning progress curriculum modifications
        learning_curve.learning_rate *= 1.2  # 20% faster learning
        learning_curve.noise_scale *= 0.8  # Less noise

        # Add dependency bonus for learning progress
        task_depth = ord(task) - ord("A")  # Calculate task depth
        if task_depth > 0:
            learning_curve.max_performance *= 1.05  # 5% higher max performance for dependent tasks

        # Generate performance curve influenced by curriculum sampling
        performances = []
        for epoch in range(num_epochs):
            # Simulate task completion and success rate
            base_performance = learning_curve.predict_performance(epoch)

            # Add curriculum-specific noise based on hyperparameters
            # Higher ema_timescale = more responsive = more noise
            # Higher progress_smoothing = more stable = less noise
            ema_noise = np.random.normal(0, ema_timescale * 10)  # Noise proportional to EMA timescale
            smoothing_noise = np.random.normal(
                0, max(0.001, (1.0 - progress_smoothing)) * 0.05
            )  # Noise inversely proportional to smoothing, with minimum

            # Apply curriculum weight influence with more realistic scaling
            weight_bonus = (task_weights[task] - 1.0 / len(tasks)) * 0.3  # Reduced from 0.5 to 0.3
            performance = base_performance + weight_bonus + ema_noise + smoothing_noise

            # Scale to 0-100 range
            performance = np.clip(performance, 0.0, 1.0) * 100
            performances.append(performance)

            # Update learning progress tracking (simulate the real curriculum behavior)
            success_rate = performance / 100.0  # Convert to 0-1 range
            task_success_rates[task].append(success_rate)

            # Update moving averages (simulating the real ema_timescale behavior)
            if epoch == 0:
                p_fast[task] = success_rate
                p_slow[task] = success_rate
                p_true[task] = success_rate
            else:
                # Fast EMA
                p_fast[task] = (success_rate * ema_timescale) + (p_fast[task] * (1.0 - ema_timescale))
                # Slow EMA (double smoothing)
                p_slow[task] = (p_fast[task] * ema_timescale) + (p_slow[task] * (1.0 - ema_timescale))
                # True performance EMA
                p_true[task] = (success_rate * ema_timescale) + (p_true[task] * (1.0 - ema_timescale))

            # Update task weights based on learning progress (every 5 epochs instead of 10)
            if epoch % 5 == 0 and epoch > 0:
                # Calculate learning progress as difference between fast and slow EMAs
                learning_progress = abs(p_fast[task] - p_slow[task])

                # Apply progress smoothing (simulating the real progress_smoothing behavior)
                if p_true[task] > 0:
                    # Apply the reweighting formula from the real implementation
                    numerator = learning_progress * (1.0 - progress_smoothing)
                    denominator = learning_progress + progress_smoothing * (1.0 - 2.0 * learning_progress)
                    if denominator > 0:
                        smoothed_progress = numerator / denominator
                    else:
                        smoothed_progress = learning_progress
                else:
                    smoothed_progress = learning_progress

                # Update task weight based on learning progress with more realistic scaling
                task_weights[task] = max(0.01, smoothed_progress * 2.0 + 0.05)  # More realistic scaling

        # Normalize task weights
        total_weight = sum(task_weights.values())
        if total_weight > 0:
            task_weights = {k: v / total_weight for k, v in task_weights.items()}

        performance_data[task] = performances

    return performance_data


def smooth_performance_data(performance_data: Dict[str, List[float]], window_size: int = 5) -> Dict[str, List[float]]:
    """Apply moving average smoothing to performance data."""
    smoothed_data = {}

    for task, performances in performance_data.items():
        smoothed_performances = []
        for i in range(len(performances)):
            # Calculate moving average
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(performances), i + window_size // 2 + 1)
            window = performances[start_idx:end_idx]
            smoothed_performances.append(np.mean(window))

        smoothed_data[task] = smoothed_performances

    return smoothed_data


def evaluate_curriculum_performance(performance_data: Dict[str, List[float]]) -> Dict[str, float]:
    """Evaluate curriculum performance metrics."""
    # Calculate final performances
    final_performances = {task: performances[-1] for task, performances in performance_data.items()}

    # Calculate cumulative efficiency
    cumulative_efficiency = sum([sum(performances) for performances in performance_data.values()])

    # Calculate average final performance
    avg_final_performance = np.mean(list(final_performances.values()))

    # Calculate performance variance (lower is better)
    performance_variance = np.var(list(final_performances.values()))

    # Calculate learning consistency (how evenly tasks improve)
    learning_consistency = 1.0 / (1.0 + performance_variance)  # Higher is better

    return {
        "avg_final_performance": avg_final_performance,
        "cumulative_efficiency": cumulative_efficiency,
        "performance_variance": performance_variance,
        "learning_consistency": learning_consistency,
        "final_performances": final_performances,
    }


def create_grid_search_visualization(
    grid_results: Dict[Tuple[float, float], Dict[str, float]],
    ema_timescales: List[float],
    progress_smoothings: List[float],
) -> None:
    """Create visualization of grid search results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Average Final Performance Heatmap
    ax1 = axes[0, 0]
    performance_matrix = np.zeros((len(progress_smoothings), len(ema_timescales)))
    for i, smoothing in enumerate(progress_smoothings):
        for j, timescale in enumerate(ema_timescales):
            performance_matrix[i, j] = grid_results[(timescale, smoothing)]["avg_final_performance"]

    im1 = ax1.imshow(performance_matrix, cmap="viridis", aspect="auto")
    ax1.set_title("Average Final Performance", fontsize=14, fontweight="bold")
    ax1.set_xlabel("EMA Timescale (log scale)")
    ax1.set_ylabel("Progress Smoothing (log scale)")

    # Set log-spaced tick labels
    x_ticks = np.logspace(np.log10(ema_timescales[0]), np.log10(ema_timescales[-1]), 5)
    y_ticks = np.logspace(np.log10(progress_smoothings[0]), np.log10(progress_smoothings[-1]), 5)

    # Find closest indices for tick positions
    x_tick_indices = [np.argmin(np.abs(ema_timescales - x)) for x in x_ticks]
    y_tick_indices = [np.argmin(np.abs(progress_smoothings - y)) for y in y_ticks]

    ax1.set_xticks(x_tick_indices)
    ax1.set_yticks(y_tick_indices)
    ax1.set_xticklabels([f"{x:.4f}" for x in x_ticks])
    ax1.set_yticklabels([f"{y:.3f}" for y in y_ticks])
    plt.colorbar(im1, ax=ax1)

    # 2. Cumulative Efficiency Heatmap
    ax2 = axes[0, 1]
    efficiency_matrix = np.zeros((len(progress_smoothings), len(ema_timescales)))
    for i, smoothing in enumerate(progress_smoothings):
        for j, timescale in enumerate(ema_timescales):
            efficiency_matrix[i, j] = grid_results[(timescale, smoothing)]["cumulative_efficiency"]

    im2 = ax2.imshow(efficiency_matrix, cmap="plasma", aspect="auto")
    ax2.set_title("Cumulative Efficiency", fontsize=14, fontweight="bold")
    ax2.set_xlabel("EMA Timescale (log scale)")
    ax2.set_ylabel("Progress Smoothing (log scale)")
    ax2.set_xticks(x_tick_indices)
    ax2.set_yticks(y_tick_indices)
    ax2.set_xticklabels([f"{x:.4f}" for x in x_ticks])
    ax2.set_yticklabels([f"{y:.3f}" for y in y_ticks])
    plt.colorbar(im2, ax=ax2)

    # 3. Learning Consistency Heatmap
    ax3 = axes[1, 0]
    consistency_matrix = np.zeros((len(progress_smoothings), len(ema_timescales)))
    for i, smoothing in enumerate(progress_smoothings):
        for j, timescale in enumerate(ema_timescales):
            consistency_matrix[i, j] = grid_results[(timescale, smoothing)]["learning_consistency"]

    im3 = ax3.imshow(consistency_matrix, cmap="coolwarm", aspect="auto")
    ax3.set_title("Learning Consistency", fontsize=14, fontweight="bold")
    ax3.set_xlabel("EMA Timescale (log scale)")
    ax3.set_ylabel("Progress Smoothing (log scale)")
    ax3.set_xticks(x_tick_indices)
    ax3.set_yticks(y_tick_indices)
    ax3.set_xticklabels([f"{x:.4f}" for x in x_ticks])
    ax3.set_yticklabels([f"{y:.3f}" for y in y_ticks])
    plt.colorbar(im3, ax=ax3)

    # 4. Performance Variance Heatmap (lower is better)
    ax4 = axes[1, 1]
    variance_matrix = np.zeros((len(progress_smoothings), len(ema_timescales)))
    for i, smoothing in enumerate(progress_smoothings):
        for j, timescale in enumerate(ema_timescales):
            variance_matrix[i, j] = grid_results[(timescale, smoothing)]["performance_variance"]

    im4 = ax4.imshow(variance_matrix, cmap="Reds", aspect="auto")
    ax4.set_title("Performance Variance (Lower is Better)", fontsize=14, fontweight="bold")
    ax4.set_xlabel("EMA Timescale (log scale)")
    ax4.set_ylabel("Progress Smoothing (log scale)")
    ax4.set_xticks(x_tick_indices)
    ax4.set_yticks(y_tick_indices)
    ax4.set_xticklabels([f"{x:.4f}" for x in x_ticks])
    ax4.set_yticklabels([f"{y:.3f}" for y in y_ticks])
    plt.colorbar(im4, ax=ax4)

    plt.tight_layout()
    plt.savefig("learning_progress_grid_search.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_learning_progress_grid_search():
    """Run the learning progress grid search demo."""
    logger.info("Starting Learning Progress Grid Search Demo...")

    # Set up parameters
    num_epochs = 150
    G, dependency_depths = create_dependency_graph()

    # Define grid search parameters with 30x30 logarithmic spacing across expanded range
    # EMA timescale: from 0.00001 to 0.1 (4 orders of magnitude)
    ema_timescales = np.logspace(-5, -1, 30)  # 30 values from 10^-5 to 10^-1

    # Progress smoothing: from 0.0001 to 0.1 (3 orders of magnitude, smaller values)
    progress_smoothings = np.logspace(-4, -1, 30)  # 30 values from 10^-4 to 10^-1

    # Store grid search results
    grid_results = {}

    print("=" * 80)
    print("LEARNING PROGRESS GRID SEARCH")
    print("=" * 80)
    print(f"EMA Timescales: {len(ema_timescales)} values from {ema_timescales[0]:.6f} to {ema_timescales[-1]:.6f}")
    print(
        f"Progress Smoothings: {len(progress_smoothings)} values from "
        f"{progress_smoothings[0]:.6f} to {progress_smoothings[-1]:.6f}"
    )
    print(f"Total Parameter Combinations: {len(ema_timescales) * len(progress_smoothings)}")
    print()

    # Run grid search
    total_combinations = len(ema_timescales) * len(progress_smoothings)
    current_combination = 0

    for _i, ema_timescale in enumerate(ema_timescales):
        for _j, progress_smoothing in enumerate(progress_smoothings):
            current_combination += 1
            print(
                f"Progress: {current_combination}/{total_combinations} "
                f"({current_combination / total_combinations * 100:.1f}%)"
            )
            print(f"Testing: EMA Timescale = {ema_timescale:.6f}, Progress Smoothing = {progress_smoothing:.6f}")

            # Generate performance data with these parameters
            performance_data = simulate_learning_progress_curriculum(num_epochs, ema_timescale, progress_smoothing)

            # Apply smoothing
            smoothed_data = smooth_performance_data(performance_data, window_size=5)

            # Evaluate performance
            metrics = evaluate_curriculum_performance(smoothed_data)

            # Store results
            grid_results[(ema_timescale, progress_smoothing)] = metrics

            print(f"  Avg Final Performance: {metrics['avg_final_performance']:.2f}")
            print(f"  Cumulative Efficiency: {metrics['cumulative_efficiency']:.2f}")
            print(f"  Learning Consistency: {metrics['learning_consistency']:.4f}")
            print(f"  Performance Variance: {metrics['performance_variance']:.2f}")
            print()

    # Find best parameters
    best_performance = max(grid_results.values(), key=lambda x: x["avg_final_performance"])
    best_efficiency = max(grid_results.values(), key=lambda x: x["cumulative_efficiency"])
    best_consistency = max(grid_results.values(), key=lambda x: x["learning_consistency"])

    # Find best overall (balanced metric)
    best_overall = max(
        grid_results.values(),
        key=lambda x: x["avg_final_performance"] * 0.4
        + (x["cumulative_efficiency"] / 100000) * 0.3
        + x["learning_consistency"] * 0.3,
    )

    print("=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)

    # Print best parameters for each metric
    for (timescale, smoothing), metrics in grid_results.items():
        if metrics == best_performance:
            print(
                f"Best Performance: EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
                f"{metrics['avg_final_performance']:.2f}"
            )
        if metrics == best_efficiency:
            print(
                f"Best Efficiency: EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
                f"{metrics['cumulative_efficiency']:.2f}"
            )
        if metrics == best_consistency:
            print(
                f"Best Consistency: EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
                f"{metrics['learning_consistency']:.4f}"
            )
        if metrics == best_overall:
            print(f"Best Overall: EMA={timescale:.6f}, Smoothing={smoothing:.6f}")

    print()

    # Print top 10 results for each metric
    print("TOP 10 PERFORMANCE RESULTS:")
    sorted_performance = sorted(grid_results.items(), key=lambda x: x[1]["avg_final_performance"], reverse=True)
    for i, ((timescale, smoothing), metrics) in enumerate(sorted_performance[:10]):
        print(
            f"{i + 1:2d}. EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
            f"Performance: {metrics['avg_final_performance']:.2f}"
        )

    print("\nTOP 10 EFFICIENCY RESULTS:")
    sorted_efficiency = sorted(grid_results.items(), key=lambda x: x[1]["cumulative_efficiency"], reverse=True)
    for i, ((timescale, smoothing), metrics) in enumerate(sorted_efficiency[:10]):
        print(
            f"{i + 1:2d}. EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
            f"Efficiency: {metrics['cumulative_efficiency']:.2f}"
        )

    print("\nTOP 10 CONSISTENCY RESULTS:")
    sorted_consistency = sorted(grid_results.items(), key=lambda x: x[1]["learning_consistency"], reverse=True)
    for i, ((timescale, smoothing), metrics) in enumerate(sorted_consistency[:10]):
        print(
            f"{i + 1:2d}. EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
            f"Consistency: {metrics['learning_consistency']:.4f}"
        )

    # Create visualization
    create_grid_search_visualization(grid_results, ema_timescales, progress_smoothings)

    print("\nResults saved to: learning_progress_grid_search.png")
    logger.info("Learning Progress Grid Search Demo completed!")


def main():
    """Main function to run the learning progress grid search demo."""
    try:
        run_learning_progress_grid_search()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

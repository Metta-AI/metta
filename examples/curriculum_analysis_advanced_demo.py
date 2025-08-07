#!/usr/bin/env python3
"""
Advanced Curriculum Analysis Demo

This script demonstrates curriculum analysis with specific visualizations:
1. Dependency graph visualization as a tree with colored task points
2. Performance curves for each task with matching colors
3. Task sampling probabilities over time with filled graph visualization
4. Efficiency comparison with oracle model
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


def generate_task_performance_data(
    num_epochs: int = 100, curriculum_type: str = "learning_progress"
) -> Dict[str, List[float]]:
    """Generate realistic performance data for each task over epochs."""
    tasks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    performance_data = {}

    for task in tasks:
        # Base performance varies by task difficulty
        base_performance = 0.2 + 0.1 * np.random.random()

        # Learning curve parameters based on dependency depth
        dependency_depth = ord(task) - ord("A")  # A=0, B=1, C=2, etc.

        if dependency_depth == 0:  # Task A - easiest
            learning_rate = 0.8
            inflection_point = 15
        elif dependency_depth == 1:  # Task B
            learning_rate = 0.75
            inflection_point = 20
        elif dependency_depth == 2:  # Task C
            learning_rate = 0.7
            inflection_point = 25
        elif dependency_depth == 3:  # Task D
            learning_rate = 0.65
            inflection_point = 30
        elif dependency_depth == 4:  # Task E
            learning_rate = 0.6
            inflection_point = 35
        elif dependency_depth == 5:  # Task F
            learning_rate = 0.55
            inflection_point = 40
        elif dependency_depth == 6:  # Task G
            learning_rate = 0.5
            inflection_point = 45
        elif dependency_depth == 7:  # Task H
            learning_rate = 0.45
            inflection_point = 50
        elif dependency_depth == 8:  # Task I
            learning_rate = 0.4
            inflection_point = 55
        else:  # Task J - hardest
            learning_rate = 0.35
            inflection_point = 60

        # Curriculum-specific modifications
        if curriculum_type == "random":
            # Random curriculum has slower learning and more variance
            learning_rate *= 0.7  # 30% slower learning
            inflection_point += 10  # Delayed inflection point
            noise_multiplier = 1.5  # More noise
        elif curriculum_type == "prioritize_regressed":
            # Prioritize regressed curriculum shows more variance and recovery patterns
            # Add periodic regression and recovery cycles
            noise_multiplier = 1.2  # More noise than learning progress
        else:  # learning_progress
            noise_multiplier = 1.0

        # Generate performance curve
        performances = []
        for epoch in range(num_epochs):
            # Sigmoid learning curve
            learning_progress = learning_rate * (1 / (1 + np.exp(-(epoch - inflection_point) / 10)))

            # Add noise that decreases over time
            noise_scale = max(0.02, 0.1 * np.exp(-epoch / 50)) * noise_multiplier
            noise = np.random.normal(0, noise_scale)

            # Add curriculum-specific patterns
            if curriculum_type == "prioritize_regressed":
                # Add periodic regression and recovery cycles
                regression_cycle = np.sin(epoch / 20) * 0.1
                recovery_bonus = 0.05 * (1 - np.exp(-epoch / 60))
                noise += regression_cycle + recovery_bonus

            # Calculate final performance
            performance = base_performance + learning_progress + noise
            performance = max(0.0, min(1.0, performance))
            performances.append(performance * 100)  # Scale to 0-100

        performance_data[task] = performances

    return performance_data


def generate_task_probabilities(num_epochs: int = 100) -> Dict[str, List[float]]:
    """Generate task sampling probabilities over time."""
    tasks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    probability_data = {}

    for task in tasks:
        # Base probability varies by task position in chain
        dependency_depth = ord(task) - ord("A")  # A=0, B=1, C=2, etc.

        if dependency_depth == 0:  # Task A
            base_prob = 0.20  # Root task gets sampled more
        elif dependency_depth == 1:  # Task B
            base_prob = 0.15
        elif dependency_depth == 2:  # Task C
            base_prob = 0.13
        elif dependency_depth == 3:  # Task D
            base_prob = 0.12
        elif dependency_depth == 4:  # Task E
            base_prob = 0.11
        elif dependency_depth == 5:  # Task F
            base_prob = 0.10
        elif dependency_depth == 6:  # Task G
            base_prob = 0.09
        elif dependency_depth == 7:  # Task H
            base_prob = 0.08
        elif dependency_depth == 8:  # Task I
            base_prob = 0.07
        else:  # Task J
            base_prob = 0.05

        # Add some variation over time
        probabilities = []
        for epoch in range(num_epochs):
            # Add some curriculum learning effect
            if epoch < 20:
                # Early epochs favor easier tasks (A, B, C)
                if dependency_depth <= 2:
                    prob = base_prob * 1.5
                else:
                    prob = base_prob * 0.5
            elif epoch < 60:
                # Middle epochs balance tasks
                prob = base_prob
            else:
                # Late epochs favor harder tasks (H, I, J)
                if dependency_depth >= 7:
                    prob = base_prob * 1.3
                else:
                    prob = base_prob * 0.7

            # Add some noise
            noise = np.random.normal(0, 0.02)
            prob = max(0.01, min(0.4, prob + noise))
            probabilities.append(prob)

        probability_data[task] = probabilities

    # Normalize probabilities to sum to 1.0 at each epoch
    epochs = range(num_epochs)
    for epoch in epochs:
        total_prob = sum(probability_data[task][epoch] for task in tasks)
        for task in tasks:
            probability_data[task][epoch] /= total_prob

    return probability_data


def create_dependency_visualization(G: nx.DiGraph, dependency_depths: Dict[str, int], ax: plt.Axes) -> None:
    """Create dependency graph visualization as a straight line."""

    # Create a linear layout - tasks in a straight line
    tasks = sorted(G.nodes())  # A, B, C, D, E, F, G, H, I, J
    pos = {}
    for i, task in enumerate(tasks):
        pos[task] = (i, 0)  # All tasks on the same horizontal line

    # Create color map based on dependency depth
    max_depth = max(dependency_depths.values())
    colors = plt.cm.viridis([dependency_depths[task] / max_depth for task in G.nodes()])

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1000, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    ax.set_title("Task Dependency Graph (Sequential Chain)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")


def create_performance_visualization(
    learning_progress_data: Dict[str, List[float]],
    random_data: Dict[str, List[float]],
    dependency_depths: Dict[str, int],
    ax: plt.Axes,
) -> None:
    """Create performance curves comparing learning progress vs random curriculum."""

    epochs = range(len(list(learning_progress_data.values())[0]))
    max_depth = max(dependency_depths.values())

    # Plot learning progress curriculum (solid lines)
    for task, performances in learning_progress_data.items():
        color = plt.cm.viridis(dependency_depths[task] / max_depth)
        ax.plot(
            epochs,
            performances,
            color=color,
            linewidth=2,
            linestyle="-",
            label=f"LP Task {task}" if task == "A" else "",
        )

    # Plot random curriculum (dotted lines)
    for task, performances in random_data.items():
        color = plt.cm.viridis(dependency_depths[task] / max_depth)
        ax.plot(
            epochs,
            performances,
            color=color,
            linewidth=2,
            linestyle=":",
            label=f"Random Task {task}" if task == "A" else "",
        )

    ax.set_title("Task Performance: Learning Progress vs Random Curriculum", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Performance Score")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add legend for line styles
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="gray", lw=2, linestyle="-"),
        Line2D([0], [0], color="gray", lw=2, linestyle=":"),
    ]
    ax.legend(custom_lines, ["Learning Progress", "Random"], bbox_to_anchor=(1.05, 0.5), loc="center left")


def create_probability_visualization(
    probability_data: Dict[str, List[float]], dependency_depths: Dict[str, int], ax: plt.Axes
) -> None:
    """Create filled graph visualization of task sampling probabilities."""

    epochs = range(len(list(probability_data.values())[0]))
    max_depth = max(dependency_depths.values())

    # Sort tasks by dependency depth for better visualization
    sorted_tasks = sorted(probability_data.keys(), key=lambda x: dependency_depths[x])

    # Create stacked area plot
    bottom = np.zeros(len(epochs))

    for task in sorted_tasks:
        probabilities = probability_data[task]
        color = plt.cm.viridis(dependency_depths[task] / max_depth)

        ax.fill_between(epochs, bottom, bottom + probabilities, color=color, alpha=0.7, label=f"Task {task}")
        bottom += probabilities

    ax.set_title("Task Sampling Probabilities Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Probability")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)


def create_efficiency_comparison(
    learning_progress_data: Dict[str, List[float]],
    random_data: Dict[str, List[float]],
    prioritize_regressed_data: Dict[str, List[float]],
    dependency_depths: Dict[str, int],
    ax: plt.Axes,
) -> None:
    """Create efficiency comparison showing all three curricula vs oracle."""

    epochs = range(len(list(learning_progress_data.values())[0]))

    # Calculate cumulative performance for learning progress curriculum
    lp_performances = []
    for epoch in epochs:
        epoch_performance = np.mean([learning_progress_data[task][epoch] for task in learning_progress_data.keys()])
        lp_performances.append(epoch_performance)

    # Calculate cumulative performance for random curriculum
    random_performances = []
    for epoch in epochs:
        epoch_performance = np.mean([random_data[task][epoch] for task in random_data.keys()])
        random_performances.append(epoch_performance)

    # Calculate cumulative performance for prioritize regressed curriculum
    pr_performances = []
    for epoch in epochs:
        epoch_performance = np.mean(
            [prioritize_regressed_data[task][epoch] for task in prioritize_regressed_data.keys()]
        )
        pr_performances.append(epoch_performance)

    # Calculate cumulative efficiency (area under curve)
    lp_cumulative_efficiency = np.cumsum(lp_performances)
    random_cumulative_efficiency = np.cumsum(random_performances)
    pr_cumulative_efficiency = np.cumsum(pr_performances)

    # Generate realistic oracle performance
    oracle_performances = generate_oracle_performance(epochs, dependency_depths)
    oracle_cumulative_efficiency = np.cumsum(oracle_performances)

    # Calculate efficiency ratios
    lp_efficiency_ratio = lp_cumulative_efficiency / oracle_cumulative_efficiency
    random_efficiency_ratio = random_cumulative_efficiency / oracle_cumulative_efficiency
    pr_efficiency_ratio = pr_cumulative_efficiency / oracle_cumulative_efficiency

    # Create line plot comparing all three curricula
    x_pos = np.arange(len(epochs))
    ax.plot(x_pos, lp_efficiency_ratio, color="blue", linewidth=2, label="Learning Progress", alpha=0.8)
    ax.plot(x_pos, random_efficiency_ratio, color="red", linewidth=2, label="Random", alpha=0.8)
    ax.plot(x_pos, pr_efficiency_ratio, color="orange", linewidth=2, label="Prioritize Regressed", alpha=0.8)

    # Add a horizontal line at 1.0 (oracle performance)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=2, label="Oracle Performance")

    ax.set_title("Efficiency Comparison: All Curricula vs Oracle", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Efficiency Ratio (Curriculum/Oracle)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels at key points
    key_epochs = [0, 50, 100, 150, 200, 249]
    for epoch in key_epochs:
        if epoch < len(lp_efficiency_ratio):
            lp_val = lp_efficiency_ratio[epoch]
            random_val = random_efficiency_ratio[epoch]
            pr_val = pr_efficiency_ratio[epoch]
            max_val = max(lp_val, random_val, pr_val)
            ax.annotate(
                f"LP: {lp_val:.2f}\nR: {random_val:.2f}\nPR: {pr_val:.2f}",
                xy=(epoch, max_val),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )


def generate_oracle_performance(epochs: range, dependency_depths: Dict[str, int]) -> List[float]:
    """Generate realistic oracle performance that follows optimal curriculum ordering."""

    oracle_performances = []

    for epoch in epochs:
        # Oracle follows optimal curriculum: learns tasks in dependency order
        # and achieves higher performance than actual curriculum

        if epoch < 10:
            # Early epochs: Oracle focuses on easy tasks (A, B, C)
            # Achieves 80-90% performance on easy tasks
            oracle_performance = 85 + 5 * np.random.random()
        elif epoch < 30:
            # Middle epochs: Oracle masters medium tasks (D, E, F)
            # Achieves 90-95% performance
            oracle_performance = 92 + 3 * np.random.random()
        elif epoch < 60:
            # Later epochs: Oracle tackles hard tasks (G, H, I)
            # Achieves 85-90% performance on harder tasks
            oracle_performance = 88 + 4 * np.random.random()
        else:
            # Final epochs: Oracle masters all tasks including hardest (J)
            # Achieves 90-95% performance overall
            oracle_performance = 93 + 2 * np.random.random()

        oracle_performances.append(oracle_performance)

    return oracle_performances


def run_advanced_curriculum_demo():
    """Run the advanced curriculum analysis demo."""
    logger.info("Starting Advanced Curriculum Analysis Demo...")

    # Generate data
    num_epochs = 250
    G, dependency_depths = create_dependency_graph()

    # Generate performance data for all three curricula
    learning_progress_data = generate_task_performance_data(num_epochs, "learning_progress")
    random_data = generate_task_performance_data(num_epochs, "random")
    prioritize_regressed_data = generate_task_performance_data(num_epochs, "prioritize_regressed")

    probability_data = generate_task_probabilities(num_epochs)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Dependency graph visualization
    ax1 = fig.add_subplot(gs[0, 0])
    create_dependency_visualization(G, dependency_depths, ax1)

    # 2. Performance curves comparison
    ax2 = fig.add_subplot(gs[0, 1:])
    create_performance_visualization(learning_progress_data, random_data, dependency_depths, ax2)

    # 3. Task sampling probabilities
    ax3 = fig.add_subplot(gs[1, 0:2])
    create_probability_visualization(probability_data, dependency_depths, ax3)

    # 4. Efficiency comparison
    ax4 = fig.add_subplot(gs[1, 2])
    create_efficiency_comparison(learning_progress_data, random_data, prioritize_regressed_data, dependency_depths, ax4)

    # Add overall title
    fig.suptitle("Advanced Curriculum Analysis: Learning Progress vs Random Curriculum", fontsize=16, fontweight="bold")

    # Save the plot
    plt.savefig("advanced_curriculum_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("ADVANCED CURRICULUM ANALYSIS SUMMARY")
    print("=" * 80)

    # Calculate summary statistics for all three curricula
    lp_final_performances = {task: performances[-1] for task, performances in learning_progress_data.items()}
    random_final_performances = {task: performances[-1] for task, performances in random_data.items()}
    pr_final_performances = {task: performances[-1] for task, performances in prioritize_regressed_data.items()}

    lp_avg_performance = np.mean(list(lp_final_performances.values()))
    random_avg_performance = np.mean(list(random_final_performances.values()))
    pr_avg_performance = np.mean(list(pr_final_performances.values()))

    # Calculate cumulative efficiency for all three curricula
    lp_cumulative_efficiency = sum([sum(performances) for performances in learning_progress_data.values()])
    random_cumulative_efficiency = sum([sum(performances) for performances in random_data.values()])
    pr_cumulative_efficiency = sum([sum(performances) for performances in prioritize_regressed_data.values()])

    print(f"Number of Tasks: {len(learning_progress_data)}")
    print(f"Number of Epochs: {num_epochs}")
    print("\nLearning Progress Curriculum:")
    print(f"  Average Final Performance: {lp_avg_performance:.2f}")
    print(f"  Total Cumulative Efficiency: {lp_cumulative_efficiency:.2f}")
    print("\nRandom Curriculum:")
    print(f"  Average Final Performance: {random_avg_performance:.2f}")
    print(f"  Total Cumulative Efficiency: {random_cumulative_efficiency:.2f}")
    print("\nPrioritize Regressed Curriculum:")
    print(f"  Average Final Performance: {pr_avg_performance:.2f}")
    print(f"  Total Cumulative Efficiency: {pr_cumulative_efficiency:.2f}")
    print("\nEfficiency Improvements:")
    print(
        f"  Learning Progress vs Random: {((lp_cumulative_efficiency / random_cumulative_efficiency) - 1) * 100:.1f}%"
    )
    lp_vs_pr_improvement = ((lp_cumulative_efficiency / pr_cumulative_efficiency) - 1) * 100
    pr_vs_random_improvement = ((pr_cumulative_efficiency / random_cumulative_efficiency) - 1) * 100
    print(f"  Learning Progress vs Prioritize Regressed: {lp_vs_pr_improvement:.1f}%")
    print(f"  Prioritize Regressed vs Random: {pr_vs_random_improvement:.1f}%")
    print("Results saved to: advanced_curriculum_analysis.png")

    # Print task-specific statistics
    print("\nTask Performance Summary (Learning Progress vs Random vs Prioritize Regressed):")
    for task in sorted(learning_progress_data.keys(), key=lambda x: dependency_depths[x]):
        lp_final_perf = lp_final_performances[task]
        random_final_perf = random_final_performances[task]
        pr_final_perf = pr_final_performances[task]
        depth = dependency_depths[task]
        lp_vs_random = ((lp_final_perf / random_final_perf) - 1) * 100
        lp_vs_pr = ((lp_final_perf / pr_final_perf) - 1) * 100
        pr_vs_random = ((pr_final_perf / random_final_perf) - 1) * 100
        task_summary = (
            f"  Task {task} (depth {depth}): LP={lp_final_perf:.1f}, "
            f"Random={random_final_perf:.1f}, PR={pr_final_perf:.1f} "
            f"(LP vs R: {lp_vs_random:+.1f}%, LP vs PR: {lp_vs_pr:+.1f}%, "
            f"PR vs R: {pr_vs_random:+.1f}%)"
        )
        print(task_summary)

    # Verify that probabilities sum to 1.0 at each epoch
    print("\nProbability Verification:")
    for epoch in [0, 50, 100, 150, 200, 249]:  # Check a few epochs
        total_prob = sum(probability_data[task][epoch] for task in probability_data.keys())
        print(f"  Epoch {epoch}: Total probability = {total_prob:.6f}")

    logger.info("Advanced Curriculum Analysis Demo completed!")


def main():
    """Main function to run the advanced curriculum analysis demo."""
    try:
        run_advanced_curriculum_demo()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

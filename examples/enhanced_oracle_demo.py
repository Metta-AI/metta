#!/usr/bin/env python3
"""
Enhanced Oracle Demo

This script demonstrates the enhanced topological oracle with realistic learning curves.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the enhanced oracle
from metta.rl.enhanced_oracle import (
    create_enhanced_oracle_from_demo_tasks,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def demonstrate_learning_curves():
    """Demonstrate the realistic learning curves for each task."""
    print("=" * 60)
    print("LEARNING CURVES DEMONSTRATION")
    print("=" * 60)

    # Create enhanced oracle
    enhanced_oracle = create_enhanced_oracle_from_demo_tasks()

    # Show learning curve parameters for each task
    print("\nLearning Curve Parameters:")
    print(f"{'Task':<4} {'Difficulty':<12} {'Max Perf':<10} {'Learning Rate':<15} {'Plateau':<10} {'Noise':<8}")
    print("-" * 70)

    for task_id in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        curve = enhanced_oracle.learning_curves[task_id]
        print(
            f"{task_id:<4} {curve.difficulty:<12.3f} {curve.max_performance:<10.3f} "
            f"{curve.learning_rate:<15.3f} {curve.plateau_threshold:<10.3f} {curve.noise_scale:<8.3f}"
        )

    # Demonstrate learning curves over time
    epochs = 100
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, task_id in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
        curve = enhanced_oracle.learning_curves[task_id]

        # Generate performance over time
        performances = []
        for epoch in range(epochs):
            perf = curve.predict_performance(epoch)
            performances.append(perf)

        # Plot learning curve
        ax = axes[i]
        ax.plot(range(epochs), performances, linewidth=2, alpha=0.8)
        ax.set_title(f"Task {task_id} (Difficulty: {curve.difficulty:.2f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("learning_curves_demo.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nLearning curves plot saved to: learning_curves_demo.png")


def demonstrate_optimal_ordering():
    """Demonstrate the optimal task ordering."""
    print("\n" + "=" * 60)
    print("OPTIMAL TASK ORDERING DEMONSTRATION")
    print("=" * 60)

    # Create enhanced oracle
    enhanced_oracle = create_enhanced_oracle_from_demo_tasks()

    print(f"\nOptimal Task Order: {enhanced_oracle.optimal_order}")

    # Show dependency levels
    levels = enhanced_oracle._get_dependency_levels()
    print("\nTasks by Dependency Level:")
    for level in sorted(levels.keys()):
        tasks = levels[level]
        efficiencies = [enhanced_oracle._calculate_learning_efficiency(t) for t in tasks]
        print(f"Level {level}: {tasks} (Efficiencies: {[f'{e:.3f}' for e in efficiencies]})")

    # Show task completion schedule
    schedule = enhanced_oracle.get_task_completion_schedule(50)
    print("\nTask Completion Schedule (first 50 epochs):")
    for task_id, completion_epochs in schedule.items():
        print(f"Task {task_id}: Completed at epoch {completion_epochs[0]}")


def demonstrate_oracle_performance():
    """Demonstrate the oracle performance over time."""
    print("\n" + "=" * 60)
    print("ORACLE PERFORMANCE DEMONSTRATION")
    print("=" * 60)

    # Create enhanced oracle
    enhanced_oracle = create_enhanced_oracle_from_demo_tasks()

    # Get oracle performance over 100 epochs
    epochs = 100
    oracle_performances = enhanced_oracle.get_optimal_curriculum_performance(epochs)

    # Plot oracle performance
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), oracle_performances, linewidth=3, color="green", alpha=0.8, label="Enhanced Oracle")
    plt.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Perfect Performance")

    plt.title("Enhanced Oracle Performance Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("oracle_performance_demo.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nOracle performance plot saved to: oracle_performance_demo.png")

    # Show performance statistics
    print("\nOracle Performance Statistics:")
    print(f"Initial Performance: {oracle_performances[0]:.2f}")
    print(f"Final Performance: {oracle_performances[-1]:.2f}")
    print(f"Average Performance: {np.mean(oracle_performances):.2f}")
    print(f"Performance Improvement: {oracle_performances[-1] - oracle_performances[0]:.2f}")


def demonstrate_enhanced_vs_simple_oracle():
    """Compare enhanced oracle with simple oracle."""
    print("\n" + "=" * 60)
    print("ENHANCED VS SIMPLE ORACLE COMPARISON")
    print("=" * 60)

    # Create enhanced oracle
    enhanced_oracle = create_enhanced_oracle_from_demo_tasks()

    # Get enhanced oracle performance
    epochs = 100
    enhanced_performances = enhanced_oracle.get_optimal_curriculum_performance(epochs)

    # Create simple oracle performance (like the old one)
    simple_performances = []
    for epoch in range(epochs):
        if epoch < 10:
            performance = 85 + 5 * np.random.random()
        elif epoch < 30:
            performance = 92 + 3 * np.random.random()
        elif epoch < 60:
            performance = 88 + 4 * np.random.random()
        else:
            performance = 93 + 2 * np.random.random()
        simple_performances.append(performance)

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), enhanced_performances, linewidth=3, color="green", alpha=0.8, label="Enhanced Oracle")
    plt.plot(range(epochs), simple_performances, linewidth=3, color="blue", alpha=0.8, label="Simple Oracle")
    plt.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Perfect Performance")

    plt.title("Enhanced vs Simple Oracle Performance", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("oracle_comparison_demo.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nOracle comparison plot saved to: oracle_comparison_demo.png")

    # Show comparison statistics
    print("\nPerformance Comparison:")
    print(f"{'Metric':<20} {'Enhanced':<12} {'Simple':<12} {'Difference':<12}")
    print("-" * 56)
    print(
        f"{'Initial Performance':<20} {enhanced_performances[0]:<12.2f} {simple_performances[0]:<12.2f} "
        f"{enhanced_performances[0] - simple_performances[0]:<12.2f}"
    )
    print(
        f"{'Final Performance':<20} {enhanced_performances[-1]:<12.2f} {simple_performances[-1]:<12.2f} "
        f"{enhanced_performances[-1] - simple_performances[-1]:<12.2f}"
    )
    print(
        f"{'Average Performance':<20} {np.mean(enhanced_performances):<12.2f} {np.mean(simple_performances):<12.2f} "
        f"{np.mean(enhanced_performances) - np.mean(simple_performances):<12.2f}"
    )


def main():
    """Run the enhanced oracle demonstration."""
    logger.info("Starting Enhanced Oracle Demo...")

    # Demonstrate learning curves
    demonstrate_learning_curves()

    # Demonstrate optimal ordering
    demonstrate_optimal_ordering()

    # Demonstrate oracle performance
    demonstrate_oracle_performance()

    # Compare enhanced vs simple oracle
    demonstrate_enhanced_vs_simple_oracle()

    print("\n" + "=" * 60)
    print("ENHANCED ORACLE DEMO COMPLETED")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("1. Realistic Learning Curves: Each task has difficulty-based parameters")
    print("2. Topological Ordering: Respects task dependencies")
    print("3. Learning Efficiency Optimization: Tasks ordered by efficiency within levels")
    print("4. Realistic Performance Prediction: Based on actual learning curves")
    print("5. Dependency-Aware Scheduling: Considers task dependencies in performance")


if __name__ == "__main__":
    main()

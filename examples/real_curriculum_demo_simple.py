#!/usr/bin/env python3
"""
Simple Real Curriculum Analysis Demo

This script demonstrates the curriculum analysis framework using
real curriculum implementations from the codebase.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def create_curriculum_configs():
    """Create configurations for real curricula from the codebase."""
    return {
        "learning_progress": {
            "path": "/env/mettagrid/curriculum/navigation/learning_progress",
            "description": "Learning Progress Curriculum",
        },
        "all": {
            "path": "/env/mettagrid/curriculum/all",
            "description": "All Tasks Curriculum",
        },
    }


@hydra.main(version_base=None, config_path="configs", config_name="trainer/test_curriculum_analysis")
def analyze_curriculum(cfg: DictConfig, curriculum_name: str, curriculum_path: str) -> Dict:
    """Analyze a single curriculum using the working integration test approach."""
    logger.info(f"Analyzing {curriculum_name} at {curriculum_path}")

    try:
        # Create trainer config - cfg has a trainer section
        trainer_cfg = cfg.trainer

        # Set analysis mode
        trainer_cfg.analysis_mode = True
        trainer_cfg.analysis_epochs = 3  # Small number for demo
        trainer_cfg.analysis_tasks_per_epoch = 2
        trainer_cfg.analysis_output_dir = f"real_curriculum_analysis/{curriculum_name}"

        # Import and run analysis
        from metta.mettagrid.curriculum.util import curriculum_from_config_path
        from metta.rl.curriculum_analysis import run_curriculum_analysis

        # Load curriculum
        curriculum = curriculum_from_config_path(curriculum_path, trainer_cfg.env_overrides)

        # Run analysis
        results = run_curriculum_analysis(trainer_cfg=trainer_cfg, curriculum=curriculum, oracle_curriculum=None)

        logger.info(f"Analysis completed for {curriculum_name}")
        return results

    except Exception as e:
        logger.error(f"Analysis failed for {curriculum_name}: {e}")
        return None


def generate_tables(all_results: Dict[str, Dict]):
    """Generate tables from real curriculum analysis results."""

    print("=" * 80)
    print("REAL CURRICULUM ANALYSIS RESULTS")
    print("=" * 80)

    # Create comparison dataframe
    comparison_data = []

    for curriculum_name, results in all_results.items():
        if results is None:
            continue

        summary = results.get("summary", {})
        comparison_data.append(
            {
                "curriculum": curriculum_name,
                "efficiency": summary.get("average_efficiency", 0),
                "time_to_threshold": summary.get("average_time_to_threshold", 0),
                "adaptation_speed": summary.get("average_adaptation_speed", 0),
                "weight_stability": summary.get("average_weight_stability", 0),
                "total_tasks": summary.get("total_tasks", 0),
                "total_epochs": summary.get("total_epochs", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    if comparison_df.empty:
        print("No successful analysis results to display.")
        return comparison_df

    # 1. Overall Performance Summary
    print("\n1. OVERALL PERFORMANCE SUMMARY")
    print("-" * 50)

    summary_stats = comparison_df.agg(
        {
            "efficiency": ["mean", "std"],
            "time_to_threshold": ["mean", "std"],
            "adaptation_speed": ["mean", "std"],
            "weight_stability": ["mean", "std"],
        }
    ).round(2)

    print(summary_stats.to_string())

    # 2. Curriculum-by-Curriculum Analysis
    print("\n2. CURRICULUM-BY-CURRICULUM ANALYSIS")
    print("-" * 50)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['curriculum']}:")
        print(f"  Efficiency: {row['efficiency']:.2f}")
        print(f"  Time to Threshold: {row['time_to_threshold']:.1f}")
        print(f"  Adaptation Speed: {row['adaptation_speed']:.3f}")
        print(f"  Weight Stability: {row['weight_stability']:.3f}")
        print(f"  Total Tasks: {row['total_tasks']}")
        print(f"  Total Epochs: {row['total_epochs']}")

    # 3. Best and Worst Performers
    print("\n3. BEST AND WORST PERFORMERS")
    print("-" * 50)

    if len(comparison_df) > 0:
        best_efficiency = comparison_df.loc[comparison_df["efficiency"].idxmax()]
        worst_efficiency = comparison_df.loc[comparison_df["efficiency"].idxmin()]

        print("\nBest Efficiency:")
        print(f"  {best_efficiency['curriculum']}: {best_efficiency['efficiency']:.2f}")

        print("\nWorst Efficiency:")
        print(f"  {worst_efficiency['curriculum']}: {worst_efficiency['efficiency']:.2f}")

        best_time = comparison_df.loc[comparison_df["time_to_threshold"].idxmin()]
        worst_time = comparison_df.loc[comparison_df["time_to_threshold"].idxmax()]

        print("\nBest Time Performance:")
        print(f"  {best_time['curriculum']}: {best_time['time_to_threshold']:.1f}")

        print("\nWorst Time Performance:")
        print(f"  {worst_time['curriculum']}: {worst_time['time_to_threshold']:.1f}")

    return comparison_df


def generate_plots(comparison_df: pd.DataFrame, all_results: Dict[str, Dict]):
    """Generate plots from real curriculum analysis results."""

    if comparison_df.empty:
        logger.warning("No data available for plotting")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Real Curriculum Analysis Results", fontsize=16, fontweight="bold")

    # 1. Efficiency Comparison
    ax1 = axes[0, 0]
    curricula = comparison_df["curriculum"]
    efficiencies = comparison_df["efficiency"]
    colors = [plt.cm.Set3(i) for i in range(len(curricula))]

    bars1 = ax1.bar(curricula, efficiencies, color=colors, alpha=0.7)
    ax1.set_title("Curriculum Efficiency Comparison")
    ax1.set_ylabel("Efficiency Score")
    ax1.set_xlabel("Curriculum Type")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, efficiency in zip(bars1, efficiencies, strict=False):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f"{efficiency:.1f}", ha="center", va="bottom")

    # 2. Time to Threshold Comparison
    ax2 = axes[0, 1]
    times = comparison_df["time_to_threshold"]

    bars2 = ax2.bar(curricula, times, color=colors, alpha=0.7)
    ax2.set_title("Time to Threshold Comparison")
    ax2.set_ylabel("Time to Threshold")
    ax2.set_xlabel("Curriculum Type")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, time in zip(bars2, times, strict=False):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{time:.1f}", ha="center", va="bottom")

    # 3. Adaptation Metrics Scatter Plot
    ax3 = axes[1, 0]
    adaptation_speeds = comparison_df["adaptation_speed"]
    weight_stabilities = comparison_df["weight_stability"]

    ax3.scatter(adaptation_speeds, weight_stabilities, c=range(len(curricula)), cmap="viridis", s=100, alpha=0.7)
    ax3.set_title("Adaptation vs Weight Stability")
    ax3.set_xlabel("Adaptation Speed")
    ax3.set_ylabel("Weight Stability")

    # Add curriculum labels
    for i, curriculum in enumerate(curricula):
        ax3.annotate(
            curriculum,
            (adaptation_speeds.iloc[i], weight_stabilities.iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # 4. Task Completion Heatmap
    ax4 = axes[1, 1]

    # Create heatmap data from task completion histories
    heatmap_data = []
    for _curriculum_name, results in all_results.items():
        if results is None or "task_history" not in results:
            continue

        task_history = results["task_history"]
        if not task_history:
            continue

        # Count tasks per epoch
        epoch_counts = {}
        for task in task_history:
            epoch = task.get("epoch", 0)
            epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1

        # Create row for heatmap
        row = [epoch_counts.get(epoch, 0) for epoch in range(3)]  # 3 epochs
        heatmap_data.append(row)

    if heatmap_data:
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[name for name, results in all_results.items() if results is not None],
            columns=[f"Epoch {i}" for i in range(3)],
        )

        sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd", ax=ax4)
        ax4.set_title("Task Completion Heatmap")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Curriculum")
    else:
        ax4.text(0.5, 0.5, "No task completion data available", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Task Completion Heatmap")

    plt.tight_layout()
    plt.savefig("real_curriculum_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info("Plots saved to real_curriculum_analysis.png")


def main():
    """Main function to run real curriculum analysis demo."""
    logger.info("Starting Real Curriculum Analysis Demo...")

    # Get curriculum configurations
    curriculum_configs = create_curriculum_configs()

    # Run analysis on each curriculum
    all_results = {}

    for curriculum_name, config in curriculum_configs.items():
        logger.info(f"Processing {curriculum_name}...")

        try:
            # Use the working integration test approach
            results = analyze_curriculum(
                cfg=None,  # Will be provided by Hydra
                curriculum_name=curriculum_name,
                curriculum_path=config["path"],
            )
            all_results[curriculum_name] = results

        except Exception as e:
            logger.error(f"Failed to analyze {curriculum_name}: {e}")
            all_results[curriculum_name] = None

    # Generate tables
    comparison_df = generate_tables(all_results)

    # Generate plots
    generate_plots(comparison_df, all_results)

    logger.info("Real Curriculum Analysis Demo completed!")

    # Print summary
    successful_analyses = sum(1 for results in all_results.values() if results is not None)
    total_curricula = len(curriculum_configs)

    print(f"\n{'=' * 50}")
    print("ANALYSIS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Successfully analyzed: {successful_analyses}/{total_curricula} curricula")
    print("Results saved to: real_curriculum_analysis.png")

    if successful_analyses > 0 and not comparison_df.empty:
        print(f"Best performing curriculum: {comparison_df.loc[comparison_df['efficiency'].idxmax(), 'curriculum']}")
        print(f"Most stable curriculum: {comparison_df.loc[comparison_df['weight_stability'].idxmax(), 'curriculum']}")


if __name__ == "__main__":
    main()

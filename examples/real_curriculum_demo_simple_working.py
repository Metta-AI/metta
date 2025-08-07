#!/usr/bin/env python3
"""
Simple Working Real Curriculum Analysis Demo

This script demonstrates the curriculum analysis framework using
real curriculum implementations from the codebase.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def run_single_curriculum_analysis():
    """Run analysis on a single curriculum using the working integration test."""
    logger.info("Running curriculum analysis...")

    try:
        # Import the working integration test
        import subprocess
        import sys

        # Run the integration test
        result = subprocess.run(
            [sys.executable, "test_curriculum_analysis_integration.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode == 0:
            logger.info("Curriculum analysis completed successfully!")
            return True
        else:
            logger.error(f"Curriculum analysis failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Failed to run curriculum analysis: {e}")
        return False


def generate_plots_from_results():
    """Generate plots from the analysis results."""
    logger.info("Generating plots from analysis results...")

    # Read the results from the integration test
    results_dir = Path("integration_test")

    if not results_dir.exists():
        logger.error("Results directory not found")
        return

    # Read curriculum metrics
    metrics_file = results_dir / "curriculum_metrics.csv"
    if not metrics_file.exists():
        logger.error("Metrics file not found")
        return

    try:
        metrics_df = pd.read_csv(metrics_file)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Real Curriculum Analysis Results", fontsize=16, fontweight="bold")

        # 1. Performance over epochs
        ax1 = axes[0, 0]
        epochs = range(len(metrics_df))
        performances = metrics_df["performance"].values

        ax1.plot(epochs, performances, "o-", linewidth=2, markersize=8)
        ax1.set_title("Performance Over Epochs")
        ax1.set_ylabel("Performance Score")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, perf in enumerate(performances):
            ax1.annotate(f"{perf:.1f}", (i, perf), textcoords="offset points", xytext=(0, 10), ha="center")

        # 2. Time to threshold over epochs
        ax2 = axes[0, 1]
        times = metrics_df["time_to_threshold"].values

        ax2.plot(epochs, times, "s-", linewidth=2, markersize=8, color="orange")
        ax2.set_title("Time to Threshold Over Epochs")
        ax2.set_ylabel("Time to Threshold")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, time in enumerate(times):
            ax2.annotate(f"{time:.1f}", (i, time), textcoords="offset points", xytext=(0, 10), ha="center")

        # 3. Efficiency over epochs
        ax3 = axes[1, 0]
        efficiencies = metrics_df["efficiency"].values

        ax3.plot(epochs, efficiencies, "^-", linewidth=2, markersize=8, color="red")
        ax3.set_title("Efficiency Over Epochs")
        ax3.set_ylabel("Efficiency Score")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for i, eff in enumerate(efficiencies):
            ax3.annotate(f"{eff:.1f}", (i, eff), textcoords="offset points", xytext=(0, 10), ha="center")

        # 4. Summary statistics
        ax4 = axes[1, 1]

        # Create summary bar chart
        variances = metrics_df["final_perf_variance"].values
        metrics = ["Performance", "Efficiency", "Time to Threshold", "Performance Variance"]
        avg_values = [np.mean(performances), np.mean(efficiencies), np.mean(times), np.mean(variances)]

        bars = ax4.bar(metrics, avg_values, color=["blue", "red", "orange", "green"], alpha=0.7)
        ax4.set_title("Average Metrics")
        ax4.set_ylabel("Average Value")
        ax4.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, avg_values, strict=False):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f"{value:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("real_curriculum_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("Plots saved to real_curriculum_analysis.png")

        # Print summary
        print("\n" + "=" * 60)
        print("REAL CURRICULUM ANALYSIS SUMMARY")
        print("=" * 60)
        print("Curriculum: Learning Progress")
        print(f"Total Epochs: {len(metrics_df)}")
        print(f"Average Performance: {np.mean(performances):.2f}")
        print(f"Average Efficiency: {np.mean(efficiencies):.2f}")
        print(f"Average Time to Threshold: {np.mean(times):.2f}")
        print(f"Average Performance Variance: {np.mean(variances):.3f}")
        print("Results saved to: real_curriculum_analysis.png")

    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")


def main():
    """Main function to run real curriculum analysis demo."""
    logger.info("Starting Real Curriculum Analysis Demo...")

    # Run the curriculum analysis
    success = run_single_curriculum_analysis()

    if success:
        # Generate plots from results
        generate_plots_from_results()

        logger.info("Real Curriculum Analysis Demo completed!")
    else:
        logger.error("Demo failed due to analysis errors")


if __name__ == "__main__":
    main()

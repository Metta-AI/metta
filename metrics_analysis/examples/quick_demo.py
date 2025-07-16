#!/usr/bin/env -S uv run
"""
Quick demo of the Metta Metrics Analysis tools.

This script demonstrates the analysis capabilities with minimal data fetching.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from metta_metrics_analysis.data_processor import DataProcessor
from metta_metrics_analysis.statistical_analysis import StatisticalAnalyzer
from metta_metrics_analysis.wandb_data_collector import WandBDataCollector


def main():
    """Run a quick analysis demo."""

    # 1. Initialize the data collector
    print("=== Quick Demo: Metta Metrics Analysis ===")
    collector = WandBDataCollector(
        entity="metta-research",
        project="metta",
        use_cache=True,
    )

    # 2. Fetch just a couple of runs with minimal data
    print("\n=== Fetching Minimal Data ===")
    print("Fetching only 2 runs with last 50 steps each...")

    data = collector.fetch_runs(
        run_filter={
            "display_name": {"$regex": "jacke"},
            "state": "finished",
            "$and": [{"summary_metrics.losses/loss": {"$exists": True}}],  # Only runs with loss data
        },
        metrics=["losses/loss"],  # Just one metric for speed
        last_n_steps=50,  # Very few steps
        max_runs=2,  # Just 2 runs
    )

    print(f"Fetched {len(data)} data points")

    if len(data) == 0:
        print("\nNo data found. Trying a simpler query...")
        # Try without the metric filter
        data = collector.fetch_runs(
            run_filter={"display_name": {"$regex": "jacke"}, "state": "finished"},
            metrics=["losses/loss", "losses/entropy", "overview/navigation_score"],
            last_n_steps=50,
            max_runs=2,
        )
        print(f"Fetched {len(data)} data points")

    if len(data) == 0:
        print("Still no data. Please check your runs have the requested metrics.")
        return

    # 3. Process the data
    print("\n=== Processing Data ===")
    processor = DataProcessor(data)

    # Show what we got
    print(f"Runs found: {processor.df['run_name'].unique()}")
    print(f"Metrics found: {processor.metric_columns}")

    # Aggregate by run
    if processor.metric_columns:
        run_summary = processor.aggregate_by_run(
            metrics=processor.metric_columns[:1],  # Just first metric
        )

        print(f"\nAggregated to {len(run_summary)} runs")
        print("\nRun Summary:")
        print(run_summary[["run_name"] + [col for col in run_summary.columns if "_mean" in col]].to_string())

        # 4. Quick Statistical Analysis
        print("\n=== Statistical Analysis ===")
        analyzer = StatisticalAnalyzer(run_summary)

        # Just compute basic IQM without bootstrap for speed
        metric_col = [col for col in run_summary.columns if "_mean" in col][0]
        iqm_value = analyzer.compute_iqm(metric=metric_col)
        print(f"\nIQM for {metric_col}: {iqm_value:.4f}")

        # If we have 2 runs, show the difference
        if len(run_summary) >= 2:
            values = run_summary[metric_col].values
            print(f"Run 1: {values[0]:.4f}")
            print(f"Run 2: {values[1]:.4f}")
            print(f"Difference: {abs(values[0] - values[1]):.4f}")

    print("\n=== Demo Complete ===")
    print("\nThis was a minimal demo. For full analysis:")
    print("1. Increase last_n_steps (e.g., 1000)")
    print("2. Increase max_runs (e.g., 10)")
    print("3. Add more metrics")
    print("4. Enable bootstrap confidence intervals")


if __name__ == "__main__":
    main()

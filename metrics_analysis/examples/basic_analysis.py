#!/usr/bin/env -S uv run
"""
Basic example of using the Metta Metrics Analysis tools.

This script demonstrates:
1. Fetching runs from WandB
2. Processing and cleaning data
3. Computing statistical analyses
4. Generating visualizations
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from metta_metrics_analysis.data_processor import DataProcessor
from metta_metrics_analysis.statistical_analysis import StatisticalAnalyzer
from metta_metrics_analysis.wandb_data_collector import WandBDataCollector


def main():
    """Run a basic analysis example."""

    # 1. Initialize the data collector
    print("=== Initializing WandB Data Collector ===")
    collector = WandBDataCollector(
        entity="metta-research",
        project="metta",
        use_cache=True,  # Cache results for faster re-runs
    )

    # 2. Fetch some example runs
    print("\n=== Fetching Runs ===")
    # Fetch jacke's runs with interesting metrics
    data = collector.fetch_runs(
        run_filter={"display_name": {"$regex": "jacke"}, "state": "finished"},
        metrics=[
            "losses/loss",
            "losses/entropy",
            "losses/value_loss",
            "losses/policy_loss",
            "eval_navigation/mean",
            "overview/navigation_score",
        ],
        config_params=["trainer.algorithm", "trainer.learning_rate"],
        last_n_steps=1000,  # Get last 1000 steps
        max_runs=5,  # Analyze top 5 finished runs
    )

    print(f"Fetched {len(data)} data points from runs")

    # 3. Process the data
    print("\n=== Processing Data ===")
    processor = DataProcessor(data)

    # Filter to complete runs only
    processor = processor.filter_complete_runs(min_steps=500)

    # Check if we have data
    if len(processor.df) == 0:
        print("No data to process. Please adjust the run_filter to match your runs.")
        print("\nTips:")
        print("- Use run_filter=None to get recent runs")
        print("- Use run_filter={'group': 'your_group_name'} for a specific sweep")
        print("- Check that the metrics you're requesting exist in your runs")
        return

    # Aggregate by run to get summary statistics
    # Get available metrics
    available_metrics = [
        col
        for col in processor.df.columns
        if col not in ["run_id", "run_name", "group", "tags", "state", "created_at", "step"]
        and not col.startswith("config.")
    ]

    print(f"\nAvailable metrics for aggregation: {available_metrics}")

    # Use whatever metrics are available
    if "losses/loss" in available_metrics:
        run_summary = processor.aggregate_by_run(
            metrics=["losses/loss", "losses/entropy"] if "losses/entropy" in available_metrics else ["losses/loss"],
            aggregations={
                "losses/loss": ["mean", "min", "last"],
                "losses/entropy": ["mean", "max", "last"] if "losses/entropy" in available_metrics else None,
            },
        )
    elif "eval_navigation/mean" in available_metrics:
        run_summary = processor.aggregate_by_run(
            metrics=["eval_navigation/mean"],
            aggregations={
                "eval_navigation/mean": ["mean", "max", "last"],
            },
        )
    else:
        # Use first available metric
        first_metric = available_metrics[0] if available_metrics else None
        if first_metric:
            run_summary = processor.aggregate_by_run(
                metrics=[first_metric],
                aggregations={
                    first_metric: ["mean", "std", "last"],
                },
            )
        else:
            print("No metrics available for aggregation!")
            return

    print(f"Aggregated to {len(run_summary)} runs")
    print("\nRun Summary:")
    print(run_summary[["run_name", "eval_navigation/success_rate_mean", "trainer/loss_mean"]].head())

    # 4. Statistical Analysis
    print("\n=== Statistical Analysis ===")

    # Use aggregated data for analysis
    analyzer = StatisticalAnalyzer(run_summary)

    # Compute IQM with confidence intervals
    print("\n--- IQM Analysis ---")

    # Find a metric to analyze
    metric_cols = [col for col in run_summary.columns if "_mean" in col]
    if not metric_cols:
        print("No aggregated metrics found for IQM analysis")
        return

    analysis_metric = metric_cols[0]  # Use first available metric
    print(f"Analyzing metric: {analysis_metric}")

    if "config.trainer.algorithm" in run_summary.columns and run_summary["config.trainer.algorithm"].nunique() > 1:
        iqm_results = analyzer.compute_iqm_with_ci(
            metric=analysis_metric,
            group_by="config.trainer.algorithm",
            confidence_level=0.95,
            n_bootstrap=1000,  # Reduced for example speed
        )
    else:
        # If no algorithm config or all same, group by run_name for demo
        iqm_results = analyzer.compute_iqm_with_ci(
            metric=analysis_metric,
            group_by="run_name",
            confidence_level=0.95,
            n_bootstrap=1000,
        )

    print("\nIQM Results by Algorithm:")
    print(iqm_results)

    # 5. Save results
    print("\n=== Saving Results ===")
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Export processed data
    processor.export_to_csv(output_dir / "processed_runs.csv")
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    iqm_results.to_csv(output_dir / "iqm_results.csv", index=False)

    print(f"\nResults saved to {output_dir}/")

    # 6. Print summary statistics
    print("\n=== Summary Statistics ===")
    summary_stats = processor.get_summary_stats()
    print("\nMetric Statistics:")
    print(summary_stats[["metric", "mean", "std", "missing_pct"]].head(10))

    # Example of performance profiles (if we have multiple algorithms)
    if "config.trainer.algorithm" in run_summary.columns:
        unique_algorithms = run_summary["config.trainer.algorithm"].nunique()
        if unique_algorithms > 1:
            print(f"\nFound {unique_algorithms} different algorithms for comparison")

            # Compute performance profiles
            profiles = analyzer.compute_performance_profiles(
                metric=analysis_metric,
                group_by="config.trainer.algorithm",
                task_column="run_id",  # Use run_id as task since we may not have task config
                higher_is_better="loss" not in analysis_metric,  # Lower is better for loss metrics
            )

            print("\nPerformance Profile Sample (first 5 thresholds):")
            print(profiles.head())


if __name__ == "__main__":
    main()

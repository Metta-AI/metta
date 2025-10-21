#!/usr/bin/env python3
"""
Plot comparison of contrastive vs non-contrastive training runs.

Fetches metrics from WandB, computes means across multiple seeds, and plots them.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy import stats

from experiments.notebooks.utils.metrics import fetch_metrics


def compute_mean_across_runs(
    metrics_dfs: dict[str, pd.DataFrame], metric_key: str = "overview/reward"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean and std across multiple runs at each step.

    Args:
        metrics_dfs: Dictionary mapping run names to DataFrames
        metric_key: The metric column to average

    Returns:
        Tuple of (stats_df, individual_runs_df) where:
        - stats_df has columns: _step, mean, std, min, max, count
        - individual_runs_df has columns: _step, value, run
    """
    # Collect all dataframes
    dfs = []
    for run_name, df in metrics_dfs.items():
        if metric_key in df.columns and "_step" in df.columns:
            temp_df = df[["_step", metric_key]].copy()
            temp_df.columns = ["_step", "value"]
            temp_df["run"] = run_name
            dfs.append(temp_df)

    if not dfs:
        return pd.DataFrame(), pd.DataFrame()

    # Combine all runs
    combined = pd.concat(dfs, ignore_index=True)

    # Group by step and compute statistics
    stats = (
        combined.groupby("_step")["value"]
        .agg(mean="mean", std="std", min="min", max="max", count="count")
        .reset_index()
    )

    return stats, combined


def compute_ttest(contrastive_individual: pd.DataFrame, no_contrastive_individual: pd.DataFrame) -> tuple[float, float]:
    """
    Compute independent samples t-test between two conditions.

    Compares the maximum value achieved by each seed (one value per seed).

    Args:
        contrastive_individual: Individual run data for contrastive runs
        no_contrastive_individual: Individual run data for non-contrastive runs

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if contrastive_individual.empty or no_contrastive_individual.empty:
        return float("nan"), float("nan")

    # Get maximum value for each run (seed) - one value per seed
    contrastive_max_per_seed = contrastive_individual.groupby("run")["value"].max().dropna().values
    no_contrastive_max_per_seed = no_contrastive_individual.groupby("run")["value"].max().dropna().values

    if len(contrastive_max_per_seed) == 0 or len(no_contrastive_max_per_seed) == 0:
        return float("nan"), float("nan")

    # Perform independent samples t-test on per-seed maximums
    t_stat, p_value = stats.ttest_ind(contrastive_max_per_seed, no_contrastive_max_per_seed)

    return t_stat, p_value


def plot_comparison(
    contrastive_stats: pd.DataFrame,
    no_contrastive_stats: pd.DataFrame,
    contrastive_individual: pd.DataFrame,
    no_contrastive_individual: pd.DataFrame,
    metric_key: str = "overview/reward",
    save_path: str | None = None,
) -> None:
    """
    Plot comparison of two conditions with mean, confidence intervals, and individual runs.

    Args:
        contrastive_stats: Statistics for contrastive runs
        no_contrastive_stats: Statistics for non-contrastive runs
        contrastive_individual: Individual run data for contrastive runs
        no_contrastive_individual: Individual run data for non-contrastive runs
        metric_key: Name of the metric being plotted
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract readable metric name from key (e.g., "env_agent/heart.get" -> "Heart Get")
    metric_name = metric_key.split("/")[-1].replace("_", " ").replace(".", " ").title()

    # Plot individual contrastive runs (semi-transparent)
    if not contrastive_individual.empty:
        for run_name in contrastive_individual["run"].unique():
            run_data = contrastive_individual[contrastive_individual["run"] == run_name]
            ax.plot(
                run_data["_step"],
                run_data["value"],
                color="#2E86AB",
                alpha=0.3,
                linewidth=1,
            )

    # Plot contrastive mean with error bars
    if not contrastive_stats.empty:
        ax.errorbar(
            contrastive_stats["_step"],
            contrastive_stats["mean"],
            yerr=contrastive_stats["std"],
            label=f"With Contrastive (n={int(contrastive_stats['count'].iloc[0])})",
            color="#2E86AB",
            linewidth=2.5,
            errorevery=max(1, len(contrastive_stats) // 20),  # Show error bars at ~20 points
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.9,
        )
        ax.fill_between(
            contrastive_stats["_step"],
            contrastive_stats["mean"] - contrastive_stats["std"],
            contrastive_stats["mean"] + contrastive_stats["std"],
            alpha=0.2,
            color="#2E86AB",
        )

    # Plot individual non-contrastive runs (semi-transparent)
    if not no_contrastive_individual.empty:
        for run_name in no_contrastive_individual["run"].unique():
            run_data = no_contrastive_individual[no_contrastive_individual["run"] == run_name]
            ax.plot(
                run_data["_step"],
                run_data["value"],
                color="#A23B72",
                alpha=0.3,
                linewidth=1,
            )

    # Plot non-contrastive mean with error bars
    if not no_contrastive_stats.empty:
        ax.errorbar(
            no_contrastive_stats["_step"],
            no_contrastive_stats["mean"],
            yerr=no_contrastive_stats["std"],
            label=f"Without Contrastive (n={int(no_contrastive_stats['count'].iloc[0])})",
            color="#A23B72",
            linewidth=2.5,
            errorevery=max(1, len(no_contrastive_stats) // 20),  # Show error bars at ~20 points
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.9,
        )
        ax.fill_between(
            no_contrastive_stats["_step"],
            no_contrastive_stats["mean"] - no_contrastive_stats["std"],
            no_contrastive_stats["mean"] + no_contrastive_stats["std"],
            alpha=0.2,
            color="#A23B72",
        )

    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title(
        f"Contrastive Loss Impact on {metric_name} (with PPO hypers, ViT, ABES)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=12, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.ticklabel_format(style="plain", axis="x")

    # Set y-axis to increment by 1
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add statistics text box in bottom right
    if not contrastive_stats.empty and not no_contrastive_stats.empty:
        # Find maximum mean value across all timesteps where BOTH mean and std are valid
        contrastive_valid = contrastive_stats.dropna(subset=["mean", "std"])
        no_contrastive_valid = no_contrastive_stats.dropna(subset=["mean", "std"])

        if not contrastive_valid.empty and not no_contrastive_valid.empty:
            # Find the absolute maximum for each condition independently
            contrastive_max_idx = contrastive_valid["mean"].idxmax()
            contrastive_max_mean = contrastive_valid.loc[contrastive_max_idx, "mean"]
            contrastive_max_std = contrastive_valid.loc[contrastive_max_idx, "std"]
            contrastive_max_step = contrastive_valid.loc[contrastive_max_idx, "_step"]

            no_contrastive_max_idx = no_contrastive_valid["mean"].idxmax()
            no_contrastive_max_mean = no_contrastive_valid.loc[no_contrastive_max_idx, "mean"]
            no_contrastive_max_std = no_contrastive_valid.loc[no_contrastive_max_idx, "std"]
            no_contrastive_max_step = no_contrastive_valid.loc[no_contrastive_max_idx, "_step"]

            mean_difference = abs(contrastive_max_mean - no_contrastive_max_mean)

            # Compute t-test
            t_stat, p_value = compute_ttest(contrastive_individual, no_contrastive_individual)

            # Debug print to verify we're finding maximums correctly
            print(f"  DEBUG: With contrastive max {contrastive_max_mean:.4f} at step {contrastive_max_step}")
            print(f"  DEBUG: Without contrastive max {no_contrastive_max_mean:.4f} at step {no_contrastive_max_step}")
            print(f"  DEBUG: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")

            # Format p-value with significance indicator
            if p_value < 0.001:
                p_str = f"{p_value:.4e} ***"
            elif p_value < 0.01:
                p_str = f"{p_value:.4f} **"
            elif p_value < 0.05:
                p_str = f"{p_value:.4f} *"
            else:
                p_str = f"{p_value:.4f}"

            stats_text = (
                f"Maximum Statistics:\n"
                f"With Contrastive: {contrastive_max_mean:.4f} ± {contrastive_max_std:.4f}\n"
                f"Without Contrastive: {no_contrastive_max_mean:.4f} ± {no_contrastive_max_std:.4f}\n"
                f"Difference: {mean_difference:.4f}\n"
                f"\nStudent's t-test:\n"
                f"t = {t_stat:.4f}, p = {p_str}"
            )

            ax.text(
                0.98,
                0.02,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
            )

    # Format x-axis to show steps in millions if large
    max_step = max(
        contrastive_stats["_step"].max() if not contrastive_stats.empty else 0,
        no_contrastive_stats["_step"].max() if not no_contrastive_stats.empty else 0,
    )
    if max_step > 1_000_000:
        ax.ticklabel_format(style="plain", axis="x")
        # Add custom formatter for millions
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e6:.1f}M"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def main():
    # Configuration
    BASE_RUN_NAME = "tasha.10.17.shaped_PPOhypers_vit"

    # Select which seeds to use (all 5 seeds)
    SEEDS_TO_USE = [67, 134, 201, 34, 672]

    # Which metrics to plot
    METRICS_TO_PLOT = [
        "env_agent/heart.get",
    ]

    # Number of samples to fetch from WandB (higher = more accurate but slower)
    NUM_SAMPLES = 5000

    # Maximum timestep to include in the plot (None for all timesteps)
    MAX_TIMESTEP = 2_000_000_000  # 2 billion

    print("=" * 80)
    print("Contrastive Loss Comparison")
    print("=" * 80)
    print(f"\nBase run name: {BASE_RUN_NAME}")
    print(f"Seeds: {SEEDS_TO_USE}")
    print(f"Metrics: {METRICS_TO_PLOT}")
    print(f"\n{'=' * 80}\n")

    # Build run names
    contrastive_runs = [f"{BASE_RUN_NAME}.contrastive.seed{seed}" for seed in SEEDS_TO_USE]
    no_contrastive_runs = [f"{BASE_RUN_NAME}.no_contrastive.seed{seed}" for seed in SEEDS_TO_USE]

    print(f"Contrastive runs ({len(contrastive_runs)}):")
    for run in contrastive_runs:
        print(f"  - {run}")

    print(f"\nNon-contrastive runs ({len(no_contrastive_runs)}):")
    for run in no_contrastive_runs:
        print(f"  - {run}")

    print(f"\n{'=' * 80}\n")

    # Fetch metrics
    print("Fetching metrics from WandB...")
    print("\nFetching contrastive runs...")
    contrastive_metrics = fetch_metrics(contrastive_runs, samples=NUM_SAMPLES, keys=METRICS_TO_PLOT + ["_step"])

    print("\nFetching non-contrastive runs...")
    no_contrastive_metrics = fetch_metrics(no_contrastive_runs, samples=NUM_SAMPLES, keys=METRICS_TO_PLOT + ["_step"])

    # Filter to maximum timestep if specified
    if MAX_TIMESTEP is not None:
        print(f"\nFiltering data to max {MAX_TIMESTEP:,} timesteps...")
        for run_name in contrastive_metrics:
            if "_step" in contrastive_metrics[run_name].columns:
                contrastive_metrics[run_name] = contrastive_metrics[run_name][
                    contrastive_metrics[run_name]["_step"] <= MAX_TIMESTEP
                ]
        for run_name in no_contrastive_metrics:
            if "_step" in no_contrastive_metrics[run_name].columns:
                no_contrastive_metrics[run_name] = no_contrastive_metrics[run_name][
                    no_contrastive_metrics[run_name]["_step"] <= MAX_TIMESTEP
                ]

    print(f"\n{'=' * 80}\n")

    # Plot each metric
    for metric in METRICS_TO_PLOT:
        print(f"\nProcessing metric: {metric}")

        # Compute statistics and get individual runs
        contrastive_stats, contrastive_individual = compute_mean_across_runs(contrastive_metrics, metric)
        no_contrastive_stats, no_contrastive_individual = compute_mean_across_runs(no_contrastive_metrics, metric)

        if contrastive_stats.empty and no_contrastive_stats.empty:
            print(f"  ⚠️  No data available for {metric}, skipping...")
            continue

        # Print summary statistics
        if not contrastive_stats.empty:
            valid_stats = contrastive_stats.dropna(subset=["mean", "std"])
            if not valid_stats.empty:
                max_idx = valid_stats["mean"].idxmax()
                max_mean = valid_stats.loc[max_idx, "mean"]
                max_std = valid_stats.loc[max_idx, "std"]
                print(f"  With contrastive - Max: {max_mean:.4f} ± {max_std:.4f}")

        if not no_contrastive_stats.empty:
            valid_stats = no_contrastive_stats.dropna(subset=["mean", "std"])
            if not valid_stats.empty:
                max_idx = valid_stats["mean"].idxmax()
                max_mean = valid_stats.loc[max_idx, "mean"]
                max_std = valid_stats.loc[max_idx, "std"]
                print(f"  Without contrastive - Max: {max_mean:.4f} ± {max_std:.4f}")

        # Create save path
        metric_filename = metric.replace("/", "_")
        save_path = f"contrastive_comparison_{metric_filename}.png"

        # Plot
        plot_comparison(
            contrastive_stats,
            no_contrastive_stats,
            contrastive_individual,
            no_contrastive_individual,
            metric_key=metric,
            save_path=save_path,
        )

    print(f"\n{'=' * 80}")
    print("Done! All plots generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot the difference curve (With Contrastive - Without Contrastive) over time.

Shows how the performance gap between conditions evolves during training,
with confidence intervals to assess significance at each timestep.
"""

import matplotlib.pyplot as plt
import pandas as pd

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

    # Debug: Check if std values are NaN/0
    print(f"  DEBUG: std values range: {stats['std'].min():.6f} to {stats['std'].max():.6f}")
    print(f"  DEBUG: NaN count in std: {stats['std'].isna().sum()}")
    print(f"  DEBUG: Zero count in std: {(stats['std'] == 0).sum()}")

    return stats, combined


def compute_difference_stats(contrastive_stats: pd.DataFrame, no_contrastive_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference between conditions at each timestep.

    Args:
        contrastive_stats: Statistics for contrastive runs
        no_contrastive_stats: Statistics for non-contrastive runs

    Returns:
        DataFrame with columns: _step, difference_mean, difference_std, difference_se
    """
    # Merge on timestep
    merged = pd.merge(
        contrastive_stats,
        no_contrastive_stats,
        on="_step",
        suffixes=("_with", "_without"),
    )

    # Compute difference (With - Without)
    merged["difference_mean"] = merged["mean_with"] - merged["mean_without"]

    # Compute standard error of the difference
    # SE = sqrt(var1/n1 + var2/n2)
    # Handle NaN values by replacing with 0 for missing std values
    # But only replace NaN, not actual 0 values
    std_with = merged["std_with"].fillna(0)
    std_without = merged["std_without"].fillna(0)
    count_with = merged["count_with"].fillna(1)
    count_without = merged["count_without"].fillna(1)

    # Debug: Check what we're working with
    print(f"  DEBUG: After fillna - std_with range: {std_with.min():.6f} to {std_with.max():.6f}")
    print(f"  DEBUG: After fillna - std_without range: {std_without.min():.6f} to {std_without.max():.6f}")
    print(f"  DEBUG: Non-zero std_with count: {(std_with > 0).sum()}")
    print(f"  DEBUG: Non-zero std_without count: {(std_without > 0).sum()}")

    merged["difference_se"] = ((std_with**2) / count_with + (std_without**2) / count_without) ** 0.5

    # Also compute pooled std for reference
    merged["difference_std"] = ((std_with**2 + std_without**2) / 2) ** 0.5

    # Debug: Check the computed values
    print(f"  DEBUG: std_with range: {std_with.min():.6f} to {std_with.max():.6f}")
    print(f"  DEBUG: std_without range: {std_without.min():.6f} to {std_without.max():.6f}")
    print(f"  DEBUG: difference_se range: {merged['difference_se'].min():.6f} to {merged['difference_se'].max():.6f}")

    return merged[["_step", "difference_mean", "difference_std", "difference_se"]]


def plot_difference_curve(
    difference_stats: pd.DataFrame,
    contrastive_stats: pd.DataFrame,
    no_contrastive_stats: pd.DataFrame,
    metric_key: str = "overview/reward",
    save_path: str | None = None,
) -> None:
    """
    Plot the difference curve with confidence intervals.

    Args:
        difference_stats: DataFrame with difference statistics
        contrastive_stats: Statistics for contrastive runs (for final values)
        no_contrastive_stats: Statistics for non-contrastive runs (for final values)
        metric_key: Name of the metric being plotted
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract readable metric name
    metric_name = metric_key.split("/")[-1].replace("_", " ").replace(".", " ").title()

    # Plot difference curve
    ax.plot(
        difference_stats["_step"],
        difference_stats["difference_mean"],
        color="#E63946",
        linewidth=3,
        label="With - Without Contrastive",
        zorder=3,
    )

    # Add confidence interval (±1 SE)
    ax.fill_between(
        difference_stats["_step"],
        difference_stats["difference_mean"] - difference_stats["difference_se"],
        difference_stats["difference_mean"] + difference_stats["difference_se"],
        alpha=0.3,
        color="#E63946",
        label="±1 SE (68% CI)",
        zorder=2,
    )

    # Add wider confidence interval (±2 SE ≈ 95% CI)
    ax.fill_between(
        difference_stats["_step"],
        difference_stats["difference_mean"] - 2 * difference_stats["difference_se"],
        difference_stats["difference_mean"] + 2 * difference_stats["difference_se"],
        alpha=0.15,
        color="#E63946",
        label="±2 SE (95% CI)",
        zorder=1,
    )

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.5, zorder=0)

    # Labels and title
    ax.set_xlabel("Training Steps", fontsize=14, fontweight="bold")
    ax.set_ylabel("Performance Difference\n(With - Without Contrastive)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Learning Curve Difference: {metric_name}\n30 seeds, PPO hypers, ViT, ABES",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=12, loc="best", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add text annotation explaining positive/negative
    ax.text(
        0.02,
        0.98,
        "Positive = Contrastive Better\nNegative = Contrastive Worse",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # Add statistics text box in bottom right
    if not difference_stats.empty:
        # Get final difference
        final_diff = difference_stats.iloc[-1]["difference_mean"]
        final_se = difference_stats.iloc[-1]["difference_se"]

        # Handle NaN values in display
        final_se_str = f"{final_se:.4f}" if not pd.isna(final_se) else "N/A"

        # Get maximum difference
        max_idx = difference_stats["difference_mean"].abs().idxmax()
        max_diff = difference_stats.loc[max_idx, "difference_mean"]
        max_se = difference_stats.loc[max_idx, "difference_se"]
        max_step = difference_stats.loc[max_idx, "_step"]

        # Handle NaN values in display
        max_se_str = f"{max_se:.4f}" if not pd.isna(max_se) else "N/A"

        # Compute final performance values
        contrastive_final = contrastive_stats.iloc[-1]["mean"]
        no_contrastive_final = no_contrastive_stats.iloc[-1]["mean"]

        stats_text = (
            f"Final Performance:\n"
            f"With: {contrastive_final:.4f}\n"
            f"Without: {no_contrastive_final:.4f}\n"
            f"Difference: {final_diff:.4f} ± {final_se_str}\n"
            f"\n"
            f"Largest Gap:\n"
            f"{max_diff:.4f} ± {max_se_str} at {max_step:,.0f} steps"
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

    # Format x-axis
    max_step = difference_stats["_step"].max()
    if max_step > 1_000_000:
        ax.ticklabel_format(style="plain", axis="x")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e6:.1f}M"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def main():
    # Configuration
    BASE_RUN_NAME_1 = "tasha.10.16.shaped_hypers_vit"
    BASE_RUN_NAME_2 = "tasha.10.15.shaped_hypers_vit_2"

    # 27 seeds from first base run
    SEEDS_BATCH_1 = [
        67,
        134,
        201,
        268,
        335,
        402,
        469,
        536,
        603,
        670,
        737,
        804,
        871,
        938,
        1005,
        1072,
        1139,
        1206,
        1273,
        1340,
        1407,
        1474,
        1541,
        1608,
        1675,
        1742,
        1809,
    ]

    # 3 additional seeds from second base run
    SEEDS_BATCH_2 = [42, 123, 456]

    # Which metrics to plot
    METRICS_TO_PLOT = [
        "env_agent/heart.get",
    ]

    # Number of samples to fetch from WandB
    NUM_SAMPLES = 5000

    # Maximum timestep to include
    MAX_TIMESTEP = 2_000_000_000  # 2 billion

    print("=" * 80)
    print("Learning Curve Difference Analysis (30 Seeds)")
    print("=" * 80)
    print(f"\nBase run name 1: {BASE_RUN_NAME_1}")
    print(f"Seeds batch 1 ({len(SEEDS_BATCH_1)}): {SEEDS_BATCH_1}")
    print(f"\nBase run name 2: {BASE_RUN_NAME_2}")
    print(f"Seeds batch 2 ({len(SEEDS_BATCH_2)}): {SEEDS_BATCH_2}")
    print(f"\nMetrics: {METRICS_TO_PLOT}")
    print(f"Max timestep: {MAX_TIMESTEP:,}")
    print(f"\n{'=' * 80}\n")

    # Build run names for batch 1 (27 seeds)
    contrastive_runs_batch1 = [f"{BASE_RUN_NAME_1}.contrastive.seed{seed}" for seed in SEEDS_BATCH_1]
    no_contrastive_runs_batch1 = [f"{BASE_RUN_NAME_1}.no_contrastive.seed{seed}" for seed in SEEDS_BATCH_1]

    # Build run names for batch 2 (3 seeds)
    contrastive_runs_batch2 = [f"{BASE_RUN_NAME_2}.contrastive.seed{seed}" for seed in SEEDS_BATCH_2]
    no_contrastive_runs_batch2 = [f"{BASE_RUN_NAME_2}.no_contrastive.seed{seed}" for seed in SEEDS_BATCH_2]

    # Combine all runs
    contrastive_runs = contrastive_runs_batch1 + contrastive_runs_batch2
    no_contrastive_runs = no_contrastive_runs_batch1 + no_contrastive_runs_batch2

    print(f"Contrastive runs ({len(contrastive_runs)}):")
    for run in contrastive_runs[:3]:
        print(f"  - {run}")
    print(f"  ... and {len(contrastive_runs) - 3} more")

    print(f"\nNon-contrastive runs ({len(no_contrastive_runs)}):")
    for run in no_contrastive_runs[:3]:
        print(f"  - {run}")
    print(f"  ... and {len(no_contrastive_runs) - 3} more")

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

        # Compute statistics
        contrastive_stats, _ = compute_mean_across_runs(contrastive_metrics, metric)
        no_contrastive_stats, _ = compute_mean_across_runs(no_contrastive_metrics, metric)

        if contrastive_stats.empty or no_contrastive_stats.empty:
            print(f"  ⚠️  No data available for {metric}, skipping...")
            continue

        # Compute difference statistics
        difference_stats = compute_difference_stats(contrastive_stats, no_contrastive_stats)

        # Print summary - find the last timestep with meaningful std
        # Look for the last timestep where both conditions have std > 0
        valid_timesteps = difference_stats[
            (contrastive_stats["std"] > 0) & (no_contrastive_stats["std"] > 0)
        ]

        if not valid_timesteps.empty:
            final_diff = valid_timesteps.iloc[-1]["difference_mean"]
            final_se = valid_timesteps.iloc[-1]["difference_se"]
            final_step = valid_timesteps.iloc[-1]["_step"]
            final_se_str = f"{final_se:.4f}" if not pd.isna(final_se) else "N/A"
            print(f"  Final difference (last valid timestep): {final_diff:.4f} ± {final_se_str} at step {final_step:,.0f}")
        else:
            print(f"  Final difference: No valid timesteps with std > 0")

        max_idx = difference_stats["difference_mean"].abs().idxmax()
        max_diff = difference_stats.loc[max_idx, "difference_mean"]
        max_se = difference_stats.loc[max_idx, "difference_se"]
        max_step = difference_stats.loc[max_idx, "_step"]
        max_se_str = f"{max_se:.4f}" if not pd.isna(max_se) else "N/A"
        print(f"  Largest gap: {max_diff:.4f} ± {max_se_str} at step {max_step:,.0f}")

        # Create save path
        metric_filename = metric.replace("/", "_")
        save_path = f"difference_curve_30seeds_{metric_filename}.png"

        # Plot
        plot_difference_curve(
            difference_stats,
            contrastive_stats,
            no_contrastive_stats,
            metric_key=metric,
            save_path=save_path,
        )

    print(f"\n{'=' * 80}")
    print("Done! All difference curves generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()

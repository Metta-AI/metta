#!/usr/bin/env python3
"""Bootstrap variance analysis with resampling.

This script uses bootstrap resampling (with replacement) to estimate
variance stability more robustly than sequential sampling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add paths for imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "common" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "mettagrid" / "python" / "src"))

from experiments.recipes.prod_benchmark.statistics.analysis import (  # noqa: E402
    FetchSpec,
    SummarySpec,
    _fetch_series,
    _reduce_summary,
)


def fetch_all_auc_values(
    run_ids: list[str],
    metric_key: str,
    percent: float,
    samples: int,
    min_timesteps: float | None,
) -> tuple[list[float], list[str], list[tuple[str, float | str]]]:
    """Fetch AUC values for all runs."""
    fetch_spec = FetchSpec(samples=samples)
    summary_spec = SummarySpec(
        type="auc",
        percent=percent,
        percent_window="last",
        normalize_steps=True,
    )

    auc_values = []
    included_runs = []
    excluded_runs = []

    print("Fetching data from W&B...")
    for run_id in run_ids:
        try:
            series = _fetch_series(run_id, metric_key, fetch_spec)

            # Check if run meets minimum timestep requirement
            max_step = float(series.steps[-1])
            if min_timesteps is not None and max_step < min_timesteps:
                print(
                    f"  EXCLUDED {run_id}: Only reaches {max_step:,.0f} timesteps (need {min_timesteps:,.0f})"
                )
                excluded_runs.append((run_id, max_step))
                continue

            auc = _reduce_summary(series, summary_spec)
            auc_values.append(auc)
            included_runs.append(run_id)
            print(f"  {run_id}: AUC = {auc:.6f} (max timestep: {max_step:,.0f})")
        except Exception as e:
            print(f"  Warning: Failed to process {run_id}: {e}")
            excluded_runs.append((run_id, "error"))
            continue

    return auc_values, included_runs, excluded_runs


def bootstrap_variance_curves(
    auc_values: list[float],
    n_bootstrap: int = 1000,
    max_sample_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute variance curves using bootstrap resampling.

    Args:
        auc_values: List of AUC values from all runs
        n_bootstrap: Number of bootstrap iterations
        max_sample_size: Maximum sample size to compute (default: len(auc_values))

    Returns:
        Dictionary with:
            - cv_mean: Mean CV for each sample size
            - cv_std: Std of CV for each sample size
            - cv_q25: 25th percentile of CV
            - cv_q75: 75th percentile of CV
            - change_mean: Mean CV change for each sample size
            - change_std: Std of CV change
    """
    n_runs = len(auc_values)
    if max_sample_size is None:
        max_sample_size = n_runs

    auc_array = np.array(auc_values)

    # Store CV values for each bootstrap iteration
    cv_bootstrap = np.zeros((n_bootstrap, max_sample_size))

    print(f"\nBootstrap resampling {n_bootstrap} times...")
    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            print(f"  Progress: {b + 1}/{n_bootstrap}")

        # Resample with replacement
        resampled_indices = np.random.choice(n_runs, size=n_runs, replace=True)
        resampled_auc = auc_array[resampled_indices]

        # Compute CV for each sample size
        for n in range(1, max_sample_size + 1):
            subset = resampled_auc[:n]
            if len(subset) > 1:
                mean_val = np.mean(subset)
                std_val = np.std(subset, ddof=1)
                if abs(mean_val) > 1e-10:
                    cv = abs(std_val / mean_val)
                else:
                    cv = float("inf")
            else:
                cv = float("inf")
            cv_bootstrap[b, n - 1] = cv

    # Compute statistics across bootstrap samples
    # Filter out inf values for statistics
    cv_bootstrap_filtered = np.where(np.isinf(cv_bootstrap), np.nan, cv_bootstrap)

    cv_mean = np.nanmean(cv_bootstrap_filtered, axis=0)
    cv_std = np.nanstd(cv_bootstrap_filtered, axis=0)
    cv_q25 = np.nanpercentile(cv_bootstrap_filtered, 25, axis=0)
    cv_q75 = np.nanpercentile(cv_bootstrap_filtered, 75, axis=0)

    # Compute change in CV between consecutive sample sizes
    change_bootstrap = np.zeros((n_bootstrap, max_sample_size))
    change_bootstrap[:, 0] = np.inf  # First value has no previous to compare

    for b in range(n_bootstrap):
        for n in range(1, max_sample_size):
            prev_cv = cv_bootstrap[b, n - 1]
            curr_cv = cv_bootstrap[b, n]
            if not np.isinf(prev_cv) and not np.isinf(curr_cv) and prev_cv != 0:
                pct_change = abs(curr_cv - prev_cv) / abs(prev_cv)
                change_bootstrap[b, n] = pct_change
            else:
                change_bootstrap[b, n] = np.inf

    change_bootstrap_filtered = np.where(
        np.isinf(change_bootstrap), np.nan, change_bootstrap
    )
    change_mean = np.nanmean(change_bootstrap_filtered, axis=0)
    change_std = np.nanstd(change_bootstrap_filtered, axis=0)

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_q25": cv_q25,
        "cv_q75": cv_q75,
        "change_mean": change_mean,
        "change_std": change_std,
    }


def find_stabilization_point(change_mean: np.ndarray, threshold: float) -> int | None:
    """Find first sample size where CV change drops below threshold."""
    for n in range(1, len(change_mean)):
        if not np.isnan(change_mean[n]) and change_mean[n] < threshold:
            return n + 1  # +1 because index starts at 0
    return None


def plot_bootstrap_variance(
    stats: dict[str, np.ndarray],
    threshold: float,
    output_path: str,
    percent: float,
    max_x: int = 20,
):
    """Create dual-panel plot showing CV and CV change."""
    sample_sizes = np.arange(1, len(stats["cv_mean"]) + 1)

    # Find stabilization point
    threshold_n = find_stabilization_point(stats["change_mean"], threshold)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Coefficient of Variation
    ax1.plot(
        sample_sizes,
        stats["cv_mean"] * 100,
        marker="o",
        linewidth=2,
        markersize=6,
        color="blue",
        label="Mean CV",
    )
    ax1.fill_between(
        sample_sizes,
        stats["cv_q25"] * 100,
        stats["cv_q75"] * 100,
        alpha=0.3,
        color="blue",
        label="IQR (25th-75th percentile)",
    )

    if threshold_n:
        ax1.axvline(
            x=threshold_n,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Stabilized at N={threshold_n}",
        )
        ax1.scatter(
            [threshold_n],
            [stats["cv_mean"][threshold_n - 1] * 100],
            color="green",
            s=300,
            marker="*",
            zorder=5,
            edgecolor="black",
            linewidth=2,
        )

    ax1.set_ylabel("Coefficient of Variation (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Bootstrap Variance Analysis: Last {percent * 100:.0f}% of Training\n(1000 bootstrap samples with replacement)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")
    ax1.set_xlim(0, max_x + 1)

    # Panel 2: CV Change Between Consecutive Points
    # Skip first point (always inf)
    valid_indices = ~np.isnan(stats["change_mean"])
    valid_x = sample_sizes[valid_indices]
    valid_y = stats["change_mean"][valid_indices] * 100

    ax2.plot(
        valid_x,
        valid_y,
        marker="s",
        linewidth=2,
        markersize=6,
        color="orange",
        label="Mean CV change",
    )

    # Add threshold line
    ax2.axhline(
        y=threshold * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target: CV change < {threshold * 100:.0f}%",
    )

    if threshold_n:
        ax2.axvline(
            x=threshold_n,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Stabilized at N={threshold_n}",
        )
        ax2.scatter(
            [threshold_n],
            [stats["change_mean"][threshold_n - 1] * 100],
            color="green",
            s=300,
            marker="*",
            zorder=5,
            edgecolor="black",
            linewidth=2,
        )

    ax2.set_xlabel("Number of Runs", fontsize=12, fontweight="bold")
    ax2.set_ylabel("CV Change from Previous (%)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="best")
    ax2.set_xlim(0, max_x + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap variance analysis with resampling"
    )
    parser.add_argument(
        "run_ids", nargs="+", help="List of run IDs to analyze (space-separated)"
    )
    parser.add_argument(
        "--metric",
        default="overview/reward",
        help="Metric to analyze (default: overview/reward)",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=0.25,
        help="Percentage of training data to analyze (default: 0.25 for last 25%%)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Variance change threshold (default: 0.05 for 5%%)",
    )
    parser.add_argument(
        "--output",
        default="variance_bootstrap.png",
        help="Output plot path (default: variance_bootstrap.png)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of samples to request from W&B (default: 2000)",
    )
    parser.add_argument(
        "--min-timesteps",
        type=float,
        default=None,
        help="Minimum timesteps required for inclusion (default: None, auto-computed)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--max-x",
        type=int,
        default=20,
        help="Maximum x-axis value for plot (default: 20)",
    )

    args = parser.parse_args()

    # Auto-compute min_timesteps if not provided
    if args.min_timesteps is None and args.percent > 0:
        args.min_timesteps = 2_000_000_000 * (1 - args.percent)

    print(f"\nBootstrap Variance Analysis")
    print(f"{'=' * 70}")
    print(f"Analyzing {len(args.run_ids)} runs")
    print(f"Metric: {args.metric}")
    print(f"Window: Last {args.percent * 100:.0f}% of training")
    if args.min_timesteps:
        print(f"Minimum timesteps required: {args.min_timesteps:,.0f}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Variance threshold: {args.threshold * 100}%\n")

    # Fetch all AUC values
    auc_values, included_runs, excluded_runs = fetch_all_auc_values(
        args.run_ids,
        args.metric,
        args.percent,
        args.samples,
        args.min_timesteps,
    )

    if len(auc_values) < 2:
        print("Error: Need at least 2 runs for variance analysis")
        return 1

    print(f"\nSuccessfully included {len(auc_values)} runs in analysis")
    if excluded_runs:
        print(f"Excluded {len(excluded_runs)} runs")

    # Bootstrap variance curves
    stats = bootstrap_variance_curves(auc_values, args.n_bootstrap)

    # Find stabilization point
    threshold_n = find_stabilization_point(stats["change_mean"], args.threshold)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Included runs: {len(included_runs)} out of {len(args.run_ids)} total")
    print(f"Final CV (mean across bootstrap): {stats['cv_mean'][-1] * 100:.2f}%")
    print(
        f"\nLooking for when CV change < {args.threshold * 100:.0f}% (averaged across {args.n_bootstrap} bootstrap samples)..."
    )

    if threshold_n:
        print(
            f"✓ STABILIZED at N = {threshold_n} runs (mean CV change: {stats['change_mean'][threshold_n - 1] * 100:.2f}%)"
        )
        print(
            f"  → Adding more runs beyond {threshold_n} changes CV by < {args.threshold * 100:.0f}% on average"
        )
    else:
        print("✗ NOT YET STABLE (need more runs)")
    print("=" * 70)

    # Create plot
    plot_bootstrap_variance(
        stats,
        args.threshold,
        args.output,
        args.percent,
        args.max_x,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

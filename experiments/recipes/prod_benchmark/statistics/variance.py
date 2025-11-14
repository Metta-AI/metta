#!/usr/bin/env python3
"""Simple variance analysis for a specific timestep window.

Usage:
    python simple_variance.py run1 run2 run3 ... run15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add paths for imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "common" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "mettagrid" / "python" / "src"))

from experiments.recipes.prod_benchmark.statistics.analysis import (  # noqa: E402
    FetchSpec,
    SummarySpec,
    _fetch_series,
    _reduce_summary,
)


def compute_variance_curve(
    run_ids: list[str],
    metric_key: str,
    percent: float = 0.25,
    samples: int = 2000,
    min_timesteps: float | None = None,
    bootstrap_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> tuple[list[int], list[float], list[tuple[float, float]], list[str]]:
    """Compute coefficient of variation as a function of sample size.

    Args:
        run_ids: List of W&B run IDs
        metric_key: Metric to analyze (e.g., 'overview/reward')
        percent: Percentage of training data to use (default: 0.25 for last 25%)
        samples: Number of samples to request from W&B (default: 2000)
        min_timesteps: Minimum timesteps required for a run to be included (default: None)

    Returns:
        (
            sample_sizes,
            mean_cv_values,
            ci_bounds,
            included_runs,
        ): Sample sizes, bootstrapped mean CV values, CI bounds, and included run IDs
    """
    # Fetch and compute AUC for each run using percentage-based window
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

    if len(auc_values) < 2:
        raise ValueError("Need at least 2 successful runs for variance analysis")

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be >= 1")

    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")

    print(f"\nSuccessfully included {len(auc_values)} runs in analysis")
    if excluded_runs:
        print(f"Excluded {len(excluded_runs)} runs that didn't meet criteria")

    rng = np.random.default_rng(seed)
    sample_sizes = list(range(1, len(auc_values) + 1))
    mean_cv_values: list[float] = []
    ci_bounds: list[tuple[float, float]] = []

    lower_quantile = (1 - ci_level) / 2
    upper_quantile = 1 - lower_quantile

    for n in sample_sizes:
        if n == 1:
            mean_cv_values.append(float("inf"))
            ci_bounds.append((float("inf"), float("inf")))
            continue

        bootstrap_cvs = []
        for _ in range(bootstrap_iterations):
            subset = rng.choice(auc_values, size=n, replace=True)
            mean_val = float(np.mean(subset))
            std_val = float(np.std(subset, ddof=1))
            if abs(mean_val) > 1e-10:
                cv = abs(std_val / mean_val)
            else:
                cv = float("inf")
            bootstrap_cvs.append(cv)

        mean_cv = float(np.mean(bootstrap_cvs))
        lower_bound = float(np.quantile(bootstrap_cvs, lower_quantile))
        upper_bound = float(np.quantile(bootstrap_cvs, upper_quantile))
        mean_cv_values.append(mean_cv)
        ci_bounds.append((lower_bound, upper_bound))

    return sample_sizes, mean_cv_values, ci_bounds, included_runs


def plot_variance(
    sample_sizes: list[int],
    mean_cv_values: list[float],
    ci_bounds: list[tuple[float, float]],
    threshold: float,
    output_path: str,
    percent: float,
    ci_level: float,
):
    """Create a simple plot of CV vs sample size."""
    # Filter out inf values for plotting
    x_vals = []
    y_vals = []
    lower_vals = []
    upper_vals = []
    for n, cv, bounds in zip(sample_sizes, mean_cv_values, ci_bounds):
        if cv != float("inf"):
            x_vals.append(n)
            y_vals.append(cv * 100)
            lower_vals.append(bounds[0] * 100)
            upper_vals.append(bounds[1] * 100)

    # Find threshold crossing
    threshold_n = None
    for i in range(1, len(mean_cv_values)):
        prev_cv = mean_cv_values[i - 1]
        curr_cv = mean_cv_values[i]
        if prev_cv != float("inf") and curr_cv != float("inf"):
            pct_change = abs(curr_cv - prev_cv) / abs(prev_cv)
            if pct_change < threshold:
                threshold_n = (
                    i + 1
                )  # +1 because index starts at 0 but sample sizes start at 1
                break

    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=8, color="blue")
    if lower_vals and upper_vals:
        plt.fill_between(
            x_vals,
            lower_vals,
            upper_vals,
            color="blue",
            alpha=0.15,
            label=f"{ci_level * 100:.0f}% bootstrap CI",
        )

    if threshold_n:
        plt.axvline(
            x=threshold_n,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Stabilized at N={threshold_n}\n(CV change < {threshold * 100:.0f}%)",
        )
        plt.scatter(
            [threshold_n],
            [mean_cv_values[threshold_n - 1] * 100],
            color="green",
            s=300,
            marker="*",
            zorder=5,
            edgecolor="black",
            linewidth=2,
        )

    plt.xlabel("Number of Runs", fontsize=14, fontweight="bold")
    plt.ylabel("Coefficient of Variation (%)", fontsize=14, fontweight="bold")
    plt.title(
        f"Variance Analysis: Last {percent * 100:.0f}% of Training\n(Stabilizes when CV change between consecutive points < {threshold * 100:.0f}%)",
        fontsize=15,
        fontweight="bold",
    )

    # Restrict x-axis to first 15 runs so emphasis stays on early behavior
    plt.xlim(0, 15)
    plt.xticks(range(0, 16, 2))

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple variance analysis for the last portion of training"
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
        default="variance_simple.png",
        help="Output plot path (default: variance_simple.png)",
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
        help="Minimum timesteps required for inclusion (default: None, auto-computed from percent)",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap resamples per sample size (default: 1000)",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence interval level for bootstrap bounds (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible bootstrapping (default: None)",
    )

    args = parser.parse_args()

    # Auto-compute min_timesteps if not provided (assumes 2B training)
    if args.min_timesteps is None and args.percent > 0:
        # If analyzing last 25% of 2B training, minimum should be 1.5B
        args.min_timesteps = 2_000_000_000 * (1 - args.percent)

    print(f"\nAnalyzing {len(args.run_ids)} runs...")
    print(f"Metric: {args.metric}")
    print(f"Window: Last {args.percent * 100:.0f}% of training")
    if args.min_timesteps:
        print(f"Minimum timesteps required: {args.min_timesteps:,.0f}")
    print(f"Samples per run: {args.samples}")
    print(f"Bootstrap iterations per point: {args.bootstrap_iterations}")
    print(f"Variance threshold: {args.threshold * 100}%\n")

    # Compute variance curve
    (
        sample_sizes,
        mean_cv_values,
        ci_bounds,
        included_runs,
    ) = compute_variance_curve(
        args.run_ids,
        args.metric,
        args.percent,
        args.samples,
        args.min_timesteps,
        args.bootstrap_iterations,
        args.ci_level,
        args.seed,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Included runs: {len(included_runs)} out of {len(args.run_ids)} total")
    print(f"Final CV (N={len(sample_sizes)}): {mean_cv_values[-1] * 100:.2f}%")
    print(
        f"\nLooking for when CV change between consecutive samples < {args.threshold * 100:.0f}%..."
    )

    # Find threshold crossing
    threshold_n = None
    stabilization_change = None
    for i in range(1, len(mean_cv_values)):
        prev_cv = mean_cv_values[i - 1]
        curr_cv = mean_cv_values[i]
        if prev_cv != float("inf") and curr_cv != float("inf"):
            pct_change = abs(curr_cv - prev_cv) / abs(prev_cv)
            if pct_change < args.threshold:
                threshold_n = i + 1
                stabilization_change = pct_change
                break

    if threshold_n:
        print(
            f"✓ STABILIZED at N = {threshold_n} runs (CV change: {stabilization_change * 100:.2f}%)"
        )
        print(
            f"  → Adding more runs beyond {threshold_n} changes CV by < {args.threshold * 100:.0f}%"
        )
    else:
        print("✗ NOT YET STABLE (need more runs)")
    print("=" * 70)

    # Create plot
    plot_variance(
        sample_sizes,
        mean_cv_values,
        ci_bounds,
        args.threshold,
        args.output,
        args.percent,
        args.ci_level,
    )


if __name__ == "__main__":
    main()

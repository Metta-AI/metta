#!/usr/bin/env python3
"""Variance analysis focused on the 1.5B→2B timestep window.

Given a list of Weights & Biases run IDs, this script:
1. Fetches the requested metric between 1.5B and 2B timesteps and computes the
   normalized area under the curve (AUC) for each run.
2. Bootstraps the mean AUC 1,000 times per sample size (by default) to obtain
   confidence intervals and the variance of the bootstrap distribution.
3. Plots variance (y-axis) against the number of sampled runs (x-axis) and
   reports the earliest point where the variance change is below the requested
   threshold (default 5%).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository-local imports work when invoked as a script
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "common" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "mettagrid" / "python" / "src"))

from experiments.recipes.prod_benchmark.statistics.analysis import (  # noqa: E402
    FetchSpec,
    SummarySpec,
    _fetch_series,
    _reduce_summary,
)

WINDOW_START = 1_500_000_000
WINDOW_END = 2_000_000_000


def _infer_run_label(run_ids: Sequence[str]) -> str:
    """Create a readable label from the provided run IDs."""

    if not run_ids:
        return ""

    first = run_ids[0]
    if ".seed" in first:
        candidate = first.split(".seed", 1)[0]
        if all(run_id.startswith(candidate) for run_id in run_ids):
            return candidate

    prefix = os.path.commonprefix(run_ids)
    prefix = prefix.rstrip("._- ")
    return prefix or first


def _slugify(text: str) -> str:
    """Return a filesystem-friendly slug."""

    if not text:
        return "variance_v2"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("._-") or "variance_v2"


def _format_step(value: float) -> str:
    """Return a human friendly timestep string."""

    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    return f"{value:,.0f}"


def _compute_auc_values(
    run_ids: Sequence[str],
    metric_key: str,
    samples: int,
    window_start: int,
    window_end: int,
) -> tuple[list[float], list[str]]:
    """Fetch the desired metric and compute normalized AUC per run."""

    fetch_spec = FetchSpec(samples=samples, min_step=window_start, max_step=window_end)
    summary_spec = SummarySpec(
        type="auc",
        percent=None,
        step_min=window_start,
        step_max=window_end,
        normalize_steps=True,
    )

    auc_values: list[float] = []
    included_runs: list[str] = []
    for run_id in run_ids:
        try:
            series = _fetch_series(run_id, metric_key, fetch_spec)
            auc = _reduce_summary(series, summary_spec)
            auc_values.append(auc)
            included_runs.append(run_id)
            print(f"  ✓ {run_id}: AUC={auc:.6f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {run_id}: skipped ({exc})")

    if len(auc_values) < 2:
        raise ValueError("Need at least two successful runs for variance analysis")

    return auc_values, included_runs


def _bootstrap_statistics(
    auc_values: Sequence[float],
    bootstrap_iterations: int,
    ci_level: float,
    seed: int | None = None,
) -> tuple[list[int], list[float], list[tuple[float, float]]]:
    """Return sample sizes, bootstrap variances, and CI bounds."""

    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be >= 1")

    auc_array = np.asarray(auc_values, dtype=float)
    rng = np.random.default_rng(seed)
    sample_sizes = list(range(1, len(auc_array) + 1))
    variance_values: list[float] = []
    ci_bounds: list[tuple[float, float]] = []
    lower_q = (1 - ci_level) / 2
    upper_q = 1 - lower_q

    for n in sample_sizes:
        idx = rng.integers(0, len(auc_array), size=(bootstrap_iterations, n))
        subset = auc_array[idx]
        means = np.mean(subset, axis=1)
        variance = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
        variance_values.append(variance)
        lower = float(np.quantile(means, lower_q))
        upper = float(np.quantile(means, upper_q))
        ci_bounds.append((lower, upper))

    return sample_sizes, variance_values, ci_bounds


def _find_stable_sample_size(
    variance_values: Sequence[float],
    threshold: float,
) -> tuple[int | None, float | None]:
    """Return first sample size where variance change drops below threshold."""

    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    for idx in range(1, len(variance_values)):
        prev = variance_values[idx - 1]
        curr = variance_values[idx]
        if not math.isfinite(prev) or not math.isfinite(curr):
            continue
        denom = prev if abs(prev) > 1e-12 else max(abs(curr), 1e-12)
        change = abs(curr - prev) / denom
        if change < threshold:
            return idx + 1, change
    return None, None


def _plot_variance(
    sample_sizes: Sequence[int],
    variance_values: Sequence[float],
    output_path: Path,
    run_label: str,
    ci_bounds: Sequence[tuple[float, float]],
    threshold_sample: int | None,
) -> None:
    """Render variance vs sample size and annotate stabilization if present."""

    plt.figure(figsize=(12, 7))
    plt.plot(sample_sizes, variance_values, marker="o", linewidth=2, color="purple")
    plt.xlabel("Number of Runs (samples)")
    plt.ylabel("Variance of Bootstrapped Mean AUC")
    plt.title(
        "\n".join(
            [
                run_label,
                f"AUC var between {_format_step(WINDOW_START)} and {_format_step(WINDOW_END)} steps",
            ]
        ).strip(),
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    if threshold_sample is not None:
        idx = threshold_sample - 1
        plt.axvline(threshold_sample, color="green", linestyle="--", linewidth=1.5)
        plt.scatter(
            [threshold_sample],
            [variance_values[idx]],
            color="green",
            zorder=5,
            label=f"Stabilizes at N={threshold_sample}",
        )
    if ci_bounds:
        # Secondary axis showing CI width to give context
        ci_widths = [hi - lo for (lo, hi) in ci_bounds]
        ax2 = plt.gca().twinx()
        ax2.plot(
            sample_sizes,
            ci_widths,
            color="orange",
            linestyle="--",
            label="CI width",
        )
        ax2.set_ylabel("Bootstrap CI width")
        # Combine legends
        handles, labels = plt.gca().get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles or handles2:
            plt.gca().legend(handles + handles2, labels + labels2, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def _print_summary(
    included_runs: Sequence[str],
    sample_sizes: Sequence[int],
    variance_values: Sequence[float],
    ci_bounds: Sequence[tuple[float, float]],
    threshold: float,
    threshold_sample: int | None,
    threshold_change: float | None,
) -> None:
    """Emit a human-readable summary of the analysis."""

    print("\n" + "=" * 70)
    print("Variance analysis summary")
    print("=" * 70)
    print(f"Analyzed runs: {len(included_runs)}")
    print(f"Final variance (N={sample_sizes[-1]}): {variance_values[-1]:.6f}")
    last_low, last_high = ci_bounds[-1]
    print(
        f"Final {len(included_runs)}-sample CI: "
        f"[{last_low:.6f}, {last_high:.6f}] (width {last_high - last_low:.6f})"
    )
    print(
        f"Stability threshold: change < {threshold * 100:.1f}% between adjacent sample counts"
    )
    if threshold_sample is not None and threshold_change is not None:
        print(
            f"✓ Stabilized at N={threshold_sample} (change={threshold_change * 100:.2f}%)"
        )
    else:
        print("✗ Not yet stable within the provided runs")
    print("=" * 70)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Bootstrap variance of AUC between 1.5B and 2B timesteps."
    )
    parser.add_argument("run_ids", nargs="+", help="List of W&B run IDs")
    parser.add_argument(
        "--metric",
        default="overview/reward",
        help="Metric key to analyze (default: overview/reward)",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap resamples per sample size (default: 1000)",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence interval level for bootstrap estimates (default: 0.95)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Relative variance change threshold (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4000,
        help="Number of history samples requested per run (default: 4000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination PNG path (default: statistics/<label>_variance_v2.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible bootstrapping",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title (default: inferred from run IDs)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_label = args.title or _infer_run_label(args.run_ids)
    print(f"Runs provided: {len(args.run_ids)}")
    print(f"Metric: {args.metric}")
    print(
        f"Timestep window: {_format_step(WINDOW_START)} → {_format_step(WINDOW_END)}"
    )
    print(f"History samples per run: {args.samples}")
    print(f"Bootstrap iterations: {args.bootstrap_iterations}")
    print(f"CI level: {args.ci_level:.2f}")
    print(f"Variance threshold: {args.threshold * 100:.1f}%")

    auc_values, included_runs = _compute_auc_values(
        args.run_ids,
        args.metric,
        args.samples,
        WINDOW_START,
        WINDOW_END,
    )
    print(f"\nSuccessfully processed {len(included_runs)} runs.")

    (
        sample_sizes,
        variance_values,
        ci_bounds,
    ) = _bootstrap_statistics(
        auc_values,
        args.bootstrap_iterations,
        args.ci_level,
        args.seed,
    )

    threshold_sample, threshold_change = _find_stable_sample_size(
        variance_values, args.threshold
    )

    output_path = (
        args.output
        if args.output
        else Path(__file__).parent / f"{_slugify(run_label)}_variance_v2.png"
    )
    _plot_variance(
        sample_sizes,
        variance_values,
        output_path,
        run_label or "variance_v2",
        ci_bounds,
        threshold_sample,
    )
    _print_summary(
        included_runs,
        sample_sizes,
        variance_values,
        ci_bounds,
        args.threshold,
        threshold_sample,
        threshold_change,
    )


if __name__ == "__main__":
    main()

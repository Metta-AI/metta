"""Variance Analysis for Production Benchmarks.

This script analyzes the variance in summary statistics across multiple runs
to determine the minimum number of runs needed for stable measurements.

The script computes two key summary statistics:
1. Area Under the Curve (AUC): Total reward accumulated over training
2. Derivative (Slope): Rate of learning progress over time

For each statistic, it computes the coefficient of variation (CV = std/mean)
as a function of sample size (1 to N runs) and identifies when the variance
drops below a configurable threshold (default: 5%).

Usage:
    Command line:
        python variance.py run_id_1 run_id_2 ... run_id_15 \\
            --metric overview/reward \\
            --threshold 0.05 \\
            --output variance_plot.png

    As a tool:
        from experiments.recipes.prod_benchmark.variance import variance_analysis

        tool = variance_analysis(
            run_ids=["run1", "run2", "run3", ...],
            metric_key="overview/reward",
            variance_threshold=0.05,
            output_path="variance_analysis.png"
        )
        tool.invoke({})

Output:
    - Console: Reports the minimum N where variance < threshold for each statistic
    - Plot: Two-panel figure showing variance curves for AUC and Derivative
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from metta.common.tool import Tool
from pydantic import Field

from experiments.recipes.prod_benchmark.statistics.analysis import (
    FetchSpec,
    SummarySpec,
    _fetch_series,
    _reduce_summary,
)


@dataclass
class VarianceResults:
    """Results from variance analysis."""

    num_samples: list[int]
    auc_variance: list[float]
    derivative_variance: list[float]
    auc_variance_change: list[float]
    derivative_variance_change: list[float]
    auc_threshold_n: int | None
    derivative_threshold_n: int | None


class VarianceAnalysisTool(Tool):
    """Analyze variance across multiple runs to determine stability."""

    run_ids: list[str] = Field(
        description="List of run IDs to analyze (should be 15 runs)"
    )
    metric_key: str = Field(default="overview/reward", description="Metric to analyze")
    variance_threshold: float = Field(
        default=0.05, description="Variance threshold as percentage (default 5%)"
    )
    fetch: FetchSpec = Field(default_factory=FetchSpec)
    summary: SummarySpec = Field(default_factory=SummarySpec)
    output_path: str | None = Field(
        default=None, description="Path to save plot (default: variance_analysis.png)"
    )

    def _compute_derivative(self, steps: np.ndarray, values: np.ndarray) -> float:
        """Compute the derivative (slope) of the reward curve using linear regression."""
        if len(steps) < 2:
            raise ValueError("Need at least 2 points to compute derivative")

        # Use linear regression to compute slope
        coefficients = np.polyfit(steps, values, 1)
        slope = float(coefficients[0])
        return slope

    def _compute_summary_statistics(
        self, run_ids: list[str]
    ) -> tuple[list[float], list[float]]:
        """Compute AUC and derivative for all runs."""
        auc_values = []
        derivative_values = []

        for run_id in run_ids:
            try:
                series = _fetch_series(run_id, self.metric_key, self.fetch)
                # Compute AUC
                auc = _reduce_summary(series, self.summary)
                auc_values.append(auc)

                # Compute derivative
                derivative = self._compute_derivative(series.steps, series.values)
                derivative_values.append(derivative)

            except Exception as e:
                print(f"Warning: Failed to process run {run_id}: {e}")
                continue

        return auc_values, derivative_values

    def _compute_variance_curve(
        self, values: list[float]
    ) -> tuple[list[int], list[float]]:
        """Compute variance as a function of sample size."""
        n_runs = len(values)
        sample_sizes = list(range(1, n_runs + 1))
        variances = []

        for n in sample_sizes:
            # Use first n samples
            subset = values[:n]
            if len(subset) > 1:
                # Compute coefficient of variation (CV = std/mean)
                # This gives us variance as a percentage
                mean_val = np.mean(subset)
                std_val = np.std(subset, ddof=1)
                if abs(mean_val) > 1e-10:
                    cv = abs(std_val / mean_val)
                else:
                    cv = float("inf")
                variances.append(cv)
            else:
                variances.append(float("inf"))

        return sample_sizes, variances

    def _compute_variance_change(self, variances: list[float]) -> list[float]:
        """Compute the rate of change in variance between consecutive sample sizes.

        Returns the percent change: |variance[i] - variance[i-1]| / variance[i-1]
        First element is inf since there's no previous value to compare.
        """
        changes = [float("inf")]  # No previous value for first sample

        for i in range(1, len(variances)):
            prev_var = variances[i - 1]
            curr_var = variances[i]

            # Avoid division by zero or inf
            if prev_var == float("inf") or prev_var == 0 or curr_var == float("inf"):
                changes.append(float("inf"))
            else:
                # Compute absolute percent change
                pct_change = abs(curr_var - prev_var) / abs(prev_var)
                changes.append(pct_change)

        return changes

    def _find_threshold_crossing(
        self, sample_sizes: list[int], variance_changes: list[float], threshold: float
    ) -> int | None:
        """Find the first sample size where variance change drops below threshold.

        This identifies when adding more runs doesn't significantly change the variance estimate.
        """
        for n, change in zip(sample_sizes, variance_changes):
            # Skip first value (which is inf)
            if n > 1 and change < threshold:
                return n
        return None

    def _plot_variance_analysis(self, results: VarianceResults, output_path: str):
        """Create visualization of variance change vs sample size."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        (ax1, ax2), (ax3, ax4) = axes

        # Plot 1: AUC Variance (absolute)
        ax1.plot(
            results.num_samples,
            [v * 100 for v in results.auc_variance],
            marker="o",
            linewidth=2,
            markersize=6,
            color="blue",
        )
        ax1.set_xlabel("Number of Runs", fontsize=12)
        ax1.set_ylabel("Coefficient of Variation (%)", fontsize=12)
        ax1.set_title("AUC: Absolute Variance", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Derivative Variance (absolute)
        ax2.plot(
            results.num_samples,
            [v * 100 for v in results.derivative_variance],
            marker="s",
            linewidth=2,
            markersize=6,
            color="orange",
        )
        ax2.set_xlabel("Number of Runs", fontsize=12)
        ax2.set_ylabel("Coefficient of Variation (%)", fontsize=12)
        ax2.set_title("Derivative: Absolute Variance", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Plot 3: AUC Variance Change (rate of change)
        # Filter out inf values for plotting
        auc_changes_filtered = [
            c if c != float("inf") else None for c in results.auc_variance_change
        ]
        ax3.plot(
            results.num_samples[1:],  # Skip first sample (has inf change)
            [c * 100 for c in auc_changes_filtered[1:] if c is not None],
            marker="o",
            linewidth=2,
            markersize=6,
            color="blue",
        )
        ax3.axhline(
            y=self.variance_threshold * 100,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"{self.variance_threshold * 100}% threshold",
        )
        if results.auc_threshold_n:
            ax3.axvline(
                x=results.auc_threshold_n,
                color="g",
                linestyle=":",
                linewidth=2,
                label=f"Stable at N={results.auc_threshold_n}",
            )
            ax3.scatter(
                [results.auc_threshold_n],
                [results.auc_variance_change[results.auc_threshold_n - 1] * 100],
                color="g",
                s=200,
                marker="*",
                zorder=5,
                edgecolor="black",
                linewidth=1.5,
            )
        ax3.set_xlabel("Number of Runs", fontsize=12)
        ax3.set_ylabel("Variance Change from Previous (%)", fontsize=12)
        ax3.set_title("AUC: Rate of Variance Change", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Derivative Variance Change (rate of change)
        deriv_changes_filtered = [
            c if c != float("inf") else None for c in results.derivative_variance_change
        ]
        ax4.plot(
            results.num_samples[1:],  # Skip first sample (has inf change)
            [c * 100 for c in deriv_changes_filtered[1:] if c is not None],
            marker="s",
            linewidth=2,
            markersize=6,
            color="orange",
        )
        ax4.axhline(
            y=self.variance_threshold * 100,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"{self.variance_threshold * 100}% threshold",
        )
        if results.derivative_threshold_n:
            ax4.axvline(
                x=results.derivative_threshold_n,
                color="g",
                linestyle=":",
                linewidth=2,
                label=f"Stable at N={results.derivative_threshold_n}",
            )
            ax4.scatter(
                [results.derivative_threshold_n],
                [
                    results.derivative_variance_change[
                        results.derivative_threshold_n - 1
                    ]
                    * 100
                ],
                color="g",
                s=200,
                marker="*",
                zorder=5,
                edgecolor="black",
                linewidth=1.5,
            )
        ax4.set_xlabel("Number of Runs", fontsize=12)
        ax4.set_ylabel("Variance Change from Previous (%)", fontsize=12)
        ax4.set_title(
            "Derivative: Rate of Variance Change", fontsize=14, fontweight="bold"
        )
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run variance analysis on the provided runs."""
        if len(self.run_ids) < 2:
            print("Error: Need at least 2 runs for variance analysis")
            return 1

        if len(self.run_ids) != 15:
            print(
                f"Warning: Expected 15 runs but got {len(self.run_ids)}. Continuing with provided runs..."
            )

        print(f"\nAnalyzing {len(self.run_ids)} runs...")
        print(f"Metric: {self.metric_key}")
        print(f"Variance threshold: {self.variance_threshold * 100}%")

        # Compute summary statistics
        auc_values, derivative_values = self._compute_summary_statistics(self.run_ids)

        if len(auc_values) < 2:
            print("Error: Not enough valid runs processed")
            return 1

        print(f"\nSuccessfully processed {len(auc_values)} runs")

        # Compute variance curves
        auc_samples, auc_variances = self._compute_variance_curve(auc_values)
        deriv_samples, deriv_variances = self._compute_variance_curve(derivative_values)

        # Compute variance change rates
        auc_variance_changes = self._compute_variance_change(auc_variances)
        deriv_variance_changes = self._compute_variance_change(deriv_variances)

        # Find threshold crossings (using variance change, not absolute variance)
        auc_threshold_n = self._find_threshold_crossing(
            auc_samples, auc_variance_changes, self.variance_threshold
        )
        derivative_threshold_n = self._find_threshold_crossing(
            deriv_samples, deriv_variance_changes, self.variance_threshold
        )

        # Create results
        results = VarianceResults(
            num_samples=auc_samples,
            auc_variance=auc_variances,
            derivative_variance=deriv_variances,
            auc_variance_change=auc_variance_changes,
            derivative_variance_change=deriv_variance_changes,
            auc_threshold_n=auc_threshold_n,
            derivative_threshold_n=derivative_threshold_n,
        )

        # Print results
        print("\n" + "=" * 70)
        print("VARIANCE ANALYSIS RESULTS")
        print("=" * 70)
        print(f"\nMetric: {self.metric_key}")
        print(f"Summary type: {self.summary.type}")
        print(f"Total runs analyzed: {len(auc_values)}")
        print(
            f"Stability threshold: Variance change < {self.variance_threshold * 100}%"
        )

        print("\n--- AUC Variance ---")
        print(f"Final variance (N={len(auc_values)}): {auc_variances[-1] * 100:.2f}%")
        if auc_threshold_n:
            change_at_n = auc_variance_changes[auc_threshold_n - 1] * 100
            print(
                f" STABILIZED at N = {auc_threshold_n} runs (change: {change_at_n:.2f}%)"
            )
        else:
            last_change = (
                auc_variance_changes[-1] * 100
                if auc_variance_changes[-1] != float("inf")
                else "inf"
            )
            print(
                f" NOT YET STABLE (last change: {last_change if isinstance(last_change, str) else f'{last_change:.2f}'}%, need more runs)"
            )

        print("\n--- Derivative Variance ---")
        print(
            f"Final variance (N={len(derivative_values)}): {deriv_variances[-1] * 100:.2f}%"
        )
        if derivative_threshold_n:
            change_at_n = deriv_variance_changes[derivative_threshold_n - 1] * 100
            print(
                f" STABILIZED at N = {derivative_threshold_n} runs (change: {change_at_n:.2f}%)"
            )
        else:
            last_change = (
                deriv_variance_changes[-1] * 100
                if deriv_variance_changes[-1] != float("inf")
                else "inf"
            )
            print(
                f" NOT YET STABLE (last change: {last_change if isinstance(last_change, str) else f'{last_change:.2f}'}%, need more runs)"
            )

        print("\n" + "=" * 70)

        # Create plot
        output_path = self.output_path or "variance_analysis.png"
        self._plot_variance_analysis(results, output_path)

        return 0


def variance_analysis(
    run_ids: list[str],
    metric_key: str = "overview/reward",
    variance_threshold: float = 0.05,
    output_path: str | None = None,
) -> VarianceAnalysisTool:
    """Create a variance analysis tool."""
    return VarianceAnalysisTool(
        run_ids=run_ids,
        metric_key=metric_key,
        variance_threshold=variance_threshold,
        output_path=output_path,
    )


def main():
    """Command-line interface for variance analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze variance across multiple runs"
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
        "--threshold",
        type=float,
        default=0.05,
        help="Variance threshold as decimal (default: 0.05 for 5%%)",
    )
    parser.add_argument(
        "--output", default="variance_analysis.png", help="Output plot path"
    )
    parser.add_argument(
        "--window-percent",
        type=float,
        default=None,
        help="Percent of data to use for AUC window (0-1). If None, uses all data. Default: None for full data",
    )

    args = parser.parse_args()

    # Create SummarySpec with custom percent if provided
    from experiments.recipes.prod_benchmark.statistics.analysis import SummarySpec

    summary_spec = (
        SummarySpec(percent=args.window_percent)
        if args.window_percent
        else SummarySpec(percent=None)
    )

    tool = variance_analysis(
        run_ids=args.run_ids,
        metric_key=args.metric,
        variance_threshold=args.threshold,
        output_path=args.output,
    )
    # Override the summary spec
    tool.summary = summary_spec
    tool.invoke({})


if __name__ == "__main__":
    main()

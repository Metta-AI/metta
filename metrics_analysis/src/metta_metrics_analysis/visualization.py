"""
Visualization Module for Metta Metrics Analysis.

This module provides plotting functions for visualizing analysis results.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class Visualizer:
    """Create visualizations for metrics analysis results."""

    def __init__(self, style: str = "seaborn", palette: str = "Set2"):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
            palette: Color palette for plots
        """
        plt.style.use(style)
        self.palette = palette
        self.colors = sns.color_palette(palette)

    def plot_iqm_comparison(
        self,
        iqm_results: pd.DataFrame,
        metric_name: str,
        group_column: str,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
    ) -> Figure:
        """
        Plot IQM values with confidence intervals.

        Args:
            iqm_results: DataFrame from compute_iqm_with_ci
            metric_name: Name of the metric being plotted
            group_column: Column used for grouping
            figsize: Figure size
            title: Plot title (auto-generated if None)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by IQM value for better visualization
        iqm_col = f"{metric_name}_iqm"
        ci_lower_col = f"{metric_name}_ci_lower"
        ci_upper_col = f"{metric_name}_ci_upper"

        sorted_results = iqm_results.sort_values(iqm_col, ascending=False)

        # Create bar plot with error bars
        x = range(len(sorted_results))
        labels = sorted_results[group_column].values
        iqm_values = sorted_results[iqm_col].values
        ci_lower = sorted_results[ci_lower_col].values
        ci_upper = sorted_results[ci_upper_col].values

        # Calculate error bar sizes
        yerr_lower = iqm_values - ci_lower
        yerr_upper = ci_upper - iqm_values
        yerr = [yerr_lower, yerr_upper]

        # Create bars
        bars = ax.bar(x, iqm_values, yerr=yerr, capsize=5, color=self.colors[: len(x)], alpha=0.8)

        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(f"{metric_name} (IQM)")
        ax.set_xlabel(group_column)

        if title is None:
            title = f"IQM Comparison: {metric_name} by {group_column}"
        ax.set_title(title)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, iqm_values, strict=False)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + yerr_upper[i],
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Add sample size annotations
        if "n_samples" in sorted_results.columns:
            for i, (bar, n) in enumerate(zip(bars, sorted_results["n_samples"], strict=False)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="gray",
                )

        plt.tight_layout()
        return fig

    def plot_performance_profiles(
        self,
        profiles: pd.DataFrame,
        figsize: tuple[int, int] = (10, 6),
        title: str = "Performance Profiles",
        log_scale: bool = True,
    ) -> Figure:
        """
        Plot performance profiles.

        Args:
            profiles: DataFrame from compute_performance_profiles
            figsize: Figure size
            title: Plot title
            log_scale: Whether to use log scale for x-axis

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each algorithm's profile
        for i, algorithm in enumerate(profiles.columns):
            ax.plot(
                profiles.index,
                profiles[algorithm],
                label=algorithm,
                color=self.colors[i % len(self.colors)],
                linewidth=2,
                marker="o",
                markersize=0,
            )

        # Customize plot
        ax.set_xlabel("Performance Ratio (τ)")
        ax.set_ylabel("Probability of being within τ of best")
        ax.set_title(title)

        if log_scale:
            ax.set_xscale("log")

        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        plt.tight_layout()
        return fig

    def plot_metric_over_time(
        self,
        data: pd.DataFrame,
        metric: str,
        group_by: str | None = None,
        figsize: tuple[int, int] = (12, 6),
        show_confidence: bool = True,
        confidence_alpha: float = 0.2,
    ) -> Figure:
        """
        Plot metric values over time (steps).

        Args:
            data: DataFrame with step and metric columns
            metric: Metric to plot
            group_by: Optional column to group by
            figsize: Figure size
            show_confidence: Whether to show confidence bands
            confidence_alpha: Alpha for confidence bands

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if group_by is None:
            # Single line plot
            grouped = data.groupby("step")[metric]
            mean_values = grouped.mean()

            ax.plot(mean_values.index, mean_values.values, color=self.colors[0], linewidth=2, label="Mean")

            if show_confidence:
                std_values = grouped.std()
                lower = mean_values - std_values
                upper = mean_values + std_values
                ax.fill_between(mean_values.index, lower, upper, alpha=confidence_alpha, color=self.colors[0])
        else:
            # Multiple lines
            for i, (name, group) in enumerate(data.groupby(group_by)):
                grouped = group.groupby("step")[metric]
                mean_values = grouped.mean()

                color = self.colors[i % len(self.colors)]
                ax.plot(mean_values.index, mean_values.values, color=color, linewidth=2, label=str(name))

                if show_confidence:
                    std_values = grouped.std()
                    lower = mean_values - std_values
                    upper = mean_values + std_values
                    ax.fill_between(mean_values.index, lower, upper, alpha=confidence_alpha, color=color)

        # Customize plot
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_scatter_comparison(
        self,
        data: pd.DataFrame,
        metric_x: str,
        metric_y: str,
        group_by: str | None = None,
        figsize: tuple[int, int] = (8, 8),
        show_diagonal: bool = True,
    ) -> Figure:
        """
        Create scatter plot comparing two metrics.

        Args:
            data: DataFrame with metrics
            metric_x: Metric for x-axis
            metric_y: Metric for y-axis
            group_by: Optional column to color by
            figsize: Figure size
            show_diagonal: Whether to show y=x line

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if group_by is None:
            ax.scatter(data[metric_x], data[metric_y], alpha=0.6, s=50, color=self.colors[0])
        else:
            for i, (name, group) in enumerate(data.groupby(group_by)):
                ax.scatter(
                    group[metric_x],
                    group[metric_y],
                    alpha=0.6,
                    s=50,
                    label=str(name),
                    color=self.colors[i % len(self.colors)],
                )
            ax.legend()

        # Add diagonal line
        if show_diagonal:
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, "k--", alpha=0.5, zorder=0)

        # Customize plot
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(f"{metric_y} vs {metric_x}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_optimality_gaps(
        self,
        gaps: pd.DataFrame,
        group_column: str,
        gap_column: str = "mean_optimality_gap_pct",
        figsize: tuple[int, int] = (10, 6),
    ) -> Figure:
        """
        Plot optimality gaps by group.

        Args:
            gaps: DataFrame from compute_optimality_gaps
            group_column: Column used for grouping
            gap_column: Gap column to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by gap value
        sorted_gaps = gaps.sort_values(gap_column)

        # Create horizontal bar plot
        y = range(len(sorted_gaps))
        labels = sorted_gaps[group_column].values
        gap_values = sorted_gaps[gap_column].values

        bars = ax.barh(y, gap_values, color=self.colors[: len(y)], alpha=0.8)

        # Customize plot
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Optimality Gap (%)")
        ax.set_title("Optimality Gaps by Algorithm")

        # Add value labels
        for bar, val in zip(bars, gap_values, strict=False):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)

        # Add vertical line at 0
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        return fig

    def create_summary_dashboard(self, data: dict[str, pd.DataFrame], figsize: tuple[int, int] = (16, 12)) -> Figure:
        """
        Create a dashboard with multiple plots.

        Args:
            data: Dictionary of DataFrames for different plots
            figsize: Overall figure size

        Returns:
            Matplotlib figure with subplots
        """
        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Add individual plots based on available data
        plot_idx = 0

        if "iqm_results" in data:
            ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            # Simplified IQM plot for dashboard
            self._plot_iqm_simple(ax, data["iqm_results"])
            plot_idx += 1

        if "performance_profiles" in data:
            ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_profiles_simple(ax, data["performance_profiles"])
            plot_idx += 1

        if "time_series" in data:
            ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_time_series_simple(ax, data["time_series"])
            plot_idx += 1

        if "optimality_gaps" in data:
            ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_gaps_simple(ax, data["optimality_gaps"])
            plot_idx += 1

        plt.suptitle("Metrics Analysis Dashboard", fontsize=16)
        return fig

    def _plot_iqm_simple(self, ax, data):
        """Simplified IQM plot for dashboard."""
        # Implementation would be similar to plot_iqm_comparison but simplified
        pass

    def _plot_profiles_simple(self, ax, data):
        """Simplified performance profiles plot for dashboard."""
        # Implementation would be similar to plot_performance_profiles but simplified
        pass

    def _plot_time_series_simple(self, ax, data):
        """Simplified time series plot for dashboard."""
        # Implementation would be similar to plot_metric_over_time but simplified
        pass

    def _plot_gaps_simple(self, ax, data):
        """Simplified optimality gaps plot for dashboard."""
        # Implementation would be similar to plot_optimality_gaps but simplified
        pass

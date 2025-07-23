"""
Visualization utilities for meta-analysis results.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


class VisualizationUtils:
    """
    Utility class for creating various visualizations for meta-analysis results.
    """

    def __init__(self):
        """Initialize the visualization utilities."""
        # Set default plotting style
        plt.style.use("default")
        self.colors = sns.color_palette("husl", 8)

    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Task Performance Correlations",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create a correlation heatmap.

        Args:
            correlation_matrix: DataFrame with correlation values
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            ax=ax,
            cbar_kws={"label": "Correlation Coefficient"},
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tasks")
        ax.set_ylabel("Tasks")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_performance_matrix(
        self,
        performance_matrix: pd.DataFrame,
        title: str = "Policy Performance Matrix",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create a performance matrix heatmap.

        Args:
            performance_matrix: DataFrame with performance values
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(performance_matrix, cmap="viridis", ax=ax, cbar_kws={"label": "Performance Score"})

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Tasks")
        ax.set_ylabel("Policies")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pca_scree(
        self, explained_variance_ratio: List[float], title: str = "PCA Scree Plot", save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a PCA scree plot.

        Args:
            explained_variance_ratio: List of explained variance ratios
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        n_components = len(explained_variance_ratio)
        x = range(1, n_components + 1)

        # Plot explained variance ratio
        ax.plot(x, explained_variance_ratio, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add cumulative line
        cumulative = np.cumsum(explained_variance_ratio)
        ax2 = ax.twinx()
        ax2.plot(x, cumulative, "r--", linewidth=2, label="Cumulative")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pca_loadings(
        self,
        components: np.ndarray,
        feature_names: List[str],
        title: str = "PCA Component Loadings",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create a PCA loadings heatmap.

        Args:
            components: PCA components array
            feature_names: List of feature names
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap of loadings
        im = ax.imshow(components, cmap="RdBu_r", aspect="auto")

        # Set labels
        ax.set_xticks(range(components.shape[1]))
        ax.set_xticklabels([f"PC{i + 1}" for i in range(components.shape[1])])
        ax.set_yticks(range(components.shape[0]))
        ax.set_yticklabels(feature_names)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Features")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Loading Value")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_performance_distribution(
        self,
        performance_matrix: pd.DataFrame,
        title: str = "Performance Distribution by Task",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create a box plot of performance distributions.

        Args:
            performance_matrix: DataFrame with performance values
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create box plot
        performance_matrix.boxplot(ax=ax)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance Score")
        ax.set_xlabel("Tasks")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_correlation_distribution(
        self, correlation_matrix: pd.DataFrame, title: str = "Correlation Distribution", save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a histogram of correlation values.

        Args:
            correlation_matrix: DataFrame with correlation values
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get upper triangle values (excluding diagonal)
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]

        # Create histogram
        ax.hist(corr_values, bins=20, alpha=0.7, edgecolor="black", color="skyblue")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add mean line
        mean_corr = float(np.mean(corr_values))
        ax.axvline(mean_corr, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_corr:.3f}")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_comprehensive_plot(
        self,
        performance_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        pca_results: Dict,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Create a comprehensive 2x2 subplot with all key visualizations.

        Args:
            performance_matrix: DataFrame with performance values
            correlation_matrix: DataFrame with correlation values
            pca_results: Dictionary with PCA results
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0, square=True, fmt=".2f", ax=axes[0, 0])
        axes[0, 0].set_title("Task Performance Correlations", fontsize=14, fontweight="bold")

        # 2. Performance distribution
        performance_matrix.boxplot(ax=axes[0, 1])
        axes[0, 1].set_title("Performance Distribution by Task", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel("Performance Score")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Performance heatmap
        sns.heatmap(performance_matrix, cmap="viridis", ax=axes[1, 0], cbar_kws={"label": "Performance"})
        axes[1, 0].set_title("Policy Performance Heatmap", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Tasks")
        axes[1, 0].set_ylabel("Policies")

        # 4. PCA scree plot
        explained_variance = pca_results.get("explained_variance_ratio", [])
        if explained_variance:
            n_components = len(explained_variance)
            axes[1, 1].plot(range(1, n_components + 1), explained_variance, "bo-")
            axes[1, 1].set_xlabel("Principal Component")
            axes[1, 1].set_ylabel("Explained Variance Ratio")
            axes[1, 1].set_title("PCA Scree Plot", fontsize=14, fontweight="bold")
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def save_all_visualizations(
        self,
        performance_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        pca_results: Dict,
        output_dir: str = "../visualizations/",
    ) -> None:
        """
        Save all visualizations to the specified directory.

        Args:
            performance_matrix: DataFrame with performance values
            correlation_matrix: DataFrame with correlation values
            pca_results: Dictionary with PCA results
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save individual plots
        self.plot_correlation_heatmap(correlation_matrix, save_path=f"{output_dir}correlation_heatmap.png")

        self.plot_performance_matrix(performance_matrix, save_path=f"{output_dir}policy_performance.png")

        if "explained_variance_ratio" in pca_results:
            self.plot_pca_scree(pca_results["explained_variance_ratio"], save_path=f"{output_dir}pca_scree_plot.png")

        if "components" in pca_results:
            self.plot_pca_loadings(
                pca_results["components"], list(performance_matrix.columns), save_path=f"{output_dir}pca_loadings.png"
            )

        self.plot_performance_distribution(performance_matrix, save_path=f"{output_dir}performance_distribution.png")

        self.plot_correlation_distribution(correlation_matrix, save_path=f"{output_dir}correlation_distribution.png")

        # Save comprehensive plot
        self.create_comprehensive_plot(
            performance_matrix, correlation_matrix, pca_results, save_path=f"{output_dir}comprehensive_analysis.png"
        )

        print(f"✅ All visualizations saved to {output_dir}")

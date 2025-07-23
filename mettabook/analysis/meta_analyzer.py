"""
Meta analyzer for policy performance data using PCA and statistical analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MetaAnalyzer:
    """Meta-analyzer for policy performance data using PCA and statistical methods."""

    def __init__(self):
        self.pca = None
        self.scaler = None

    def perform_pca(self, performance_matrix: pd.DataFrame, n_components: int = None) -> Dict[str, Any]:
        """Perform PCA on performance matrix."""
        # Handle missing values
        performance_matrix_filled = performance_matrix.fillna(performance_matrix.mean())

        # Standardize the data
        self.scaler = StandardScaler()
        performance_matrix_scaled = self.scaler.fit_transform(performance_matrix_filled)

        # Perform PCA
        if n_components is None:
            n_components = min(performance_matrix.shape[0], performance_matrix.shape[1])

        self.pca = PCA(n_components=n_components)
        pca_scores = self.pca.fit_transform(performance_matrix_scaled)

        return {
            "components": self.pca.components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "explained_variance": self.pca.explained_variance_,
            "singular_values": self.pca.singular_values_,
            "pca_scores": pca_scores,
            "feature_names": performance_matrix.columns.tolist(),
            "sample_names": performance_matrix.index.tolist(),
        }

    def get_optimal_components(self, performance_matrix: pd.DataFrame, threshold: float = 0.95) -> int:
        """Find optimal number of components to explain threshold of variance."""
        # Handle missing values
        performance_matrix_filled = performance_matrix.fillna(performance_matrix.mean())

        # Standardize the data
        scaler = StandardScaler()
        performance_matrix_scaled = scaler.fit_transform(performance_matrix_filled)

        # Perform PCA with maximum components
        max_components = min(performance_matrix.shape[0], performance_matrix.shape[1])
        pca = PCA(n_components=max_components)
        pca.fit(performance_matrix_scaled)

        # Find number of components needed to explain threshold of variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_components = np.argmax(cumulative_variance >= threshold) + 1

        return optimal_components

    def generate_summary_statistics(
        self, performance_matrix: pd.DataFrame, correlation_matrix: pd.DataFrame, pca_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        # Basic performance statistics
        performance_stats = {
            "mean_performance": float(performance_matrix.mean().mean()),
            "std_performance": float(performance_matrix.std().mean()),
            "min_performance": float(performance_matrix.min().min()),
            "max_performance": float(performance_matrix.max().max()),
            "total_policies": len(performance_matrix.index),
            "total_tasks": len(performance_matrix.columns),
            "missing_data_percentage": float(
                (
                    performance_matrix.isnull().sum().sum()
                    / (len(performance_matrix.index) * len(performance_matrix.columns))
                )
                * 100
            ),
        }

        # Correlation statistics
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        correlation_stats = {
            "mean_correlation": float(np.mean(corr_values)),
            "std_correlation": float(np.std(corr_values)),
            "min_correlation": float(np.min(corr_values)),
            "max_correlation": float(np.max(corr_values)),
            "correlation_count": len(corr_values),
        }

        # PCA statistics
        pca_stats = {
            "total_variance_explained": float(np.sum(pca_results["explained_variance_ratio"])),
            "first_component_variance": float(pca_results["explained_variance_ratio"][0]),
            "second_component_variance": float(pca_results["explained_variance_ratio"][1])
            if len(pca_results["explained_variance_ratio"]) > 1
            else 0.0,
            "components_95_percent": self.get_optimal_components(performance_matrix, 0.95),
            "components_90_percent": self.get_optimal_components(performance_matrix, 0.90),
            "components_80_percent": self.get_optimal_components(performance_matrix, 0.80),
        }

        # Task-specific statistics
        task_stats = {}
        for task in performance_matrix.columns:
            task_data = performance_matrix[task].dropna()
            if len(task_data) > 0:
                task_stats[task] = {
                    "mean": float(task_data.mean()),
                    "std": float(task_data.std()),
                    "min": float(task_data.min()),
                    "max": float(task_data.max()),
                    "count": len(task_data),
                }

        # Policy-specific statistics
        policy_stats = {}
        for policy in performance_matrix.index:
            policy_data = performance_matrix.loc[policy].dropna()
            if len(policy_data) > 0:
                policy_stats[policy] = {
                    "mean": float(policy_data.mean()),
                    "std": float(policy_data.std()),
                    "min": float(policy_data.min()),
                    "max": float(policy_data.max()),
                    "count": len(policy_data),
                }

        return {
            "performance": performance_stats,
            "correlation": correlation_stats,
            "pca": pca_stats,
            "tasks": task_stats,
            "policies": policy_stats,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
        }

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save analysis results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {output_path}")

    def load_results(self, input_file: str) -> Dict[str, Any]:
        """Load analysis results from JSON file."""
        with open(input_file, "r") as f:
            return json.load(f)

    def get_task_importance(self, pca_results: Dict[str, Any], n_components: int = 3) -> Dict[str, float]:
        """Calculate task importance based on PCA loadings."""
        if "components" not in pca_results or "feature_names" not in pca_results:
            raise ValueError("PCA results must contain 'components' and 'feature_names'")

        components = pca_results["components"][:n_components]
        feature_names = pca_results["feature_names"]

        # Calculate importance as sum of squared loadings across components
        importance = {}
        for i, feature in enumerate(feature_names):
            importance[feature] = float(np.sum(components[:, i] ** 2))

        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance

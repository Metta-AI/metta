#!/usr/bin/env -S uv run
"""
Factor analysis tool for policy performance data.

Usage:
    ./tools/factor_analysis.py ++performance_matrix=analysis_results/comprehensive_performance_matrix.csv
    ++output_dir=factor_analysis_results ++enable_wandb=true
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from metta.common.util.logging import setup_mettagrid_logger


class PolicyFactorAnalyzer:
    """Factor analysis for policy performance data using EM and cross-validation."""

    def __init__(self, output_dir: Path, enable_wandb: bool = False):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_wandb = enable_wandb
        self.wandb_run = None

        # Set random seed for reproducibility
        np.random.seed(42)

        # Analysis parameters
        self.max_factors = 20
        self.n_folds = 5
        self.n_repeats = 10

    def init_wandb(self, run_name: str) -> None:
        """Initialize wandb run for logging."""
        if self.enable_wandb:
            self.wandb_run = wandb.init(
                project="metta-analysis",
                name=run_name,
                entity="metta-research",
                config={
                    "analysis_type": "factor_analysis",
                    "max_factors": self.max_factors,
                    "n_folds": self.n_folds,
                    "n_repeats": self.n_repeats,
                },
            )
            self.logger.info(f"Initialized wandb run: {self.wandb_run.name}")

    def log_cv_results(self, cv_results: Dict) -> None:
        """Log cross-validation results to wandb."""
        if self.enable_wandb and self.wandb_run:
            # Log cross-validation metrics
            for i, n_factors in enumerate(cv_results["n_factors"]):
                self.wandb_run.log(
                    {
                        "cv/n_factors": n_factors,
                        "cv/mean_mse": cv_results["mean_mse"][i],
                        "cv/std_mse": cv_results["std_mse"][i],
                    }
                )

    def log_em_results(self, em_results: Dict, n_factors: int) -> None:
        """Log EM factor analysis results to wandb."""
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.log(
                {
                    "em/n_factors": n_factors,
                    "em/explained_variance_ratio": em_results["explained_variance_ratio"],
                    "em/noise_variance_mean": np.mean(em_results["noise_variance"]),
                }
            )

    def log_clustering_results(self, clustering: Dict) -> None:
        """Log clustering results to wandb."""
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.log(
                {
                    "clustering/n_clusters": len(set(clustering["cluster_labels"])),
                    "clustering/silhouette_score": clustering.get("silhouette_score", 0),
                }
            )

    def finish_wandb(self) -> None:
        """Finish wandb run."""
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.finish()
            self.logger.info("Finished wandb run")

    def load_performance_matrix(self, matrix_file: Path) -> pd.DataFrame:
        """Load and preprocess performance matrix, then compute environment correlation matrix."""
        self.logger.info(f"Loading performance matrix from {matrix_file}")

        # Load matrix
        performance_matrix = pd.read_csv(matrix_file, index_col=0)

        # Handle missing values
        performance_matrix = performance_matrix.fillna(performance_matrix.mean())

        # Standardize the data
        scaler = StandardScaler()
        performance_matrix_scaled = pd.DataFrame(
            scaler.fit_transform(performance_matrix), index=performance_matrix.index, columns=performance_matrix.columns
        )

        self.logger.info(f"Performance matrix shape: {performance_matrix.shape}")
        self.logger.info(f"Policies: {len(performance_matrix.index)}")
        self.logger.info(f"Environments: {len(performance_matrix.columns)}")

        # Compute environment × environment correlation matrix
        env_correlation_matrix = performance_matrix_scaled.corr()

        self.logger.info(f"Environment correlation matrix shape: {env_correlation_matrix.shape}")
        self.logger.info(f"Correlation range: [{env_correlation_matrix.values.min():.3f}, {env_correlation_matrix.values.max():.3f}]")

        return env_correlation_matrix

    def cross_validate_dimensionality(self, data: np.ndarray) -> Dict:
        """
        Use K-fold cross-validation to determine optimal number of factors.

        Args:
            data: Standardized performance matrix

        Returns:
            Dictionary with cross-validation results
        """
        self.logger.info("Starting cross-validation for dimensionality selection...")

        cv_results = {"n_factors": [], "mean_mse": [], "std_mse": [], "fold_results": []}

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for n_factors in range(1, min(self.max_factors + 1, data.shape[1] + 1)):
            self.logger.info(f"Testing {n_factors} factors...")

            fold_mses = []

            for _fold, (train_idx, test_idx) in enumerate(kf.split(data)):
                X_train, X_test = data[train_idx], data[test_idx]

                # Fit Factor Analysis model
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                fa.fit(X_train)

                # Transform test data and reconstruct
                X_test_transformed = fa.transform(X_test)
                X_test_reconstructed = fa.inverse_transform(X_test_transformed)

                # Calculate reconstruction error
                mse = mean_squared_error(X_test, X_test_reconstructed)
                fold_mses.append(mse)

            # Store results
            cv_results["n_factors"].append(n_factors)
            cv_results["mean_mse"].append(np.mean(fold_mses))
            cv_results["std_mse"].append(np.std(fold_mses))
            cv_results["fold_results"].append(fold_mses)

            self.logger.info(f"  {n_factors} factors: MSE = {np.mean(fold_mses):.4f} ± {np.std(fold_mses):.4f}")

        return cv_results

    def expectation_maximization_analysis(self, data: np.ndarray, n_factors: int) -> Dict:
        """
        Perform Expectation Maximization factor analysis.

        Args:
            data: Standardized performance matrix
            n_factors: Number of factors to extract

        Returns:
            Dictionary with EM factor analysis results
        """
        self.logger.info(f"Performing EM factor analysis with {n_factors} factors...")

        # Fit Factor Analysis model
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        fa.fit(data)

        # Extract results
        factors = fa.components_  # Factor loadings (factors × variables)
        factor_scores = fa.transform(data)  # Factor scores (samples × factors)
        noise_variance = fa.noise_variance_  # Noise variance for each variable

        # Calculate explained variance
        total_variance = np.var(data, axis=0).sum()
        explained_variance = total_variance - noise_variance.sum()
        explained_variance_ratio = explained_variance / total_variance

        return {
            "factors": factors,
            "factor_scores": factor_scores,
            "noise_variance": noise_variance,
            "explained_variance_ratio": explained_variance_ratio,
            "model": fa,
        }

    def analyze_factor_structure(self, factors: np.ndarray, env_names: List[str]) -> Dict:
        """
        Analyze the structure of extracted factors from environment correlation matrix.

        Args:
            factors: Factor loadings matrix (factors × environments)
            env_names: Names of environments

        Returns:
            Dictionary with factor structure analysis and environment representations
        """
        self.logger.info("Analyzing environment factor structure...")

        factor_analysis = {
            "factor_loadings": {},
            "factor_interpretations": [],
            "environment_representations": {},
            "environment_clusters": {}
        }

        # Analyze each factor
        for i, factor in enumerate(factors):
            # Get top loadings for this factor
            abs_loadings = np.abs(factor)
            top_indices = np.argsort(abs_loadings)[::-1][:10]  # Top 10 loadings

            factor_loadings = {}
            for idx in top_indices:
                factor_loadings[env_names[idx]] = factor[idx]

            factor_analysis["factor_loadings"][f"Factor_{i + 1}"] = factor_loadings

            # Interpret factor based on high loadings
            high_loadings = [(env_names[idx], factor[idx]) for idx in top_indices if abs_loadings[idx] > 0.3]

            if high_loadings:
                # Group by environment category
                categories = {}
                for env_name, loading in high_loadings:
                    category = env_name.split("/")[0] if "/" in env_name else env_name
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((env_name, loading))

                interpretation = {
                    "factor_id": f"Factor_{i + 1}",
                    "high_loadings": high_loadings,
                    "categories": categories,
                    "variance_explained": np.var(factor),
                    "factor_strength": np.linalg.norm(factor),
                }
                factor_analysis["factor_interpretations"].append(interpretation)

        # Create environment representation vectors
        for j, env_name in enumerate(env_names):
            env_representation = factors[:, j]  # Column j contains loadings for environment j
            factor_analysis["environment_representations"][env_name] = {
                "factor_loadings": env_representation.tolist(),
                "total_strength": np.linalg.norm(env_representation),
                "dominant_factors": [(i, abs(env_representation[i])) for i in np.argsort(np.abs(env_representation))[-3:][::-1]]
            }

        return factor_analysis

    def cluster_policies(self, factor_scores: np.ndarray, policy_names: List[str]) -> Dict:
        """
        Cluster policies based on their factor scores.

        Args:
            factor_scores: Factor scores matrix
            policy_names: Names of policies

        Returns:
            Dictionary with clustering results
        """
        self.logger.info("Clustering policies based on factor scores...")

        # Hierarchical clustering
        linkage_matrix = linkage(factor_scores, method="ward")

        # Determine optimal number of clusters using elbow method
        distortions = []
        K_range = range(2, min(11, len(policy_names)))

        for k in K_range:
            cluster_labels = fcluster(linkage_matrix, k, criterion="maxclust")
            # Calculate within-cluster sum of squares
            wcss = 0
            for i in range(k):
                cluster_points = factor_scores[cluster_labels == i + 1]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    wcss += np.sum((cluster_points - centroid) ** 2)
            distortions.append(wcss)

        # Choose optimal number of clusters (elbow method)
        optimal_k = self._find_elbow(K_range, distortions)

        # Get final cluster labels
        cluster_labels = fcluster(linkage_matrix, optimal_k, criterion="maxclust")

        # Group policies by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = f"Cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(policy_names[i])

        return {
            "linkage_matrix": linkage_matrix,
            "optimal_clusters": optimal_k,
            "cluster_labels": cluster_labels,
            "clusters": clusters,
            "distortions": distortions,
            "k_range": list(K_range),
        }

    def _find_elbow(self, x: List, y: List) -> int:
        """Find elbow point in curve using second derivative method."""
        if len(x) < 3:
            return x[-1] if x else 2

        # Calculate second derivative
        d2y = np.diff(np.diff(y))

        # Find point with maximum second derivative
        elbow_idx = np.argmax(np.abs(d2y)) + 1
        return x[elbow_idx]

    def create_visualizations(
        self, data: pd.DataFrame, cv_results: Dict, em_results: Dict, factor_analysis: Dict, clustering: Dict
    ) -> None:
        """Create comprehensive visualizations."""
        self.logger.info("Creating visualizations...")

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Cross-validation results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(cv_results["n_factors"], cv_results["mean_mse"], yerr=cv_results["std_mse"], marker="o", capsize=5)
        ax.set_xlabel("Number of Factors")
        ax.set_ylabel("Mean Squared Error")
        ax.set_title("Cross-Validation Results for Dimensionality Selection")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "cross_validation_results.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Factor loadings heatmap
        if "factors" in em_results:
            factors = em_results["factors"]
            eval_names = data.columns.tolist()

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                factors,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                xticklabels=eval_names,
                yticklabels=[f"Factor_{i + 1}" for i in range(len(factors))],
                center=0,
                ax=ax,
            )
            ax.set_title("Factor Loadings Matrix")
            ax.set_xlabel("Evaluation Environments")
            ax.set_ylabel("Factors")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(self.output_dir / "factor_loadings_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 3. Factor scores scatter plot (first two factors)
        if "factor_scores" in em_results:
            factor_scores = em_results["factor_scores"]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(
                factor_scores[:, 0], factor_scores[:, 1], c=clustering["cluster_labels"], cmap="tab10", alpha=0.7
            )
            ax.set_xlabel("Factor 1")
            ax.set_ylabel("Factor 2")
            ax.set_title("Policy Factor Scores (First Two Factors)")
            ax.grid(True, alpha=0.3)

            # Add policy labels for some points
            for i, policy_name in enumerate(data.index):
                if i % 5 == 0:  # Label every 5th policy
                    ax.annotate(
                        policy_name.split(".")[-1][:10],
                        (factor_scores[i, 0], factor_scores[i, 1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

            plt.tight_layout()
            plt.savefig(self.output_dir / "factor_scores_scatter.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 4. Dendrogram for hierarchical clustering
        if "linkage_matrix" in clustering:
            fig, ax = plt.subplots(figsize=(12, 8))
            dendrogram(clustering["linkage_matrix"], labels=data.index.tolist(), leaf_rotation=90, leaf_font_size=8)
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Policies")
            ax.set_ylabel("Distance")
            plt.tight_layout()
            plt.savefig(self.output_dir / "clustering_dendrogram.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 5. Explained variance plot
        if "factors" in em_results:
            factors = em_results["factors"]
            explained_var = np.var(factors, axis=1)
            cumulative_var = np.cumsum(explained_var) / explained_var.sum()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Individual explained variance
            ax1.bar(range(1, len(explained_var) + 1), explained_var)
            ax1.set_xlabel("Factor")
            ax1.set_ylabel("Explained Variance")
            ax1.set_title("Individual Factor Explained Variance")
            ax1.grid(True, alpha=0.3)

            # Cumulative explained variance
            ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker="o")
            ax2.axhline(y=0.8, color="r", linestyle="--", alpha=0.7, label="80% threshold")
            ax2.set_xlabel("Number of Factors")
            ax2.set_ylabel("Cumulative Explained Variance")
            ax2.set_title("Cumulative Explained Variance")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "explained_variance.png", dpi=300, bbox_inches="tight")
            plt.close()

    def save_results(self, cv_results: Dict, em_results: Dict, factor_analysis: Dict, clustering: Dict) -> None:
        """Save all analysis results to files."""

        # Save cross-validation results
        cv_file = self.output_dir / "cross_validation_results.json"
        cv_data = {
            "n_factors": cv_results["n_factors"],
            "mean_mse": [float(x) for x in cv_results["mean_mse"]],
            "std_mse": [float(x) for x in cv_results["std_mse"]],
            "optimal_n_factors": cv_results["n_factors"][np.argmin(cv_results["mean_mse"])],
        }
        with open(cv_file, "w") as f:
            json.dump(cv_data, f, indent=2)

        # Save factor analysis results
        fa_file = self.output_dir / "factor_analysis_results.json"
        fa_data = {
            "explained_variance_ratio": float(em_results["explained_variance_ratio"]),
            "factor_interpretations": factor_analysis["factor_interpretations"],
            "environment_representations": factor_analysis["environment_representations"],
            "clusters": clustering["clusters"],
            "optimal_clusters": clustering["optimal_clusters"],
        }
        with open(fa_file, "w") as f:
            json.dump(fa_data, f, indent=2, default=str)

        # Save factor loadings as CSV
        if "factors" in em_results:
            loadings_df = pd.DataFrame(
                em_results["factors"], columns=[f"Factor_{i + 1}" for i in range(len(em_results["factors"]))]
            )
            loadings_file = self.output_dir / "factor_loadings.csv"
            loadings_df.to_csv(loadings_file)

        # Save factor scores as CSV
        if "factor_scores" in em_results:
            scores_df = pd.DataFrame(
                em_results["factor_scores"],
                columns=[f"Factor_{i + 1}" for i in range(em_results["factor_scores"].shape[1])],
            )
            scores_file = self.output_dir / "factor_scores.csv"
            scores_df.to_csv(scores_file)

        # Save clustering results
        cluster_file = self.output_dir / "clustering_results.json"
        cluster_data = {
            "optimal_clusters": clustering["optimal_clusters"],
            "clusters": clustering["clusters"],
            "cluster_labels": clustering["cluster_labels"].tolist(),
        }
        with open(cluster_file, "w") as f:
            json.dump(cluster_data, f, indent=2)

        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info(f"  - Cross-validation: {cv_file}")
        self.logger.info(f"  - Factor analysis: {fa_file}")
        self.logger.info(f"  - Factor loadings: {loadings_file}")
        self.logger.info(f"  - Factor scores: {scores_file}")
        self.logger.info(f"  - Clustering: {cluster_file}")


def main():
    parser = argparse.ArgumentParser(description="Perform factor analysis on environment correlation matrix")
    parser.add_argument(
        "--performance-matrix",
        type=Path,
        required=True,
        help="CSV file with performance matrix (policies × environments) - will compute environment correlation matrix",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("factor_analysis_results"), help="Directory to save analysis results"
    )
    parser.add_argument("--max-factors", type=int, default=20, help="Maximum number of factors to test")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_mettagrid_logger("factor_analysis")
    logger.info("Starting factor analysis")

    # Create analyzer
    analyzer = PolicyFactorAnalyzer(args.output_dir, args.enable_wandb)
    analyzer.max_factors = args.max_factors
    analyzer.n_folds = args.n_folds

    try:
        # Initialize wandb if enabled
        if args.enable_wandb:
            run_id = str(int(time.time()))
            run_name = f"msb_compEval_{run_id}"
            analyzer.init_wandb(run_name)

        # Load performance matrix and compute environment correlation matrix
        env_correlation_matrix = analyzer.load_performance_matrix(args.performance_matrix)

        # Convert to numpy array
        data_array = env_correlation_matrix.values

        # Cross-validation for dimensionality selection
        cv_results = analyzer.cross_validate_dimensionality(data_array)

        # Find optimal number of factors
        optimal_n_factors = cv_results["n_factors"][np.argmin(cv_results["mean_mse"])]
        logger.info(f"Optimal number of factors: {optimal_n_factors}")

        # Perform EM factor analysis
        em_results = analyzer.expectation_maximization_analysis(data_array, optimal_n_factors)

        # Analyze factor structure
        factor_analysis = analyzer.analyze_factor_structure(em_results["factors"], env_correlation_matrix.columns.tolist())

        # Cluster environments (not policies)
        clustering = analyzer.cluster_policies(em_results["factor_scores"], env_correlation_matrix.index.tolist())

        # Create visualizations
        analyzer.create_visualizations(env_correlation_matrix, cv_results, em_results, factor_analysis, clustering)

        # Save results
        analyzer.save_results(cv_results, em_results, factor_analysis, clustering)

        # Print summary
        print("\nEnvironment Factor Analysis Summary:")
        print(f"  Optimal factors: {optimal_n_factors}")
        print(f"  Explained variance: {em_results['explained_variance_ratio']:.3f}")
        print(f"  Optimal clusters: {clustering['optimal_clusters']}")
        print(f"  Total environments: {len(env_correlation_matrix.index)}")
        print(f"  Environment correlation matrix shape: {env_correlation_matrix.shape}")

        # Log results to wandb
        analyzer.log_cv_results(cv_results)
        analyzer.log_em_results(em_results, optimal_n_factors)
        analyzer.log_clustering_results(clustering)
        analyzer.finish_wandb()

        return 0

    except Exception as e:
        logger.error(f"Factor analysis failed: {e}")
        analyzer.finish_wandb()  # Ensure wandb is finished even on error
        return 1


if __name__ == "__main__":
    exit(main())

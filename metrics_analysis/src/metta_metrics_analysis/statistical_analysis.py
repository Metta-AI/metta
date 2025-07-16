"""
Statistical Analysis Module for Metta Metrics Analysis.

This module provides statistical methods for analyzing performance metrics,
including IQM, performance profiles, and bootstrap confidence intervals.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Perform statistical analysis on processed run data."""

    def __init__(self, data: pd.DataFrame, seed: int = 42):
        """
        Initialize the statistical analyzer.

        Args:
            data: Processed DataFrame from DataProcessor
            seed: Random seed for reproducibility
        """
        self.df = data.copy()
        self.seed = seed
        np.random.seed(seed)

        # Identify grouping columns
        self.group_columns = [
            col
            for col in self.df.columns
            if col in ["run_id", "run_name", "group", "algorithm", "method"] or col.startswith("config.")
        ]

    def compute_iqm(
        self, metric: str, group_by: str | None = None, trim_percentage: float = 0.25
    ) -> float | pd.DataFrame:
        """
        Compute Interquartile Mean (IQM) for a metric.

        IQM is the mean of the middle 50% of values, which is more robust
        to outliers than the standard mean.

        Args:
            metric: Metric to compute IQM for
            group_by: Optional column to group by
            trim_percentage: Percentage to trim from each end (default 0.25 for IQM)

        Returns:
            IQM value(s) as float or DataFrame
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        def calculate_iqm(values):
            """Calculate IQM for a series of values."""
            values = values.dropna()
            if len(values) == 0:
                return np.nan

            # Sort values
            sorted_values = np.sort(values)
            n = len(sorted_values)

            # Calculate trim indices
            lower_idx = int(n * trim_percentage)
            upper_idx = int(n * (1 - trim_percentage))

            # Handle edge cases
            if lower_idx >= upper_idx:
                return np.mean(sorted_values)

            # Return mean of middle values
            return np.mean(sorted_values[lower_idx:upper_idx])

        if group_by is None:
            # Calculate single IQM
            return calculate_iqm(self.df[metric])
        else:
            # Calculate IQM by group
            result = self.df.groupby(group_by)[metric].apply(calculate_iqm)
            return result.to_frame(name=f"{metric}_iqm")

    def compute_iqm_with_ci(
        self,
        metric: str,
        group_by: str | None = None,
        stratify_by: str | None = None,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        trim_percentage: float = 0.25,
    ) -> pd.DataFrame:
        """
        Compute IQM with bootstrap confidence intervals.

        Args:
            metric: Metric to analyze
            group_by: Column to group by (e.g., "algorithm")
            stratify_by: Column to stratify bootstrap by (e.g., "task")
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples
            trim_percentage: Percentage to trim for IQM

        Returns:
            DataFrame with IQM and confidence intervals
        """
        results = []

        # Get unique groups
        if group_by is None:
            groups = [("all", self.df)]
        else:
            groups = list(self.df.groupby(group_by))

        for group_name, group_data in tqdm(groups, desc="Computing IQM with CI"):
            # Get values for this group
            values = group_data[metric].dropna().values

            if len(values) == 0:
                continue

            # Compute point estimate
            iqm_point = self._compute_iqm_single(values, trim_percentage)

            # Bootstrap for confidence intervals
            if stratify_by and stratify_by in group_data.columns:
                # Stratified bootstrap
                bootstrap_iqms = self._stratified_bootstrap_iqm(
                    group_data, metric, stratify_by, n_bootstrap, trim_percentage
                )
            else:
                # Regular bootstrap
                bootstrap_iqms = []
                for _ in range(n_bootstrap):
                    boot_sample = np.random.choice(values, size=len(values), replace=True)
                    boot_iqm = self._compute_iqm_single(boot_sample, trim_percentage)
                    bootstrap_iqms.append(boot_iqm)

            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_iqms, lower_percentile)
            ci_upper = np.percentile(bootstrap_iqms, upper_percentile)

            results.append(
                {
                    group_by: group_name if group_by else "all",
                    f"{metric}_iqm": iqm_point,
                    f"{metric}_ci_lower": ci_lower,
                    f"{metric}_ci_upper": ci_upper,
                    f"{metric}_ci_width": ci_upper - ci_lower,
                    "n_samples": len(values),
                }
            )

        return pd.DataFrame(results)

    def _compute_iqm_single(self, values: np.ndarray, trim_percentage: float) -> float:
        """Compute IQM for a single array of values."""
        if len(values) == 0:
            return np.nan

        sorted_values = np.sort(values)
        n = len(sorted_values)

        lower_idx = int(n * trim_percentage)
        upper_idx = int(n * (1 - trim_percentage))

        if lower_idx >= upper_idx:
            return np.mean(sorted_values)

        return np.mean(sorted_values[lower_idx:upper_idx])

    def _stratified_bootstrap_iqm(
        self, data: pd.DataFrame, metric: str, stratify_by: str, n_bootstrap: int, trim_percentage: float
    ) -> list[float]:
        """Perform stratified bootstrap for IQM calculation."""
        bootstrap_iqms = []

        # Get strata
        strata = data.groupby(stratify_by)[metric].apply(lambda x: x.dropna().values)
        strata_sizes = {k: len(v) for k, v in strata.items()}

        for _ in range(n_bootstrap):
            # Sample within each stratum
            bootstrap_values = []

            for stratum_name, stratum_values in strata.items():
                if len(stratum_values) > 0:
                    # Sample with replacement within stratum
                    stratum_sample = np.random.choice(stratum_values, size=len(stratum_values), replace=True)
                    bootstrap_values.extend(stratum_sample)

            # Compute IQM on bootstrap sample
            if bootstrap_values:
                boot_iqm = self._compute_iqm_single(np.array(bootstrap_values), trim_percentage)
                bootstrap_iqms.append(boot_iqm)

        return bootstrap_iqms

    def compute_performance_profiles(
        self,
        metric: str,
        group_by: str,
        task_column: str | None = None,
        thresholds: np.ndarray | None = None,
        higher_is_better: bool = True,
    ) -> pd.DataFrame:
        """
        Compute performance profiles for comparing algorithms.

        Performance profiles show the probability that an algorithm achieves
        a certain performance threshold across tasks.

        Args:
            metric: Metric to analyze
            group_by: Column identifying algorithms/methods
            task_column: Column identifying tasks (if None, treats each run as a task)
            thresholds: Performance thresholds (if None, auto-generated)
            higher_is_better: Whether higher metric values are better

        Returns:
            DataFrame with performance profile data
        """
        # Prepare data
        if task_column is None:
            # Treat each run as a separate task
            task_column = "run_id"

        # Aggregate by algorithm and task
        agg_data = self.df.groupby([group_by, task_column])[metric].mean().reset_index()

        # Pivot to have algorithms as columns
        pivot_data = agg_data.pivot(index=task_column, columns=group_by, values=metric)

        # Compute performance ratios
        if higher_is_better:
            best_per_task = pivot_data.max(axis=1)
            ratios = pivot_data.div(best_per_task, axis=0)
        else:
            best_per_task = pivot_data.min(axis=1)
            ratios = best_per_task.div(pivot_data, axis=1)

        # Generate thresholds if not provided
        if thresholds is None:
            thresholds = np.linspace(0, 2, 201)  # 0 to 2 in steps of 0.01

        # Compute profiles
        profiles = {}
        for algorithm in ratios.columns:
            profile = []
            for threshold in thresholds:
                # Probability of being within threshold of best
                prob = (ratios[algorithm] >= (1 / threshold)).mean()
                profile.append(prob)
            profiles[algorithm] = profile

        # Create result DataFrame
        result_df = pd.DataFrame(profiles, index=thresholds)
        result_df.index.name = "threshold"

        return result_df

    def compute_optimality_gaps(
        self,
        metric: str,
        optimal_scores: dict[str, float],
        task_column: str,
        group_by: str | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute optimality gaps relative to known optimal scores.

        Args:
            metric: Metric to analyze
            optimal_scores: Dict mapping task names to optimal scores
            task_column: Column identifying tasks
            group_by: Optional column to group by
            normalize: Whether to normalize gaps (as percentage of optimal)

        Returns:
            DataFrame with optimality gaps
        """
        results = []

        # Filter to tasks with known optimal scores
        known_tasks = self.df[self.df[task_column].isin(optimal_scores.keys())].copy()

        if len(known_tasks) == 0:
            logger.warning("No tasks found with known optimal scores")
            return pd.DataFrame()

        # Add optimal score column
        known_tasks["optimal_score"] = known_tasks[task_column].map(optimal_scores)

        # Compute gaps
        known_tasks["optimality_gap"] = known_tasks["optimal_score"] - known_tasks[metric]

        if normalize:
            # Normalize by optimal score (avoid division by zero)
            known_tasks["optimality_gap_pct"] = np.where(
                known_tasks["optimal_score"] != 0,
                (known_tasks["optimality_gap"] / np.abs(known_tasks["optimal_score"])) * 100,
                0,
            )
            gap_column = "optimality_gap_pct"
        else:
            gap_column = "optimality_gap"

        # Aggregate results
        if group_by is None:
            # Overall statistics
            results.append(
                {
                    "group": "all",
                    f"mean_{gap_column}": known_tasks[gap_column].mean(),
                    f"std_{gap_column}": known_tasks[gap_column].std(),
                    f"median_{gap_column}": known_tasks[gap_column].median(),
                    f"max_{gap_column}": known_tasks[gap_column].max(),
                    "n_tasks": known_tasks[task_column].nunique(),
                }
            )
        else:
            # Statistics by group
            for group_name, group_data in known_tasks.groupby(group_by):
                results.append(
                    {
                        group_by: group_name,
                        f"mean_{gap_column}": group_data[gap_column].mean(),
                        f"std_{gap_column}": group_data[gap_column].std(),
                        f"median_{gap_column}": group_data[gap_column].median(),
                        f"max_{gap_column}": group_data[gap_column].max(),
                        "n_tasks": group_data[task_column].nunique(),
                    }
                )

        return pd.DataFrame(results)

    def compare_algorithms(
        self,
        metric: str,
        group_by: str,
        test: str = "wilcoxon",
        correction: str | None = "bonferroni",
        task_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Perform pairwise statistical comparisons between algorithms.

        Args:
            metric: Metric to compare
            group_by: Column identifying algorithms
            test: Statistical test ("wilcoxon", "ttest", "mannwhitney")
            correction: Multiple comparison correction ("bonferroni", "fdr", None)
            task_column: Column for paired tests (if applicable)

        Returns:
            DataFrame with pairwise comparison results
        """
        from itertools import combinations

        from statsmodels.stats.multitest import multipletests

        # Get unique algorithms
        algorithms = self.df[group_by].unique()

        if len(algorithms) < 2:
            logger.warning("Need at least 2 algorithms to compare")
            return pd.DataFrame()

        results = []

        # Perform pairwise comparisons
        for alg1, alg2 in combinations(algorithms, 2):
            data1 = self.df[self.df[group_by] == alg1][metric].dropna()
            data2 = self.df[self.df[group_by] == alg2][metric].dropna()

            if len(data1) == 0 or len(data2) == 0:
                continue

            # Perform statistical test
            if test == "wilcoxon" and task_column:
                # Paired test - need matching tasks
                paired_data = self._get_paired_data(alg1, alg2, metric, group_by, task_column)
                if len(paired_data) > 0:
                    statistic, p_value = stats.wilcoxon(paired_data[f"{metric}_1"], paired_data[f"{metric}_2"])
                else:
                    continue
            elif test == "mannwhitney":
                statistic, p_value = stats.mannwhitneyu(data1, data2)
            elif test == "ttest":
                statistic, p_value = stats.ttest_ind(data1, data2)
            else:
                raise ValueError(f"Unknown test: {test}")

            # Effect size (Cohen's d)
            effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.std() ** 2 + data2.std() ** 2) / 2)

            results.append(
                {
                    "algorithm_1": alg1,
                    "algorithm_2": alg2,
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "mean_diff": data1.mean() - data2.mean(),
                    "n_1": len(data1),
                    "n_2": len(data2),
                }
            )

        results_df = pd.DataFrame(results)

        # Apply multiple comparison correction
        if correction and len(results_df) > 0:
            if correction == "bonferroni":
                results_df["p_adjusted"] = results_df["p_value"] * len(results_df)
                results_df["p_adjusted"] = results_df["p_adjusted"].clip(upper=1.0)
            elif correction == "fdr":
                _, p_adjusted, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
                results_df["p_adjusted"] = p_adjusted

        return results_df

    def _get_paired_data(self, alg1: str, alg2: str, metric: str, group_by: str, task_column: str) -> pd.DataFrame:
        """Get paired data for two algorithms across tasks."""
        data1 = self.df[self.df[group_by] == alg1][[task_column, metric]]
        data2 = self.df[self.df[group_by] == alg2][[task_column, metric]]

        # Aggregate by task (in case of multiple runs per task)
        data1_agg = data1.groupby(task_column)[metric].mean().reset_index()
        data2_agg = data2.groupby(task_column)[metric].mean().reset_index()

        # Merge on task
        paired = data1_agg.merge(data2_agg, on=task_column, suffixes=("_1", "_2"))

        return paired

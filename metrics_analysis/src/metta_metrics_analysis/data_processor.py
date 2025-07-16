"""
Data Processing Module for Metta Metrics Analysis.

This module handles data transformation, cleaning, and preparation
for statistical analysis.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and transform WandB run data for analysis."""

    def __init__(self, data: pd.DataFrame | list[dict[str, Any]]):
        """
        Initialize the data processor.

        Args:
            data: Raw data from WandB collector (DataFrame or list of dicts)
        """
        if isinstance(data, list):
            self.df = pd.DataFrame(data)
        else:
            self.df = data.copy()

        # Store original data for reference
        self._original_df = self.df.copy()

        # Process initial data
        self._process_initial_data()

    def _process_initial_data(self) -> None:
        """Perform initial data processing."""
        # Convert timestamps
        if "created_at" in self.df.columns:
            self.df["created_at"] = pd.to_datetime(self.df["created_at"])

        # Sort by run and step
        if "step" in self.df.columns:
            self.df = self.df.sort_values(["run_id", "step"])

        # Identify metric columns
        self.metric_columns = [
            col
            for col in self.df.columns
            if col not in ["run_id", "run_name", "group", "tags", "state", "created_at", "step"]
            and not col.startswith("config.")
        ]

        logger.info(f"Processed {len(self.df)} rows with {len(self.metric_columns)} metrics")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df.copy()

    def pivot_by_step(self, metric: str, fillna: float | str | None = None) -> pd.DataFrame:
        """
        Pivot data to have runs as columns and steps as rows.

        Args:
            metric: Metric to pivot
            fillna: Value to fill NaN entries

        Returns:
            Pivoted DataFrame with steps as index and runs as columns
        """
        if metric not in self.metric_columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        pivot_df = self.df.pivot(index="step", columns="run_id", values=metric)

        if fillna is not None:
            pivot_df = pivot_df.fillna(fillna)

        return pivot_df

    def aggregate_by_run(
        self, metrics: list[str] | None = None, aggregations: dict[str, str | list[str]] | None = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics by run.

        Args:
            metrics: List of metrics to aggregate (defaults to all)
            aggregations: Dict mapping metrics to aggregation functions
                         e.g., {"reward": ["mean", "max"], "loss": "min"}

        Returns:
            DataFrame with one row per run
        """
        if metrics is None:
            metrics = self.metric_columns

        if aggregations is None:
            # Default aggregations
            aggregations = {metric: ["mean", "std", "min", "max", "last"] for metric in metrics}

        # Ensure all specified metrics have aggregations
        for metric in metrics:
            if metric not in aggregations:
                aggregations[metric] = ["mean", "std", "min", "max", "last"]

        # Group by run
        run_groups = self.df.groupby("run_id")

        # Apply aggregations
        agg_results = []
        for metric, agg_funcs in aggregations.items():
            if metric in self.df.columns:
                result = run_groups[metric].agg(agg_funcs)

                # Flatten column names
                if isinstance(agg_funcs, list):
                    result.columns = [f"{metric}_{func}" for func in agg_funcs]
                else:
                    result.name = f"{metric}_{agg_funcs}"

                agg_results.append(result)

        # Combine results
        agg_df = pd.concat(agg_results, axis=1)

        # Add run metadata
        run_metadata = run_groups.first()[["run_name", "group", "tags", "state", "created_at"]]

        # Add config columns
        config_cols = [col for col in self.df.columns if col.startswith("config.")]
        if config_cols:
            config_data = run_groups[config_cols].first()
            run_metadata = pd.concat([run_metadata, config_data], axis=1)

        result_df = pd.concat([run_metadata, agg_df], axis=1).reset_index()

        return result_df

    def filter_complete_runs(self, min_steps: int | None = None) -> "DataProcessor":
        """
        Filter to only include complete runs.

        Args:
            min_steps: Minimum number of steps required (if None, uses median)

        Returns:
            New DataProcessor with filtered data
        """
        if "step" not in self.df.columns:
            logger.warning("No step column found, returning original data")
            return self

        # Calculate steps per run
        steps_per_run = self.df.groupby("run_id")["step"].max()

        if min_steps is None:
            min_steps = int(steps_per_run.median())
            logger.info(f"Using median steps as threshold: {min_steps}")

        # Filter runs
        complete_runs = steps_per_run[steps_per_run >= min_steps].index
        filtered_df = self.df[self.df["run_id"].isin(complete_runs)]

        logger.info(f"Filtered from {len(steps_per_run)} to {len(complete_runs)} runs")

        return DataProcessor(filtered_df)

    def interpolate_missing(self, method: str = "linear", limit: int | None = None) -> "DataProcessor":
        """
        Interpolate missing values in metrics.

        Args:
            method: Interpolation method ('linear', 'ffill', 'bfill', etc.)
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            New DataProcessor with interpolated data
        """
        df_copy = self.df.copy()

        # Interpolate each metric for each run
        for metric in self.metric_columns:
            if metric in df_copy.columns:
                df_copy[metric] = df_copy.groupby("run_id")[metric].transform(
                    lambda x: x.interpolate(method=method, limit=limit)
                )

        return DataProcessor(df_copy)

    def normalize_metrics(
        self, metrics: list[str] | None = None, method: str = "minmax", per_run: bool = False
    ) -> "DataProcessor":
        """
        Normalize specified metrics.

        Args:
            metrics: Metrics to normalize (defaults to all)
            method: Normalization method ('minmax', 'zscore', 'robust')
            per_run: Whether to normalize per run or globally

        Returns:
            New DataProcessor with normalized data
        """
        if metrics is None:
            metrics = self.metric_columns

        df_copy = self.df.copy()

        for metric in metrics:
            if metric not in df_copy.columns:
                continue

            if per_run:
                # Normalize within each run
                df_copy[f"{metric}_normalized"] = df_copy.groupby("run_id")[metric].transform(
                    lambda x: self._normalize_series(x, method)
                )
            else:
                # Normalize globally
                df_copy[f"{metric}_normalized"] = self._normalize_series(df_copy[metric], method)

        return DataProcessor(df_copy)

    def _normalize_series(self, series: pd.Series, method: str) -> pd.Series:
        """Normalize a pandas Series."""
        if method == "minmax":
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            else:
                return series * 0  # All values are the same

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std > 0:
                return (series - mean) / std
            else:
                return series * 0

        elif method == "robust":
            # Use median and IQR
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                return (series - median) / iqr
            else:
                return series * 0

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def add_derived_metrics(self, metric_definitions: dict[str, str]) -> "DataProcessor":
        """
        Add derived metrics based on expressions.

        Args:
            metric_definitions: Dict mapping new metric names to expressions
                               e.g., {"efficiency": "reward / steps"}

        Returns:
            New DataProcessor with added metrics
        """
        df_copy = self.df.copy()

        for new_metric, expression in metric_definitions.items():
            try:
                # Use pandas eval for safe expression evaluation
                df_copy[new_metric] = df_copy.eval(expression)
                logger.info(f"Added derived metric: {new_metric}")
            except Exception as e:
                logger.warning(f"Failed to create metric {new_metric}: {e}")

        return DataProcessor(df_copy)

    def export_to_csv(self, filepath: str | Path, **kwargs) -> None:
        """Export processed data to CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(filepath, index=False, **kwargs)
        logger.info(f"Exported data to {filepath}")

    def export_to_parquet(self, filepath: str | Path, **kwargs) -> None:
        """Export processed data to Parquet format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_parquet(filepath, index=False, **kwargs)
        logger.info(f"Exported data to {filepath}")

    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for all metrics."""
        summary_stats = []

        for metric in self.metric_columns:
            if metric in self.df.columns:
                stats = {
                    "metric": metric,
                    "count": self.df[metric].count(),
                    "mean": self.df[metric].mean(),
                    "std": self.df[metric].std(),
                    "min": self.df[metric].min(),
                    "25%": self.df[metric].quantile(0.25),
                    "50%": self.df[metric].median(),
                    "75%": self.df[metric].quantile(0.75),
                    "max": self.df[metric].max(),
                    "missing": self.df[metric].isna().sum(),
                    "missing_pct": self.df[metric].isna().mean() * 100,
                }
                summary_stats.append(stats)

        return pd.DataFrame(summary_stats)

    def align_runs_by_step(self, max_step: int | None = None, fill_method: str = "forward") -> "DataProcessor":
        """
        Align all runs to have the same step indices.

        Args:
            max_step: Maximum step to include (if None, uses minimum across runs)
            fill_method: How to fill missing steps ('forward', 'interpolate', 'zero')

        Returns:
            New DataProcessor with aligned data
        """
        if "step" not in self.df.columns:
            logger.warning("No step column found, returning original data")
            return self

        # Find common step range
        step_ranges = self.df.groupby("run_id")["step"].agg(["min", "max"])

        if max_step is None:
            max_step = int(step_ranges["max"].min())
            logger.info(f"Using minimum max step across runs: {max_step}")

        min_step = int(step_ranges["min"].max())

        # Create aligned dataframe
        aligned_data = []

        for run_id in self.df["run_id"].unique():
            run_data = self.df[self.df["run_id"] == run_id].copy()

            # Create full step range
            full_steps = pd.DataFrame({"step": range(min_step, max_step + 1)})

            # Merge with existing data
            merged = full_steps.merge(run_data, on="step", how="left")

            # Fill run metadata
            for col in ["run_id", "run_name", "group", "tags", "state", "created_at"]:
                if col in run_data.columns:
                    merged[col] = merged[col].fillna(run_data[col].iloc[0])

            # Fill config columns
            config_cols = [col for col in run_data.columns if col.startswith("config.")]
            for col in config_cols:
                merged[col] = merged[col].fillna(run_data[col].iloc[0])

            # Fill metrics based on method
            for metric in self.metric_columns:
                if metric in merged.columns:
                    if fill_method == "forward":
                        merged[metric] = merged[metric].fillna(method="ffill")
                    elif fill_method == "interpolate":
                        merged[metric] = merged[metric].interpolate(method="linear")
                    elif fill_method == "zero":
                        merged[metric] = merged[metric].fillna(0)

            aligned_data.append(merged)

        aligned_df = pd.concat(aligned_data, ignore_index=True)

        return DataProcessor(aligned_df)

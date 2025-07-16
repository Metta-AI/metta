"""
WandB Data Collector for Metta Metrics Analysis.

This module provides functionality to fetch run data from Weights & Biases,
with specific support for Metta's metric naming conventions and run structure.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WandBDataCollector:
    """Collect and cache run data from Weights & Biases."""

    def __init__(
        self,
        entity: str = "metta-research",
        project: str = "metta",
        api_key: str | None = None,
        use_cache: bool = True,
        cache_dir: str | None = None,
        config: str | dict[str, Any] | None = None,
    ):
        """
        Initialize the WandB data collector.

        Args:
            entity: WandB entity name
            project: WandB project name
            api_key: Optional API key (uses WANDB_API_KEY env var if not provided)
            use_cache: Whether to cache fetched data
            cache_dir: Directory for caching data (defaults to ~/.metta/wandb_cache)
            config: Path to config file or config dict
        """
        self.entity = entity
        self.project = project
        self.use_cache = use_cache

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.metta/wandb_cache")
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb API
        if api_key:
            wandb.login(key=api_key)
        self.api = wandb.Api()

        # Load config if provided
        self.config = self._load_config(config) if config else {}

        # Metric groups for common analysis patterns
        self.metric_groups = {
            "training_core": [
                "trainer/loss",
                "trainer/policy_loss",
                "trainer/value_loss",
                "trainer/entropy",
                "trainer/learning_rate",
            ],
            "training_stats": [
                "trainer/explained_variance",
                "trainer/kl_divergence",
                "trainer/clip_fraction",
                "trainer/approx_kl",
            ],
            "agent_behavior": [
                "env_agent/reward_mean",
                "env_agent/episode_length_mean",
                "env_agent/actions_taken/*",
            ],
            "eval_navigation": [
                "eval_navigation/success_rate",
                "eval_navigation/path_efficiency",
                "eval_navigation/steps_mean",
                "eval_navigation/reward_mean",
            ],
            "eval_memory": [
                "eval_memory/success_rate",
                "eval_memory/reward_mean",
                "eval_memory/steps_mean",
            ],
            "system_metrics": [
                "monitor/gpu_utilization",
                "monitor/gpu_memory_used_gb",
                "monitor/cpu_percent",
                "timing_per_epoch/total_s",
            ],
        }

    def _load_config(self, config: str | dict[str, Any]) -> dict[str, Any]:
        """Load configuration from file or dict."""
        if isinstance(config, str):
            config_path = Path(config)
            if config_path.suffix == ".yaml":
                import yaml

                with open(config_path) as f:
                    return yaml.safe_load(f)
            elif config_path.suffix == ".json":
                with open(config_path) as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        return config

    def add_metric_group(self, name: str, metrics: list[str]) -> None:
        """Add a custom metric group for easy reference."""
        self.metric_groups[name] = metrics

    def fetch_runs(
        self,
        run_filter: str | dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        config_params: list[str] | None = None,
        last_n_steps: int | None = None,
        max_runs: int | None = None,
        include_system_metrics: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch runs from WandB with specified filters and metrics.

        Args:
            run_filter: Either a pattern string (e.g., "sky_comprehensive_*") or
                       a dict of wandb filters (e.g., {"group": "sweep_1"})
            metrics: List of metrics to fetch. Can include wildcards (*) and
                    metric group names (prefixed with @)
            config_params: List of config parameters to extract
            last_n_steps: Only fetch last N steps of each run
            max_runs: Maximum number of runs to fetch
            include_system_metrics: Whether to include system monitoring metrics

        Returns:
            DataFrame with run data
        """
        # Build query filters
        filters = self._build_filters(run_filter)

        # Expand metric groups and wildcards
        expanded_metrics = self._expand_metrics(metrics or [])
        if include_system_metrics:
            expanded_metrics.extend(self.metric_groups["system_metrics"])

        # Check cache first
        cache_key = self._get_cache_key(filters, expanded_metrics, config_params, last_n_steps)
        if self.use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} runs from cache")
                return cached_data

        # Fetch runs from API
        logger.info(f"Fetching runs from {self.entity}/{self.project}")
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters, per_page=100)

        # Collect data from runs
        all_data = []
        run_count = 0

        for run in tqdm(runs, desc="Fetching runs", disable=False):
            if max_runs and run_count >= max_runs:
                break

            run_data = self._extract_run_data(run, expanded_metrics, config_params, last_n_steps)
            if run_data:
                all_data.extend(run_data)
                run_count += 1

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Save to cache
        if self.use_cache and not df.empty:
            self._save_to_cache(cache_key, df)

        logger.info(f"Fetched {len(df)} data points from {run_count} runs")
        return df

    def _build_filters(self, run_filter: str | dict[str, Any] | None) -> dict[str, Any]:
        """Build WandB API filters from user input."""
        if run_filter is None:
            return {}

        if isinstance(run_filter, str):
            # Convert pattern to regex filter
            # Replace * with .* for regex
            pattern = run_filter.replace("*", ".*")
            return {"display_name": {"$regex": pattern}}

        return run_filter

    def _expand_metrics(self, metrics: list[str]) -> list[str]:
        """Expand metric groups and wildcards."""
        expanded = []

        for metric in metrics:
            if metric.startswith("@"):
                # Metric group reference
                group_name = metric[1:]
                if group_name in self.metric_groups:
                    expanded.extend(self.metric_groups[group_name])
                else:
                    logger.warning(f"Unknown metric group: {group_name}")
            else:
                expanded.append(metric)

        return list(set(expanded))  # Remove duplicates

    def _extract_run_data(
        self, run: Any, metrics: list[str], config_params: list[str] | None, last_n_steps: int | None
    ) -> list[dict[str, Any]]:
        """Extract data from a single run."""
        run_data = []

        # Basic run info
        base_info = {
            "run_id": run.id,
            "run_name": run.name,
            "group": run.group,
            "tags": ",".join(run.tags) if run.tags else "",
            "state": run.state,
            "created_at": run.created_at,
        }

        # Extract config parameters
        if config_params and run.config:
            for param in config_params:
                value = self._get_nested_value(run.config, param)
                base_info[f"config.{param}"] = value

        # Determine which metrics are available
        available_metrics = set()
        if run.summary:
            available_metrics.update(run.summary.keys())

        # Filter requested metrics to available ones
        metrics_to_fetch = []
        for metric in metrics:
            if "*" in metric:
                # Wildcard matching
                pattern = metric.replace("*", ".*")
                regex = re.compile(pattern)
                matching = [m for m in available_metrics if regex.match(m)]
                metrics_to_fetch.extend(matching)
            elif metric in available_metrics:
                metrics_to_fetch.append(metric)

        metrics_to_fetch = list(set(metrics_to_fetch))  # Remove duplicates

        if not metrics_to_fetch:
            # No metrics to fetch, just return run info
            run_data.append(base_info)
            return run_data

        # Fetch history data
        try:
            history = run.scan_history(keys=metrics_to_fetch + ["_step"])
            history_data = list(history)

            # Apply last_n_steps filter if specified
            if last_n_steps and len(history_data) > last_n_steps:
                history_data = history_data[-last_n_steps:]

            # Create a row for each step
            for row in history_data:
                step_data = base_info.copy()
                step_data["step"] = row.get("_step", 0)

                # Add metric values
                for metric in metrics_to_fetch:
                    step_data[metric] = row.get(metric)

                run_data.append(step_data)

        except Exception as e:
            logger.warning(f"Error fetching history for run {run.id}: {e}")
            # Fall back to summary data
            summary_data = base_info.copy()
            summary_data["step"] = run.summary.get("_step", 0)

            for metric in metrics_to_fetch:
                summary_data[metric] = run.summary.get(metric)

            run_data.append(summary_data)

        return run_data

    def _get_nested_value(self, config: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _get_cache_key(
        self, filters: dict[str, Any], metrics: list[str], config_params: list[str] | None, last_n_steps: int | None
    ) -> str:
        """Generate cache key for the query."""
        import hashlib

        key_parts = [
            self.entity,
            self.project,
            json.dumps(filters, sort_keys=True),
            json.dumps(sorted(metrics)),
            json.dumps(sorted(config_params or [])),
            str(last_n_steps),
        ]

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Load data from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            # Check if cache is recent (within 24 hours)
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                try:
                    return pd.read_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Error loading cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def get_available_metrics(self, run_id: str) -> set[str]:
        """Get all available metrics for a specific run."""
        try:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            metrics = set()

            if run.summary:
                metrics.update(run.summary.keys())

            # Remove internal wandb metrics
            metrics = {m for m in metrics if not m.startswith("_")}

            return metrics

        except Exception as e:
            logger.error(f"Error fetching metrics for run {run_id}: {e}")
            return set()

    def list_runs(self, filters: dict[str, Any] | None = None, limit: int = 100) -> pd.DataFrame:
        """List runs with basic information."""
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters or {}, per_page=min(limit, 100))

        run_info = []
        for i, run in enumerate(runs):
            if i >= limit:
                break

            run_info.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "group": run.group,
                    "tags": ",".join(run.tags) if run.tags else "",
                    "state": run.state,
                    "created_at": run.created_at,
                    "duration": run.summary.get("_runtime", 0) if run.summary else 0,
                    "step": run.summary.get("_step", 0) if run.summary else 0,
                }
            )

        return pd.DataFrame(run_info)

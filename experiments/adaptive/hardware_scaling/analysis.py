#!/usr/bin/env python3
"""Analysis module for hardware scaling experiments.

This module provides tools to analyze the results of hardware scaling experiments,
including:
- Scaling law fitting
- Figure of Merit (FOM) calculations
- Optimal configuration recommendations
- Visualization of results
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Results from scaling law analysis."""

    hardware_config: str
    gpus: int
    nodes: int
    best_score: float
    best_hyperparams: Dict[str, Any]
    time_to_target: float  # Hours to reach target performance
    samples_to_target: int  # Environment steps to reach target
    total_cost: float  # Total training cost in dollars
    sample_efficiency: float  # 1 / normalized_samples
    time_efficiency: float  # 1 / normalized_time
    cost_efficiency: float  # 1 / normalized_cost


class HardwareScalingAnalyzer:
    """Analyzer for hardware scaling experiment results."""

    def __init__(self, target_performance: float = 0.9):
        """Initialize the analyzer.

        Args:
            target_performance: Target performance level (0-1) for efficiency calculations
        """
        self.target_performance = target_performance
        self.results: List[ScalingResult] = []

    def analyze_wandb_runs(self, runs: List[Any]) -> pd.DataFrame:
        """Analyze runs from WandB to extract scaling results.

        Args:
            runs: List of WandB run objects

        Returns:
            DataFrame with analysis results
        """
        results = []

        # Group runs by hardware configuration
        runs_by_hardware = {}
        for run in runs:
            # Extract hardware ID from run name
            hw_id = self._extract_hardware_id(run.name)
            if hw_id:
                if hw_id not in runs_by_hardware:
                    runs_by_hardware[hw_id] = []
                runs_by_hardware[hw_id].append(run)

        # Analyze each hardware configuration
        for hw_id, hw_runs in runs_by_hardware.items():
            result = self._analyze_hardware_config(hw_id, hw_runs)
            if result:
                results.append(result)
                self.results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame([r.__dict__ for r in results])
        return df

    def fit_scaling_laws(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Fit power law scaling relationships.

        Args:
            df: DataFrame with scaling results

        Returns:
            Dictionary of scaling law parameters for each metric
        """
        scaling_laws = {}

        # Fit scaling for samples to target
        if "samples_to_target" in df.columns:
            gpus = df["gpus"].values * df["nodes"].values  # Total GPUs
            samples = df["samples_to_target"].values

            def power_law(x, a, b):
                return a * np.power(x, b)

            try:
                params, _ = curve_fit(power_law, gpus, samples)
                scaling_laws["samples"] = tuple(params)
                logger.info(
                    f"Samples scaling: S = {params[0]:.2e} * GPUs^{params[1]:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to fit samples scaling law: {e}")

        # Fit scaling for time to target
        if "time_to_target" in df.columns:
            gpus = df["gpus"].values * df["nodes"].values
            time = df["time_to_target"].values

            try:
                params, _ = curve_fit(power_law, gpus, time)
                scaling_laws["time"] = tuple(params)
                logger.info(f"Time scaling: T = {params[0]:.2e} * GPUs^{params[1]:.3f}")
            except Exception as e:
                logger.warning(f"Failed to fit time scaling law: {e}")

        # Fit scaling for cost
        if "total_cost" in df.columns:
            gpus = df["gpus"].values * df["nodes"].values
            cost = df["total_cost"].values

            try:
                params, _ = curve_fit(power_law, gpus, cost)
                scaling_laws["cost"] = tuple(params)
                logger.info(f"Cost scaling: C = {params[0]:.2e} * GPUs^{params[1]:.3f}")
            except Exception as e:
                logger.warning(f"Failed to fit cost scaling law: {e}")

        return scaling_laws

    def calculate_fom(
        self, df: pd.DataFrame, weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate Figure of Merit for different weight combinations.

        Args:
            df: DataFrame with scaling results
            weights: Dictionary with 'sample', 'time', 'cost' weights (must sum to 1)

        Returns:
            DataFrame with FOM scores added
        """
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

        # Normalize efficiency metrics
        df["sample_efficiency_norm"] = (
            df["sample_efficiency"] / df["sample_efficiency"].max()
        )
        df["time_efficiency_norm"] = df["time_efficiency"] / df["time_efficiency"].max()
        df["cost_efficiency_norm"] = df["cost_efficiency"] / df["cost_efficiency"].max()

        # Calculate FOM
        df["fom"] = (
            weights.get("sample", 0) * df["sample_efficiency_norm"]
            + weights.get("time", 0) * df["time_efficiency_norm"]
            + weights.get("cost", 0) * df["cost_efficiency_norm"]
        )

        return df

    def recommend_configurations(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Recommend optimal hardware configurations for different priorities.

        Args:
            df: DataFrame with scaling results and FOM scores

        Returns:
            Dictionary of recommendations for different priority scenarios
        """
        recommendations = {}

        # Define priority scenarios
        scenarios = {
            "research": {"sample": 0.6, "time": 0.2, "cost": 0.2},
            "development": {"sample": 0.2, "time": 0.6, "cost": 0.2},
            "budget": {"sample": 0.2, "time": 0.2, "cost": 0.6},
            "balanced": {"sample": 0.33, "time": 0.33, "cost": 0.34},
        }

        for scenario_name, weights in scenarios.items():
            # Calculate FOM for this scenario
            df_scenario = self.calculate_fom(df.copy(), weights)

            # Find best configuration
            best_idx = df_scenario["fom"].idxmax()
            best_config = df_scenario.loc[best_idx]

            recommendations[scenario_name] = {
                "hardware": best_config["hardware_config"],
                "gpus": int(best_config["gpus"]),
                "nodes": int(best_config["nodes"]),
                "fom": float(best_config["fom"]),
                "best_hyperparams": best_config["best_hyperparams"],
                "expected_score": float(best_config["best_score"]),
                "expected_time_hours": float(best_config["time_to_target"]),
                "expected_cost": float(best_config["total_cost"]),
            }

        return recommendations

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a text report of the analysis results.

        Args:
            df: DataFrame with complete analysis results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("HARDWARE SCALING EXPERIMENT REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total configurations tested: {len(df)}")
        report.append(f"Best absolute score: {df['best_score'].max():.4f}")
        report.append(
            f"Most sample-efficient: {df.loc[df['sample_efficiency'].idxmax()]['hardware_config']}"
        )
        report.append(
            f"Fastest to target: {df.loc[df['time_efficiency'].idxmax()]['hardware_config']}"
        )
        report.append(
            f"Most cost-effective: {df.loc[df['cost_efficiency'].idxmax()]['hardware_config']}"
        )
        report.append("")

        # Scaling laws
        scaling_laws = self.fit_scaling_laws(df)
        if scaling_laws:
            report.append("SCALING LAWS")
            report.append("-" * 40)
            for metric, (a, b) in scaling_laws.items():
                report.append(f"{metric.capitalize()}: {a:.2e} * GPUs^{b:.3f}")
            report.append("")

        # Recommendations
        recommendations = self.recommend_configurations(df)
        report.append("RECOMMENDED CONFIGURATIONS")
        report.append("-" * 40)
        for scenario, config in recommendations.items():
            report.append(f"\n{scenario.upper()} PRIORITY:")
            report.append(
                f"  Hardware: {config['hardware']} ({config['gpus']} GPUs on {config['nodes']} nodes)"
            )
            report.append(f"  Expected score: {config['expected_score']:.4f}")
            report.append(
                f"  Time to target: {config['expected_time_hours']:.2f} hours"
            )
            report.append(f"  Total cost: ${config['expected_cost']:.2f}")
            report.append(f"  FOM score: {config['fom']:.4f}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def _extract_hardware_id(self, run_name: str) -> Optional[str]:
        """Extract hardware configuration ID from run name."""
        parts = run_name.split("_")
        for i, part in enumerate(parts):
            if (
                part.startswith("g")
                and i + 1 < len(parts)
                and parts[i + 1].startswith("n")
            ):
                return f"{part}_{parts[i + 1]}"
        return None

    def _analyze_hardware_config(
        self, hw_id: str, runs: List[Any]
    ) -> Optional[ScalingResult]:
        """Analyze runs for a specific hardware configuration."""
        if not runs:
            return None

        # Extract GPU and node counts from hw_id
        parts = hw_id.split("_")
        gpus = int(parts[0][1:])  # Remove 'g' prefix
        nodes = int(parts[1][1:])  # Remove 'n' prefix

        # Find best run
        best_run = None
        best_score = -float("inf")

        for run in runs:
            if run.state == "finished" and run.summary:
                score = run.summary.get(
                    "sweep/score", run.summary.get("evaluator/eval_arena/score", 0)
                )
                if score > best_score:
                    best_score = score
                    best_run = run

        if not best_run:
            return None

        # Extract metrics from best run
        summary = best_run.summary
        config = best_run.config

        # Extract hyperparameters
        best_hyperparams = {
            "batch_size": config.get("trainer", {}).get("batch_size"),
            "minibatch_size": config.get("trainer", {}).get("minibatch_size"),
            "learning_rate": config.get("trainer", {})
            .get("optimizer", {})
            .get("learning_rate"),
            "ppo_clip": config.get("trainer", {})
            .get("losses", {})
            .get("ppo", {})
            .get("clip_coef"),
            "value_clip": config.get("trainer", {})
            .get("losses", {})
            .get("value", {})
            .get("clip_coef"),
            "gae_lambda": config.get("trainer", {})
            .get("losses", {})
            .get("value", {})
            .get("gae_lambda"),
        }

        # Calculate efficiency metrics
        # These would normally come from the training curves
        # For now, using placeholder calculations
        samples_to_target = summary.get("samples_to_target", 100_000_000)
        time_to_target = summary.get(
            "time_to_target", best_run.runtime / 3600 if best_run.runtime else 1.0
        )
        total_cost = summary.get(
            "sweep/cost", gpus * nodes * time_to_target * 1.5
        )  # $1.5 per GPU-hour

        # Normalize for efficiency calculations (lower is better, so invert)
        sample_efficiency = 1.0 / (samples_to_target / 1e8)  # Normalize to 100M
        time_efficiency = 1.0 / (time_to_target / 1.0)  # Normalize to 1 hour
        cost_efficiency = 1.0 / (total_cost / 10.0)  # Normalize to $10

        return ScalingResult(
            hardware_config=hw_id,
            gpus=gpus,
            nodes=nodes,
            best_score=best_score,
            best_hyperparams=best_hyperparams,
            time_to_target=time_to_target,
            samples_to_target=samples_to_target,
            total_cost=total_cost,
            sample_efficiency=sample_efficiency,
            time_efficiency=time_efficiency,
            cost_efficiency=cost_efficiency,
        )


def main():
    """Example usage of the analyzer."""
    import wandb

    # Initialize WandB
    api = wandb.Api()

    # Fetch runs from a specific project
    runs = api.runs("entity/hardware-scaling")

    # Analyze results
    analyzer = HardwareScalingAnalyzer(target_performance=0.9)
    df = analyzer.analyze_wandb_runs(runs)

    # Generate report
    report = analyzer.generate_report(df)
    print(report)

    # Save results
    df.to_csv("hardware_scaling_results.csv", index=False)
    with open("hardware_scaling_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()

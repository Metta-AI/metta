"""
Curriculum Analysis Module

This module provides tools for analyzing curriculum performance and calculating regret metrics
based on the CurriculumRegretProfile framework. It enables comparison of different curricula
across various scenarios and provides standardized regret calculations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CurriculumMetrics:
    """Container for curriculum performance metrics."""

    efficiency: float
    time_to_threshold: int
    time_to_first_mastery: int
    final_perf_variance: float
    task_weights: Dict[str, float]
    sampling_history: List[Dict[str, float]]


@dataclass
class RegretMetrics:
    """Container for regret calculation results."""

    efficiency_regret: float
    time_regret: int
    normalized_efficiency_regret: float
    normalized_time_regret: float


class CurriculumRegretAnalyzer:
    """
    Analyzer for calculating curriculum regret metrics.

    Based on the CurriculumRegretProfile framework, this class provides methods
    to compare curriculum performance against oracle baselines and calculate
    standardized regret metrics.
    """

    def __init__(self, max_epochs: int = 200):
        self.max_epochs = max_epochs

    def calculate_regret(
        self, curriculum_metrics: CurriculumMetrics, oracle_metrics: CurriculumMetrics
    ) -> RegretMetrics:
        """
        Calculate regret metrics by comparing curriculum performance to oracle baseline.

        Args:
            curriculum_metrics: Performance metrics for the curriculum being evaluated
            oracle_metrics: Performance metrics for the oracle baseline

        Returns:
            RegretMetrics object containing calculated regret values
        """
        # Efficiency regret (higher oracle efficiency - lower curriculum efficiency = positive regret)
        efficiency_regret = oracle_metrics.efficiency - curriculum_metrics.efficiency

        # Time regret calculation
        if curriculum_metrics.time_to_threshold == -1 and oracle_metrics.time_to_threshold != -1:
            # Curriculum failed but oracle succeeded - maximum penalty
            time_regret = self.max_epochs
        elif curriculum_metrics.time_to_threshold == -1 and oracle_metrics.time_to_threshold == -1:
            # Both failed - no regret
            time_regret = 0
        elif curriculum_metrics.time_to_threshold != -1 and oracle_metrics.time_to_threshold == -1:
            # Curriculum succeeded but oracle failed - negative regret (better than oracle)
            time_regret = -self.max_epochs
        else:
            # Both succeeded - difference in completion time
            time_regret = curriculum_metrics.time_to_threshold - oracle_metrics.time_to_threshold

        # Normalize regrets
        normalized_efficiency_regret = efficiency_regret / max(oracle_metrics.efficiency, 1e-6)
        normalized_time_regret = time_regret / self.max_epochs

        return RegretMetrics(
            efficiency_regret=efficiency_regret,
            time_regret=time_regret,
            normalized_efficiency_regret=normalized_efficiency_regret,
            normalized_time_regret=normalized_time_regret,
        )

    def compare_curricula(
        self, curricula_results: Dict[str, CurriculumMetrics], oracle_name: str = "oracle"
    ) -> pd.DataFrame:
        """
        Compare multiple curricula and calculate regret against oracle baseline.

        Args:
            curricula_results: Dictionary mapping curriculum names to their metrics
            oracle_name: Name of the oracle curriculum in the results

        Returns:
            DataFrame with comparison results including regret metrics
        """
        if oracle_name not in curricula_results:
            raise ValueError(f"Oracle curriculum '{oracle_name}' not found in results")

        oracle_metrics = curricula_results[oracle_name]
        comparison_data = []

        for name, metrics in curricula_results.items():
            if name == oracle_name:
                # Oracle has zero regret by definition
                regret = RegretMetrics(0.0, 0, 0.0, 0.0)
            else:
                regret = self.calculate_regret(metrics, oracle_metrics)

            row_data = {
                "curriculum": name,
                "efficiency": metrics.efficiency,
                "time_to_threshold": metrics.time_to_threshold,
                "time_to_first_mastery": metrics.time_to_first_mastery,
                "final_perf_variance": metrics.final_perf_variance,
                "efficiency_regret": regret.efficiency_regret,
                "time_regret": regret.time_regret,
                "normalized_efficiency_regret": regret.normalized_efficiency_regret,
                "normalized_time_regret": regret.normalized_time_regret,
            }
            comparison_data.append(row_data)

        return pd.DataFrame(comparison_data)

    def analyze_curriculum_adaptation(self, curriculum_metrics: CurriculumMetrics) -> Dict[str, float]:
        """
        Analyze how quickly a curriculum adapts to performance changes.

        Args:
            curriculum_metrics: Performance metrics for the curriculum

        Returns:
            Dictionary containing adaptation analysis metrics
        """
        if not curriculum_metrics.sampling_history:
            return {"adaptation_speed": 0.0, "weight_stability": 0.0}

        # Calculate weight stability (lower variance = more stable)
        weight_arrays = []
        for history_entry in curriculum_metrics.sampling_history:
            weights = list(history_entry.values())
            weight_arrays.append(weights)

        weight_matrix = np.array(weight_arrays)
        weight_variance = np.var(weight_matrix, axis=0).mean()
        weight_stability = 1.0 / (1.0 + weight_variance)  # Higher stability for lower variance

        # Calculate adaptation speed (how quickly weights change)
        if len(weight_arrays) > 1:
            weight_changes = []
            for i in range(1, len(weight_arrays)):
                change = np.mean(np.abs(np.array(weight_arrays[i]) - np.array(weight_arrays[i - 1])))
                weight_changes.append(change)
            adaptation_speed = np.mean(weight_changes)
        else:
            adaptation_speed = 0.0

        return {"adaptation_speed": adaptation_speed, "weight_stability": weight_stability}


class CurriculumScenarioAnalyzer:
    """
    Analyzer for comparing curricula across different scenarios.

    This class provides methods to run comprehensive curriculum comparisons
    across multiple scenarios and generate standardized reports.
    """

    def __init__(self, regret_analyzer: CurriculumRegretAnalyzer):
        self.regret_analyzer = regret_analyzer

    def run_scenario_comparison(self, scenario_results: Dict[str, Dict[str, CurriculumMetrics]]) -> pd.DataFrame:
        """
        Run comprehensive comparison across multiple scenarios.

        Args:
            scenario_results: Dictionary mapping scenario names to curriculum results

        Returns:
            DataFrame with comprehensive comparison results
        """
        all_results = []

        for scenario_name, curricula_results in scenario_results.items():
            # Calculate regret for this scenario
            scenario_df = self.regret_analyzer.compare_curricula(curricula_results)
            scenario_df["scenario"] = scenario_name

            # Add adaptation analysis
            adaptation_metrics = []
            for _, row in scenario_df.iterrows():
                curriculum_name = row["curriculum"]
                if curriculum_name in curricula_results:
                    adaptation = self.regret_analyzer.analyze_curriculum_adaptation(curricula_results[curriculum_name])
                    adaptation_metrics.append(adaptation)
                else:
                    adaptation_metrics.append({"adaptation_speed": 0.0, "weight_stability": 0.0})

            # Add adaptation columns
            scenario_df["adaptation_speed"] = [m["adaptation_speed"] for m in adaptation_metrics]
            scenario_df["weight_stability"] = [m["weight_stability"] for m in adaptation_metrics]

            all_results.append(scenario_df)

        return pd.concat(all_results, ignore_index=True)

    def generate_summary_report(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary report from comparison results.

        Args:
            comparison_df: DataFrame with comparison results

        Returns:
            Dictionary containing summary statistics
        """
        # Filter out oracle from analysis
        analysis_df = comparison_df[comparison_df["curriculum"] != "oracle"]

        summary = {
            "total_scenarios": len(analysis_df["scenario"].unique()),
            "total_curricula": len(analysis_df["curriculum"].unique()),
            "best_overall_efficiency": analysis_df["efficiency"].max(),
            "worst_overall_efficiency": analysis_df["efficiency"].min(),
            "average_efficiency_regret": analysis_df["efficiency_regret"].mean(),
            "average_time_regret": analysis_df["time_regret"].mean(),
            "curriculum_rankings": {},
        }

        # Calculate curriculum rankings
        for curriculum in analysis_df["curriculum"].unique():
            curriculum_data = analysis_df[analysis_df["curriculum"] == curriculum]
            summary["curriculum_rankings"][curriculum] = {
                "avg_efficiency": curriculum_data["efficiency"].mean(),
                "avg_efficiency_regret": curriculum_data["efficiency_regret"].mean(),
                "avg_time_regret": curriculum_data["time_regret"].mean(),
                "avg_adaptation_speed": curriculum_data["adaptation_speed"].mean(),
                "avg_weight_stability": curriculum_data["weight_stability"].mean(),
            }

        return summary


def create_curriculum_metrics(
    efficiency: float,
    time_to_threshold: int,
    time_to_first_mastery: int,
    final_perf_variance: float,
    task_weights: Dict[str, float],
    sampling_history: Optional[List[Dict[str, float]]] = None,
) -> CurriculumMetrics:
    """
    Factory function to create CurriculumMetrics objects.

    Args:
        efficiency: Overall learning efficiency
        time_to_threshold: Time to reach performance threshold
        time_to_first_mastery: Time to first task mastery
        final_perf_variance: Variance in final performance
        task_weights: Current task weights
        sampling_history: History of sampling probabilities

    Returns:
        CurriculumMetrics object
    """
    if sampling_history is None:
        sampling_history = []

    return CurriculumMetrics(
        efficiency=efficiency,
        time_to_threshold=time_to_threshold,
        time_to_first_mastery=time_to_first_mastery,
        final_perf_variance=final_perf_variance,
        task_weights=task_weights,
        sampling_history=sampling_history,
    )

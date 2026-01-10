"""
Alignment metrics for multi-agent systems.

This module implements the GAMMA (General Alignment Metric for Multi-agent Autonomy)
framework for quantifying alignment in autonomous agent swarms.

Based on: "GAMMA: A Framework-Agnostic Alignment Metric for Autonomous Swarms"
by Marcel Blattner and Adam Goldstein
"""

from metta.alignment.metrics.directional_intent import DirectionalIntentMetric
from metta.alignment.metrics.energy_proportionality import EnergyProportionalityMetric
from metta.alignment.metrics.gamma import GAMMAMetric, IndividualAlignmentMetric
from metta.alignment.metrics.goal_attainment import GoalAttainmentMetric
from metta.alignment.metrics.path_efficiency import PathEfficiencyMetric
from metta.alignment.metrics.time_efficiency import TimeEfficiencyMetric

__all__ = [
    "DirectionalIntentMetric",
    "PathEfficiencyMetric",
    "GoalAttainmentMetric",
    "TimeEfficiencyMetric",
    "EnergyProportionalityMetric",
    "GAMMAMetric",
    "IndividualAlignmentMetric",
]

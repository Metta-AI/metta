"""Evaluation missions and utilities for scripted agent testing."""

from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_LEVELS, apply_difficulty
from cogames.cogs_vs_clips.evals.eval_missions import (
    CarbonDesert,
    ClipCarbon,
    ClipCharger,
    ClipGermanium,
    ClipOxygen,
    ClipSilicon,
    GermaniumRush,
    HighRegenSprint,
    OxygenBottleneck,
    SiliconWorkbench,
    SlowOxygen,
    SparseBalanced,
)
from cogames.cogs_vs_clips.evals.exploration_evals import (
    Experiment1Mission,
    Experiment2Mission,
    Experiment4Mission,
    Experiment5Mission,
    Experiment6Mission,
    Experiment7Mission,
    Experiment8Mission,
    Experiment9Mission,
    Experiment10Mission,
)

__all__ = [
    # Difficulty variants
    "DIFFICULTY_LEVELS",
    "apply_difficulty",
    # Eval missions
    "CarbonDesert",
    "ClipCarbon",
    "ClipCharger",
    "ClipGermanium",
    "ClipOxygen",
    "ClipSilicon",
    "GermaniumRush",
    "HighRegenSprint",
    "OxygenBottleneck",
    "SiliconWorkbench",
    "SlowOxygen",
    "SparseBalanced",
    # Exploration experiments
    "Experiment1Mission",
    "Experiment2Mission",
    "Experiment4Mission",
    "Experiment5Mission",
    "Experiment6Mission",
    "Experiment7Mission",
    "Experiment8Mission",
    "Experiment9Mission",
    "Experiment10Mission",
]


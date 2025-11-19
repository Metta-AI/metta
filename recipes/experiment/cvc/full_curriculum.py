"""Backward compatibility wrapper for full_curriculum.

This module redirects to mission_variant_curriculum with all_variants_per_mission=False.
"""

from recipes.experiment.cvc import mission_variant_curriculum
from recipes.experiment.cvc.mission_variant_curriculum import (
    DIAGNOSTIC_MISSIONS,
    FULL_CURRICULUM_MISSIONS,
    TRAINING_FACILITY_MISSIONS,
    evaluate,
    play,
)

# Re-export everything for backward compatibility
__all__ = [
    "make_curriculum",
    "train",
    "evaluate",
    "play",
    "experiment",
    "FULL_CURRICULUM_MISSIONS",
    "DIAGNOSTIC_MISSIONS",
    "TRAINING_FACILITY_MISSIONS",
]


def make_curriculum(
    num_cogs: int = 4,
    base_missions=None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config=None,
    variants=None,
    **kwargs,
):
    """Create a full curriculum (backward compatibility wrapper)."""
    return mission_variant_curriculum.make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        algorithm_config=algorithm_config,
        variants=variants,
        all_variants_per_mission=False,
        stats_max_cap=1.0,
        **kwargs,
    )


def train(
    num_cogs: int = 4,
    curriculum=None,
    base_missions=None,
    enable_detailed_slice_logging: bool = False,
    variants=None,
    eval_variants=None,
    eval_difficulty: str | None = "standard",
    **kwargs,
):
    """Create a training tool for full curriculum (backward compatibility wrapper)."""
    return mission_variant_curriculum.train(
        base_missions=base_missions,
        num_cogs=num_cogs,
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        exclude_variants=None,
        all_variants_per_mission=False,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        **kwargs,
    )


def experiment(
    run_name=None,
    num_cogs: int = 4,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    additional_args=None,
    **kwargs,
):
    """Submit a full curriculum training job (backward compatibility wrapper)."""
    return mission_variant_curriculum.experiment(
        base_missions=None,  # Will default to FULL_CURRICULUM_MISSIONS
        run_name=run_name,
        num_cogs=num_cogs,
        heartbeat_timeout=heartbeat_timeout,
        skip_git_check=skip_git_check,
        variants=None,  # No variants for full curriculum
        exclude_variants=None,
        all_variants_per_mission=False,
        additional_args=additional_args,
        **kwargs,
    )

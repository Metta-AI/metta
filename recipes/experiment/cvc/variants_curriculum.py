"""Backward compatibility wrapper for variants_curriculum.

This module redirects to mission_variant_curriculum with all_variants_per_mission=True.
"""

from recipes.experiment.cvc import mission_variant_curriculum
from recipes.experiment.cvc.mission_variant_curriculum import evaluate, play


def make_curriculum(
    base_missions,
    num_cogs: int = 4,
    enable_detailed_slice_logging: bool = False,
    algorithm_config=None,
    exclude_variants=None,
    **kwargs,
):
    """Create a variants curriculum (backward compatibility wrapper)."""
    return mission_variant_curriculum.make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        algorithm_config=algorithm_config,
        variants=None,
        exclude_variants=exclude_variants,
        all_variants_per_mission=True,
        stats_max_cap=0.5,
        **kwargs,
    )


def train(
    base_missions,
    num_cogs: int = 4,
    curriculum=None,
    enable_detailed_slice_logging: bool = False,
    exclude_variants=None,
    eval_variants=None,
    eval_difficulty: str | None = "standard",
    **kwargs,
):
    """Create a training tool for variants curriculum (backward compatibility wrapper)."""
    return mission_variant_curriculum.train(
        base_missions=base_missions,
        num_cogs=num_cogs,
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=None,
        exclude_variants=exclude_variants,
        all_variants_per_mission=True,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        **kwargs,
    )


def experiment(
    base_missions,
    run_name=None,
    num_cogs: int = 4,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    exclude_variants=None,
    additional_args=None,
    **kwargs,
):
    """Submit a variants curriculum training job (backward compatibility wrapper)."""
    return mission_variant_curriculum.experiment(
        base_missions=base_missions,
        run_name=run_name,
        num_cogs=num_cogs,
        heartbeat_timeout=heartbeat_timeout,
        skip_git_check=skip_git_check,
        variants=None,
        exclude_variants=exclude_variants,
        all_variants_per_mission=True,
        additional_args=additional_args,
        **kwargs,
    )


__all__ = [
    "make_curriculum",
    "train",
    "evaluate",
    "play",
    "experiment",
]

"""Performance testing wrapper for CoGs vs Clips training.

This wrapper adds performance configuration parameters to the train function
for easy testing of LP performance optimizations.
"""

from __future__ import annotations

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import DiscreteRandomCurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.tools.train import TrainTool
from recipes.experiment.cogs_v_clips import DEFAULT_CURRICULUM_MISSIONS, make_training_env, train


def train_with_perf_config(
    num_cogs: int = 4,
    mission: Optional[str] = None,
    base_missions: Optional[list[str]] = None,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    max_evals: Optional[int] = None,
    bc_policy_uri: Optional[str] = None,
    bc_teacher_lead_prob: float = 1.0,
    use_lp: bool = True,
    # Performance optimization parameters
    perf_invalidation_batch_size: int = 100,
    perf_cache_task_list: bool = True,
    perf_log_metrics: bool = False,
    num_active_tasks: int = 1000,
    # Required variants for all runs
    force_variants: Sequence[str] = ("inventory_heart_tune", "heart_chorus"),
) -> TrainTool:
    """Create a training tool for CoGs vs Clips with performance configuration.

    This is a wrapper around cogs_v_clips.train that adds performance
    configuration parameters for easy testing.

    Args:
        num_cogs: Number of CoGs agents to train
        mission: Optional single mission to train on
        base_missions: List of missions for curriculum
        variants: Training variants to use (merged with force_variants)
        eval_variants: Evaluation variants to use
        eval_difficulty: Difficulty level for evaluation
        max_evals: Maximum number of evaluations
        bc_policy_uri: URI for behavior cloning policy
        bc_teacher_lead_prob: Probability of following teacher in BC
        use_lp: Use Learning Progress curriculum (vs DiscreteRandom)
        perf_invalidation_batch_size: Batch size for cache invalidation (1-1000)
        perf_cache_task_list: Enable task list caching
        perf_log_metrics: Log performance metrics
        num_active_tasks: Size of active task pool
        force_variants: Variants that are always applied (default: inventory_heart_tune, heart_chorus)
    """
    training_missions = base_missions or DEFAULT_CURRICULUM_MISSIONS
    if mission is not None:
        training_missions = [mission]

    # Merge force_variants with user-provided variants
    merged_variants = list(force_variants)
    if variants:
        # Add user variants, avoiding duplicates
        for v in variants:
            if v not in merged_variants:
                merged_variants.append(v)
    final_variants = merged_variants if merged_variants else None

    # Create algorithm config with performance parameters
    if use_lp:
        cur_alg = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            slow_timescale_factor=0.2,
            rand_task_rate=0.01,
            exploration_bonus=0.1,
            min_samples_for_lp=10,
            lp_score_temperature=0.0,
            z_score_amplification=50.0,
            show_curriculum_troubleshooting_logging=False,  # Disable per-task metrics (3000+ metrics with 1000 tasks)
            early_progress_amplification=0.5,
            # Performance parameters (configurable)
            perf_invalidation_batch_size=perf_invalidation_batch_size,
            perf_cache_task_list=perf_cache_task_list,
            perf_log_metrics=perf_log_metrics,
        )
    else:
        cur_alg = DiscreteRandomCurriculumConfig()  # noqa: F821

    # Build curriculum tasks
    all_mission_tasks = []
    for mission_name in training_missions:
        mission_env = make_training_env(
            num_cogs=num_cogs,
            mission=mission_name,
            variants=final_variants,
        )
        mission_tasks = cc.bucketed(mission_env)
        mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])
        all_mission_tasks.append(mission_tasks)

    merged_tasks = cc.merge(all_mission_tasks)

    # Create curriculum with configurable num_active_tasks
    curriculum = merged_tasks.to_curriculum(
        num_active_tasks=num_active_tasks,
        algorithm_config=cur_alg,
    )

    # Call original train function with pre-configured curriculum
    # Note: final_variants is passed to evaluation suite configuration
    return train(
        num_cogs=num_cogs,
        curriculum=curriculum,
        variants=final_variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        max_evals=max_evals,
        bc_policy_uri=bc_policy_uri,
        bc_teacher_lead_prob=bc_teacher_lead_prob,
    )


__all__ = ["train_with_perf_config"]

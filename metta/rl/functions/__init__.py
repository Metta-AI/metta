"""Functional training utilities for Metta.

This module provides functional implementations of the core training loop components,
extracting the rollout and train logic from MettaTrainer into standalone functions.
"""

# Re-export all functions for backward compatibility
from metta.rl.functions.advantage import (
    compute_advantage,
    normalize_advantage_distributed,
)
from metta.rl.functions.policy_management import (
    cleanup_old_policies,
    ensure_initial_policy,
    save_policy_with_metadata,
    validate_policy_environment_match,
)
from metta.rl.functions.rollout import (
    get_lstm_config,
    get_observation,
    run_policy_inference,
)
from metta.rl.functions.stats import (
    accumulate_rollout_stats,
    build_wandb_stats,
    compute_gradient_stats,
    compute_timing_stats,
    process_stats,
    process_training_stats,
)
from metta.rl.functions.training import (
    calculate_batch_sizes,
    calculate_explained_variance,
    calculate_prioritized_sampling_params,
    compute_ppo_losses,
    evaluate_policy,
    generate_replay,
    maybe_update_l2_weights,
    process_minibatch_update,
    setup_distributed_vars,
    should_run_on_interval,
)

__all__ = [
    # Advantage computation
    "compute_advantage",
    "normalize_advantage_distributed",
    # Policy management
    "cleanup_old_policies",
    "ensure_initial_policy",
    "save_policy_with_metadata",
    "validate_policy_environment_match",
    # Rollout functions
    "get_lstm_config",
    "get_observation",
    "run_policy_inference",
    # Stats functions
    "accumulate_rollout_stats",
    "build_wandb_stats",
    "compute_gradient_stats",
    "compute_timing_stats",
    "process_stats",
    "process_training_stats",
    # Training functions
    "calculate_batch_sizes",
    "calculate_explained_variance",
    "calculate_prioritized_sampling_params",
    "compute_ppo_losses",
    "evaluate_policy",
    "generate_replay",
    "maybe_update_l2_weights",
    "process_minibatch_update",
    "setup_distributed_vars",
    "should_run_on_interval",
]

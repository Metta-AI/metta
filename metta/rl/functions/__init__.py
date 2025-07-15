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
)

__all__ = [
    # Rollout functions
    "get_observation",
    "run_policy_inference",
    "get_lstm_config",
    # Training functions
    "process_minibatch_update",
    "compute_ppo_losses",
    "calculate_batch_sizes",
    "calculate_prioritized_sampling_params",
    "calculate_explained_variance",
    "setup_distributed_vars",
    "maybe_update_l2_weights",
    "evaluate_policy",
    "generate_replay",
    # Advantage functions
    "compute_advantage",
    "normalize_advantage_distributed",
    # Stats functions
    "accumulate_rollout_stats",
    "process_training_stats",
    "compute_timing_stats",
    "build_wandb_stats",
    "process_stats",
    # Policy management
    "cleanup_old_policies",
    "save_policy_with_metadata",
    "validate_policy_environment_match",
    "ensure_initial_policy",
]

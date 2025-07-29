"""Functional training utilities for Metta.

This module provides functional implementations of the core training loop components,
extracting the rollout and train logic from MettaTrainer into standalone functions.
"""

# Advantage computation
from .advantage import compute_advantage, normalize_advantage_distributed

# Batch utilities
from .batch_utils import calculate_batch_sizes, calculate_prioritized_sampling_params

# Distributed utilities
from .distributed import setup_device_and_distributed, setup_distributed_vars

# Evaluation utilities
from .evaluation import evaluate_policy, generate_replay

# Loss computation
from .losses import compute_ppo_losses, process_minibatch_update

# Optimization utilities
from .optimization import (
    compute_gradient_stats,
    maybe_update_l2_weights,
)

# Policy management
from .policy_management import (
    load_or_initialize_policy,
    maybe_load_checkpoint,
    validate_policy_environment_match,
    wrap_agent_distributed,
)

# Rollout utilities
from .rollout import get_lstm_config, get_observation, run_policy_inference

# Statistics and metrics
from .stats import (
    accumulate_rollout_stats,
    build_wandb_stats,
    compute_timing_stats,
    process_stats,
    process_training_stats,
)

__all__ = [
    # Advantage
    "compute_advantage",
    "normalize_advantage_distributed",
    # Batch utils
    "calculate_batch_sizes",
    "calculate_prioritized_sampling_params",
    # Distributed
    "setup_device_and_distributed",
    "setup_distributed_vars",
    # Evaluation
    "evaluate_policy",
    "generate_replay",
    # Losses
    "compute_ppo_losses",
    "process_minibatch_update",
    # Optimization
    "compute_gradient_stats",
    "maybe_update_l2_weights",
    # Policy management
    "load_or_initialize_policy",
    "maybe_load_checkpoint",
    "validate_policy_environment_match",
    "wrap_agent_distributed",
    # Rollout
    "get_lstm_config",
    "get_observation",
    "run_policy_inference",
    # Stats
    "accumulate_rollout_stats",
    "build_wandb_stats",
    "compute_timing_stats",
    "process_stats",
    "process_training_stats",
]

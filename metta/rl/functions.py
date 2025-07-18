# Temporary compatibility layer for metta.rl.functions imports
# This file makes old imports work while functions is now a directory
# This file will be removed when all imports are properly updated

# Import all functions directly from submodules to avoid circular imports
from metta.rl.functions.advantage import (
    compute_advantage,
    normalize_advantage_distributed,
)
from metta.rl.functions.policy_management import (
    cleanup_old_policies,
    maybe_load_checkpoint,
    save_policy_with_metadata,
    validate_policy_environment_match,
    wrap_agent_distributed,
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
    compute_gradient_stats,
    compute_ppo_losses,
    evaluate_policy,
    generate_replay,
    maybe_update_l2_weights,
    process_minibatch_update,
    setup_device_and_distributed,
    setup_distributed_vars,
    should_run,
)

# Re-export all for wildcard imports
__all__ = [
    # From advantage
    "compute_advantage",
    "normalize_advantage_distributed",
    # From policy_management
    "cleanup_old_policies",
    "maybe_load_checkpoint",
    "save_policy_with_metadata",
    "validate_policy_environment_match",
    "wrap_agent_distributed",
    # From rollout
    "get_lstm_config",
    "get_observation",
    "run_policy_inference",
    # From stats
    "accumulate_rollout_stats",
    "build_wandb_stats",
    "compute_timing_stats",
    "process_stats",
    "process_training_stats",
    # From training
    "calculate_batch_sizes",
    "calculate_explained_variance",
    "calculate_prioritized_sampling_params",
    "compute_gradient_stats",
    "compute_ppo_losses",
    "evaluate_policy",
    "generate_replay",
    "maybe_update_l2_weights",
    "process_minibatch_update",
    "setup_device_and_distributed",
    "setup_distributed_vars",
    "should_run",
]

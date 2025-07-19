"""Functional training utilities for Metta.
This module provides functional implementations of the core training loop components,
extracting the rollout and train logic from MettaTrainer into standalone functions.
"""

# Re-export all functions for backward compatibility
from metta.rl.functions.advantage import (
    compute_advantage as compute_advantage,
)
from metta.rl.functions.advantage import (
    normalize_advantage_distributed as normalize_advantage_distributed,
)
from metta.rl.functions.policy_management import (
    cleanup_old_policies as cleanup_old_policies,
)
from metta.rl.functions.policy_management import (
    maybe_load_checkpoint as maybe_load_checkpoint,
)
from metta.rl.functions.policy_management import (
    save_policy_with_metadata as save_policy_with_metadata,
)
from metta.rl.functions.policy_management import (
    validate_policy_environment_match as validate_policy_environment_match,
)
from metta.rl.functions.policy_management import (
    wrap_agent_distributed as wrap_agent_distributed,
)
from metta.rl.functions.rollout import (
    get_lstm_config as get_lstm_config,
)
from metta.rl.functions.rollout import (
    get_observation as get_observation,
)
from metta.rl.functions.rollout import (
    run_policy_inference as run_policy_inference,
)
from metta.rl.functions.stats import (
    accumulate_rollout_stats as accumulate_rollout_stats,
)
from metta.rl.functions.stats import (
    build_wandb_stats as build_wandb_stats,
)
from metta.rl.functions.stats import (
    compute_timing_stats as compute_timing_stats,
)
from metta.rl.functions.stats import (
    process_stats as process_stats,
)
from metta.rl.functions.stats import (
    process_training_stats as process_training_stats,
)
from metta.rl.functions.training import (
    calculate_batch_sizes as calculate_batch_sizes,
)
from metta.rl.functions.training import (
    calculate_explained_variance as calculate_explained_variance,
)
from metta.rl.functions.training import (
    calculate_prioritized_sampling_params as calculate_prioritized_sampling_params,
)
from metta.rl.functions.training import (
    compute_gradient_stats as compute_gradient_stats,
)
from metta.rl.functions.training import (
    compute_ppo_losses as compute_ppo_losses,
)
from metta.rl.functions.training import (
    evaluate_policy as evaluate_policy,
)
from metta.rl.functions.training import (
    generate_replay as generate_replay,
)
from metta.rl.functions.training import (
    maybe_update_l2_weights as maybe_update_l2_weights,
)
from metta.rl.functions.training import (
    process_minibatch_update as process_minibatch_update,
)
from metta.rl.functions.training import (
    setup_device_and_distributed as setup_device_and_distributed,
)
from metta.rl.functions.training import (
    setup_distributed_vars as setup_distributed_vars,
)
from metta.rl.functions.training import (
    should_run as should_run,
)

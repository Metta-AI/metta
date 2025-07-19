# Compatibility layer for metta.rl.functions imports
# This allows Python to treat metta.rl.functions as either a module or package
#
# This compatibility layer exists because:
# 1. Some Python environments try to import metta.rl.functions as a package
# 2. We're in the process of deprecating metta.rl.functions.py
# 3. This ensures imports work consistently across different environments
#
# TODO: Remove this compatibility layer once metta.rl.functions.py is fully
# deprecated and all imports have been migrated to their new locations

# Import everything from the parent functions.py module
from ..functions import *

# Also explicitly re-export commonly used functions to ensure they're available
from ..functions import (
    accumulate_rollout_stats,
    build_wandb_stats,
    calculate_batch_sizes,
    calculate_explained_variance,
    calculate_prioritized_sampling_params,
    cleanup_old_policies,
    compute_advantage,
    compute_gradient_stats,
    compute_timing_stats,
    get_lstm_config,
    get_observation,
    process_minibatch_update,
    process_training_stats,
    run_policy_inference,
    save_policy_with_metadata,
    validate_policy_environment_match,
)

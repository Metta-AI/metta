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
# noqa: F403, F401, TID252 - This is a compatibility layer
from metta.rl.functions import *  # noqa: F403, TID252

# Also explicitly re-export commonly used functions to ensure they're available
# noqa comments needed for compatibility layer
from metta.rl.functions import (  # noqa: TID252
    accumulate_rollout_stats,  # noqa: F401
    build_wandb_stats,  # noqa: F401
    calculate_batch_sizes,  # noqa: F401
    calculate_explained_variance,  # noqa: F401
    calculate_prioritized_sampling_params,  # noqa: F401
    cleanup_old_policies,  # noqa: F401
    compute_advantage,  # noqa: F401
    compute_gradient_stats,  # noqa: F401
    compute_timing_stats,  # noqa: F401
    get_lstm_config,  # noqa: F401
    get_observation,  # noqa: F401
    process_minibatch_update,  # noqa: F401
    process_training_stats,  # noqa: F401
    run_policy_inference,  # noqa: F401
    save_policy_with_metadata,  # noqa: F401
    validate_policy_environment_match,  # noqa: F401
)

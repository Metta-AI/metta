# metta/__init__.py
"""Metta: A reinforcement learning codebase for multi-agent gridworlds."""

# Import key API functions and configs for convenience
from metta.api import (
    # Factory classes
    Agent,
    # Configuration classes (Pydantic models)
    CheckpointConfig,
    Environment,
    # Training components
    Experience,
    Kickstarter,
    Losses,
    # New wrapper classes
    Optimizer,
    OptimizerConfig,
    PPOConfig,
    # Helper classes
    PreBuiltConfigCurriculum,
    RunDirectories,
    SimulationConfig,
    Stopwatch,
    TrainerConfig,
    # Functions from rl.functions
    accumulate_rollout_stats,
    # Helper functions
    calculate_anneal_beta,
    cleanup_distributed,
    compute_advantage,
    create_evaluation_config_suite,
    create_replay_config,
    load_checkpoint,
    maybe_update_l2_weights,
    perform_rollout_step,
    process_minibatch_update,
    save_checkpoint,
    save_experiment_config,
    setup_device_and_distributed,
    setup_distributed_vars,
    setup_run_directories,
    should_run_on_interval,
    wrap_agent_distributed,
)

__version__ = "0.1.0"

__all__ = [
    # Factory classes
    "Agent",
    "Environment",
    "Optimizer",
    # Configuration classes (Pydantic models)
    "CheckpointConfig",
    "OptimizerConfig",
    "PPOConfig",
    "SimulationConfig",
    "TrainerConfig",
    # Training components
    "Experience",
    "Kickstarter",
    "Losses",
    "Stopwatch",
    # Helper functions
    "calculate_anneal_beta",
    "cleanup_distributed",
    "create_evaluation_config_suite",
    "create_replay_config",
    "load_checkpoint",
    "maybe_update_l2_weights",
    "save_checkpoint",
    "save_experiment_config",
    "setup_device_and_distributed",
    "setup_distributed_vars",
    "setup_run_directories",
    "should_run_on_interval",
    "wrap_agent_distributed",
    # Functions from rl.functions
    "accumulate_rollout_stats",
    "compute_advantage",
    "perform_rollout_step",
    "process_minibatch_update",
    # Helper classes
    "PreBuiltConfigCurriculum",
    "RunDirectories",
    # Version
    "__version__",
]

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
    Optimizer,
    OptimizerConfig,
    PPOConfig,
    # Helper classes
    RunDirectories,
    SimulationConfig,
    Stopwatch,
    TrainerConfig,
    accumulate_rollout_stats,
    # Helper functions
    calculate_anneal_beta,
    compute_advantages_with_config,
    create_policy_store,
    # Functions from rl.functions
    perform_rollout_step,
    process_minibatch_update,
    save_checkpoint,
    save_experiment_config,
    setup_run_directories,
)

__version__ = "0.1.0"

__all__ = [
    # Factory classes
    "Agent",
    "Environment",
    "Optimizer",
    # Configuration classes
    "CheckpointConfig",
    "OptimizerConfig",
    "PPOConfig",
    "SimulationConfig",
    "TrainerConfig",
    # Helper functions
    "calculate_anneal_beta",
    "save_checkpoint",
    "setup_run_directories",
    "save_experiment_config",
    "create_policy_store",
    "compute_advantages_with_config",
    # Functions from rl.functions
    "perform_rollout_step",
    "process_minibatch_update",
    "accumulate_rollout_stats",
    # Training components
    "Experience",
    "Kickstarter",
    "Losses",
    "Stopwatch",
    # Helper classes
    "RunDirectories",
]

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
    RunDirectories,
    SimulationConfig,
    Stopwatch,
    TrainerConfig,
    # Functions from rl.functions
    accumulate_rollout_stats,
    # Helper functions
    calculate_anneal_beta,
    compute_advantage,
    perform_rollout_step,
    process_minibatch_update,
    save_experiment_config,
    setup_run_directories,
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
    "save_experiment_config",
    "setup_run_directories",
    # Functions from rl.functions
    "accumulate_rollout_stats",
    "compute_advantage",
    "perform_rollout_step",
    "process_minibatch_update",
    # Helper classes
    "RunDirectories",
    # Version
    "__version__",
]

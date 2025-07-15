"""
Clean API for Metta - provides direct instantiation without Hydra.

This API exposes the core training components from Metta, allowing users to:
1. Create environments, agents, and training components programmatically
2. Use the same Pydantic configuration classes as the main codebase
3. Control the training loop directly with full visibility
4. Support distributed training with minimal setup
"""

# Re-export all components for backward compatibility
from metta.api.agent import Agent, _get_default_agent_config
from metta.api.checkpoint import (
    ensure_initial_policy,
    load_checkpoint,
    save_checkpoint,
    wrap_agent_distributed,
)
from metta.api.directories import RunDirectories, save_experiment_config, setup_run_directories
from metta.api.distributed import cleanup_distributed, setup_distributed_training
from metta.api.environment import (
    Environment,
    NavigationBucketedCurriculum,
    PreBuiltConfigCurriculum,
    _get_default_env_config,
)
from metta.api.evaluation import (
    create_evaluation_config_suite,
    create_replay_config,
    evaluate_policy_suite,
    generate_replay_simple,
)
from metta.api.optimizer import Optimizer
from metta.api.training import TrainerState, calculate_anneal_beta
from metta.api.wandb import cleanup_wandb, initialize_wandb

__all__ = [
    # Factory classes
    "Environment",
    "Agent",
    # Wrapper classes
    "Optimizer",
    # Helper functions unique to api
    "calculate_anneal_beta",
    "setup_run_directories",
    "save_experiment_config",
    "save_checkpoint",
    "setup_distributed_training",
    "cleanup_distributed",
    "initialize_wandb",
    "cleanup_wandb",
    "load_checkpoint",
    "wrap_agent_distributed",
    "ensure_initial_policy",
    # Helper classes
    "RunDirectories",
    "PreBuiltConfigCurriculum",
    "NavigationBucketedCurriculum",
    # Evaluation/replay configuration
    "create_evaluation_config_suite",
    "create_replay_config",
    "evaluate_policy_suite",
    "generate_replay_simple",
    # Training state management
    "TrainerState",
    # Private functions (not in __all__ but still available)
    "_get_default_env_config",
    "_get_default_agent_config",
]

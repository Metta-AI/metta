# metta/__init__.py
"""Metta: A reinforcement learning codebase for multi-agent gridworlds."""

# Import key API functions and configs for convenience
from metta.api import (
    # Factory classes (new API)
    Agent,
    # Configuration classes
    AgentModelConfig,
    CheckpointConfig,
    EnvConfig,
    Environment,
    ExperienceConfig,
    ExperienceManager,
    GameConfig,
    Optimizer,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    # Core functions
    compute_advantages,
    eval_policy,
    load_checkpoint,
    make_agent,
    make_environment,
    make_experience_manager,
    make_optimizer,
    rollout,
    save_checkpoint,
    train_ppo,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration classes
    "AgentModelConfig",
    "CheckpointConfig",
    "EnvConfig",
    "ExperienceConfig",
    "GameConfig",
    "OptimizerConfig",
    "PPOConfig",
    "SimulationConfig",
    # Factory classes (new API)
    "Agent",
    "Environment",
    "ExperienceManager",
    "Optimizer",
    # Core functions
    "compute_advantages",
    "eval_policy",
    "load_checkpoint",
    "rollout",
    "save_checkpoint",
    "train_ppo",
    # Deprecated functions (kept for backward compatibility)
    "make_agent",
    "make_environment",
    "make_experience_manager",
    "make_optimizer",
]

# metta/__init__.py
"""Metta: A reinforcement learning codebase for multi-agent gridworlds."""

# Import key API functions and configs for convenience
from metta.api import (
    # Configuration classes
    AgentModelConfig,
    CheckpointConfig,
    EnvConfig,
    ExperienceConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    # Core functions
    compute_advantages,
    eval_policy,
    load_checkpoint,
    make_agent,
    make_curriculum,
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
    "OptimizerConfig",
    "PPOConfig",
    "SimulationConfig",
    # Core functions
    "compute_advantages",
    "eval_policy",
    "load_checkpoint",
    "make_agent",
    "make_curriculum",
    "make_environment",
    "make_experience_manager",
    "make_optimizer",
    "rollout",
    "save_checkpoint",
    "train_ppo",
]

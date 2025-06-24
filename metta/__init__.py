"""Metta - Multi-agent reinforcement learning library."""

# Core runtime configuration
# Agent classes and factory
from .agent import (
    AttentionAgent,
    BaseAgent,
    LargeCNNAgent,
    MultiHeadAttentionAgent,
    SimpleCNNAgent,
    create_agent,
    list_agents,
    register_agent,
)

# Environment creation
from .env import (
    ENV_PRESETS,
    create_curriculum_env,
    create_env,
    create_env_from_preset,
    create_vectorized_env,
)
from .runtime import RuntimeConfig, configure, get_runtime, set_runtime

# Simulation and evaluation
from .sim.registry import (
    SimulationRegistry,
    SimulationSpec,
    get_simulation_suite,
    register_default_simulations,
    register_simulation,
)
from .train.config import AgentConfig, TrainerConfig, TrainingConfig, WandbConfig

# Training interfaces
from .train.job import JobBuilder, TrainingJob, quick_train
from .train.minimal import Experiment, Metta, train

__all__ = [
    # Runtime
    "RuntimeConfig",
    "configure",
    "get_runtime",
    "set_runtime",
    # Agents
    "BaseAgent",
    "SimpleCNNAgent",
    "LargeCNNAgent",
    "AttentionAgent",
    "MultiHeadAttentionAgent",
    "create_agent",
    "register_agent",
    "list_agents",
    # Environments
    "create_env",
    "create_env_from_preset",
    "create_vectorized_env",
    "create_curriculum_env",
    "ENV_PRESETS",
    # Training - Job-based
    "TrainingJob",
    "JobBuilder",
    "quick_train",
    "TrainingConfig",
    "TrainerConfig",
    "AgentConfig",
    "WandbConfig",
    # Training - Minimal
    "Metta",
    "train",
    "Experiment",
    # Simulations
    "SimulationRegistry",
    "SimulationSpec",
    "register_simulation",
    "get_simulation_suite",
    "register_default_simulations",
]

# Register default simulations on import
register_default_simulations()

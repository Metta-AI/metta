# metta/__init__.py
"""Metta: A reinforcement learning codebase for multi-agent gridworlds."""

# Import key API functions and configs for convenience
from metta.api import (
    # Object type constants
    TYPE_AGENT,
    TYPE_ALTAR,
    TYPE_ARMORY,
    TYPE_FACTORY,
    TYPE_GENERATOR_BLUE,
    TYPE_GENERATOR_GREEN,
    TYPE_GENERATOR_RED,
    TYPE_GENERIC_CONVERTER,
    TYPE_LAB,
    TYPE_LASERY,
    TYPE_MINE_BLUE,
    TYPE_MINE_GREEN,
    TYPE_MINE_RED,
    TYPE_TEMPLE,
    TYPE_WALL,
    # Factory classes
    Agent,
    # Configuration classes (Pydantic models)
    CheckpointConfig,
    Environment,
    KickstartConfig,
    OptimizerConfig,
    PPOConfig,
    PrioritizedExperienceReplayConfig,
    SimulationConfig,
    TrainerConfig,
    TrainingComponents,
    VTraceConfig,
    # Helper functions
    calculate_anneal_beta,
    create_default_trainer_config,
    evaluate_policy,
    load_checkpoint,
    save_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    # Factory classes
    "Agent",
    "Environment",
    "TrainingComponents",
    # Configuration classes
    "CheckpointConfig",
    "KickstartConfig",
    "OptimizerConfig",
    "PPOConfig",
    "PrioritizedExperienceReplayConfig",
    "SimulationConfig",
    "TrainerConfig",
    "VTraceConfig",
    # Helper functions
    "calculate_anneal_beta",
    "create_default_trainer_config",
    "evaluate_policy",
    "load_checkpoint",
    "save_checkpoint",
    # Object type constants
    "TYPE_AGENT",
    "TYPE_ALTAR",
    "TYPE_ARMORY",
    "TYPE_FACTORY",
    "TYPE_GENERATOR_BLUE",
    "TYPE_GENERATOR_GREEN",
    "TYPE_GENERATOR_RED",
    "TYPE_LAB",
    "TYPE_LASERY",
    "TYPE_MINE_BLUE",
    "TYPE_MINE_GREEN",
    "TYPE_MINE_RED",
    "TYPE_TEMPLE",
    "TYPE_WALL",
    "TYPE_GENERIC_CONVERTER",
]

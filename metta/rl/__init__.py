"""Modular RL components for flexible training."""

# Core components
from metta.rl.checkpointer import AutoCheckpointer, TrainingCheckpointer
from metta.rl.collectors import RolloutCollector
from metta.rl.configs import ExperienceConfig, PPOConfig, TrainerConfig
from metta.rl.evaluator import PolicyEvaluator
from metta.rl.experience import Experience

# Functional training
from metta.rl.functional_trainer import (
    TrainingState,
    create_training_components,
    default_training_step,
    functional_training_loop,
)

# Utilities
from metta.rl.losses import Losses
from metta.rl.optimizers import PPOOptimizer
from metta.rl.stats_logger import StatsLogger

# Legacy trainer (for backward compatibility)
from metta.rl.trainer import LegacyMettaTrainer, MettaTrainer
from metta.rl.trainer_checkpoint import TrainerCheckpoint

__all__ = [
    # Core components
    "RolloutCollector",
    "PPOOptimizer",
    "PolicyEvaluator",
    "TrainingCheckpointer",
    "AutoCheckpointer",
    "StatsLogger",
    "Experience",
    # Configs
    "PPOConfig",
    "TrainerConfig",
    "ExperienceConfig",
    # Functional training
    "TrainingState",
    "create_training_components",
    "default_training_step",
    "functional_training_loop",
    # Legacy
    "MettaTrainer",
    "LegacyMettaTrainer",
    # Utilities
    "Losses",
    "TrainerCheckpoint",
]

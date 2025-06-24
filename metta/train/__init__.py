"""Training interfaces for Metta."""

# Configuration
from .config import (
    AgentConfig,
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
    WandbConfig,
)

# Training jobs
from .job import (
    JobBuilder,
    TrainingJob,
    quick_train,
)

# Minimal interface
from .minimal import (
    Experiment,
    Metta,
    train,
)

# Simple trainer
from .simple_trainer import (
    SimpleTrainer,
    train_agent,
)

__all__ = [
    # Configuration
    "TrainingConfig",
    "TrainerConfig",
    "AgentConfig",
    "OptimizerConfig",
    "WandbConfig",
    # Jobs
    "TrainingJob",
    "JobBuilder",
    "quick_train",
    # Minimal
    "Metta",
    "train",
    "Experiment",
    # Simple
    "SimpleTrainer",
    "train_agent",
]

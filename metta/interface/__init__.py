"""
metta.interface package - provides a clean interface for Metta training components.

This package contains modular components for training Metta agents:
- agent.py: Agent creation and loading
- directories.py: Run directory and distributed setup utilities
- environment.py: Environment creation and curriculum helpers
- evaluation.py: Policy evaluation and replay generation
- training.py: Training utilities and helpers
"""

from metta.interface.agent import Agent
from metta.interface.directories import RunDirectories, setup_run_directories
from metta.interface.environment import Environment, PreBuiltConfigCurriculum
from metta.interface.evaluation import (
    create_evaluation_config_suite,
    create_replay_config,
    evaluate_policy_suite,
    generate_replay_simple,
)
from metta.interface.training import (
    Optimizer,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    # Agent
    "Agent",
    # Directories
    "RunDirectories",
    "setup_run_directories",
    # Environment
    "Environment",
    "PreBuiltConfigCurriculum",
    # Evaluation
    "create_evaluation_config_suite",
    "create_replay_config",
    "evaluate_policy_suite",
    "generate_replay_simple",
    # Training
    "Optimizer",
    "save_checkpoint",
    "load_checkpoint",
]

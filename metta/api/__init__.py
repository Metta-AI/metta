"""
metta.api package - provides a clean API for Metta training components.

This package contains modular components for training Metta agents:
- agent.py: Agent creation and loading
- directories.py: Run directory and distributed setup utilities
- environment.py: Environment creation and curriculum helpers
- evaluation.py: Policy evaluation and replay generation
- training.py: Training utilities and helpers
"""

from metta.api.agent import Agent
from metta.api.directories import RunDirectories, setup_device_and_distributed, setup_run_directories
from metta.api.environment import Environment, PreBuiltConfigCurriculum
from metta.api.evaluation import (
    create_evaluation_config_suite,
    create_replay_config,
    evaluate_policy_suite,
    generate_replay_simple,
)
from metta.api.training import Optimizer, load_checkpoint, save_checkpoint

__all__ = [
    # Agent
    "Agent",
    # Directories
    "RunDirectories",
    "setup_run_directories",
    "setup_device_and_distributed",
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

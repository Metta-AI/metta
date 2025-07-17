"""
Clean API for Metta - provides direct instantiation without Hydra.

This API exposes the core training components from Metta, allowing users to:
1. Create environments, agents, and training components programmatically
2. Use the same Pydantic configuration classes as the main codebase
3. Control the training loop directly with full visibility
4. Support distributed training with minimal setup
"""

# Re-export all components for backward compatibility
from metta.api.agent import Agent
from metta.api.directories import RunDirectories, save_experiment_config, setup_run_directories
from metta.api.environment import Environment, NavigationBucketedCurriculum, PreBuiltConfigCurriculum
from metta.api.evaluation import create_evaluation_config_suite, create_replay_config
from metta.rl.functions import wrap_agent_distributed

__all__ = [
    # Factory classes
    "Environment",
    "Agent",
    # Helper functions unique to api.py
    "setup_run_directories",
    "save_experiment_config",
    # Helper classes
    "RunDirectories",
    "NavigationBucketedCurriculum",
    "PreBuiltConfigCurriculum",
    # Evaluation/replay configuration
    "create_evaluation_config_suite",
    "create_replay_config",
    # Re-exported from functions
    "wrap_agent_distributed",
]

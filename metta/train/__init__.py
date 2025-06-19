"""Functional training API for Metta.

This module provides a functional interface for training RL agents,
breaking down the monolithic MettaTrainer into composable functions.
"""

from .checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from .evaluate import evaluate_policy
from .losses import PPOLossConfig, compute_ppo_loss
from .rollout import RolloutConfig, rollout
from .update import OptimizerConfig, update_policy

__all__ = [
    "rollout",
    "RolloutConfig",
    "compute_ppo_loss",
    "PPOLossConfig",
    "update_policy",
    "OptimizerConfig",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "evaluate_policy",
]

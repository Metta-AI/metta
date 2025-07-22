"""
metta.interface package - provides a clean interface for Metta training components.

This package contains modular components for training Metta agents:
- agent.py: Agent creation and loading
- directories.py: Run directory and distributed setup utilities
- environment.py: Environment creation and curriculum helpers
- evaluation.py: Policy evaluation and replay generation
- training.py: Training utilities and helpers
"""

from typing import Any, Optional, Tuple

import torch

from metta.interface.agent import Agent
from metta.interface.directories import RunDirectories, setup_run_directories
from metta.interface.environment import Environment, PreBuiltConfigCurriculum
from metta.interface.evaluation import (
    create_evaluation_config_suite,
    create_replay_config,
    evaluate_policy_suite,
    generate_replay_simple,
)
from metta.interface.optimizer import Optimizer

# Import checkpoint functions from rl.util modules
from metta.rl.util.distributed import cleanup_distributed
from metta.rl.util.policy_management import (
    ensure_initial_policy,
    save_policy_with_metadata,
)


# Wrapper functions to maintain documented API
def save_checkpoint(
    epoch: int,
    agent_step: int,
    agent: torch.nn.Module,
    optimizer: Any,
    policy_store: Any,
    checkpoint_path: str,
    checkpoint_interval: int,
    stats: Optional[dict] = None,
    force_save: bool = False,
) -> Optional[Any]:
    """Save checkpoint if epoch matches checkpoint_interval (or force_save).

    This is a wrapper around save_policy_with_metadata that maintains
    the documented interface API.
    """
    if not force_save and epoch % checkpoint_interval != 0:
        return None

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None  # Non-master ranks skip

    # Call the underlying function with adapted parameters
    return save_policy_with_metadata(
        policy=agent,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals=stats or {},
        timer=None,
        initial_policy_record=None,
        run_name="",
        is_master=True,
    )


def load_checkpoint(
    checkpoint_dir: str,
    agent: torch.nn.Module,
    optimizer: Optional[Any],
    policy_store: Any,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, Optional[str]]:
    """Load checkpoint returning (agent_step, epoch, policy_path).

    This is a wrapper around maybe_load_checkpoint that maintains
    the documented interface API.
    """
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    # Load trainer checkpoint directly for simple API
    checkpoint = TrainerCheckpoint.load(checkpoint_dir)
    if checkpoint is None:
        return 0, 0, None

    # Restore optimizer state if possible
    if optimizer and checkpoint.optimizer_state_dict:
        try:
            if hasattr(optimizer, "optimizer"):
                optimizer.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            else:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        except ValueError:
            pass  # Ignore mismatch

    return checkpoint.agent_step, checkpoint.epoch, checkpoint.policy_path


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
    "cleanup_distributed",
    "ensure_initial_policy",
]

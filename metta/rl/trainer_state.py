from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TrainerState:
    """Lightweight, fast, mutable container for training loop state. Also provides a way for losses to influence
    trainer's control flow."""

    agent_step: int = 0
    epoch: int = 0
    update_epoch: int = 0
    mb_idx: int = 0
    num_mbs: int = 0
    optimizer: torch.optim.Optimizer | None = None

    # Control flags and scratch metrics
    early_stop_update_epoch: bool = False

"""Policy update functions."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "adam"  # "adam" or "muon"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5


def update_policy(
    agent,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    config: OptimizerConfig,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Update policy weights using computed loss.

    Args:
        agent: The policy/agent to update
        optimizer: Optimizer instance
        loss: Computed loss to backpropagate
        config: Optimizer configuration
        lr_scheduler: Optional learning rate scheduler
    """
    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Gradient clipping
    if config.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)

    # Optimizer step
    optimizer.step()

    # Learning rate scheduling
    if lr_scheduler is not None:
        lr_scheduler.step()


def create_optimizer(
    agent,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Create optimizer for the agent.

    Args:
        agent: The policy/agent to optimize
        config: Optimizer configuration

    Returns:
        Optimizer instance
    """
    if config.type == "adam":
        return torch.optim.Adam(
            agent.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.type == "muon":
        # Import here to avoid dependency if not using muon
        from heavyball import ForeachMuon

        return ForeachMuon(
            agent.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps

    Returns:
        LR scheduler instance or None
    """
    if warmup_steps > 0:
        # Linear warmup then cosine decay
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Just cosine annealing
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

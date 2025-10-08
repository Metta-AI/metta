import torch
from heavyball import ForeachMuon
import schedulefree

from metta.agent.policy import Policy
from metta.rl.trainer_config import OptimizerConfig


def create_optimizer(cfg: OptimizerConfig, policy: Policy) -> torch.optim.Optimizer:
    """Create optimizer and load state if available."""
    optimizer_type = cfg.type

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    elif optimizer_type == "muon":
        optimizer = ForeachMuon(
            policy.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=int(cfg.weight_decay),
        )
    elif optimizer_type == "adamw_schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            policy.parameters(),
            lr=0.3,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
        )
    elif optimizer_type == "sgd_schedulefree":
        optimizer = schedulefree.SGDScheduleFree(
            policy.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
        )
    else:
        raise ValueError(f"Optimizer type must be one of 'adam', 'muon', 'adamw_schedulefree', 'sgd_schedulefree', got {optimizer_type}")

    # # Load optimizer state if available
    # if trainer_state and "optimizer_state" in trainer_state:
    #     try:
    #         optimizer.load_state_dict(trainer_state["optimizer_state"])
    #         logger.info("Successfully loaded optimizer state from checkpoint")
    #     except ValueError:
    #         logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

    # For ScheduleFree optimizers, put them in train mode immediately

    return optimizer


def is_schedulefree_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    """Check if optimizer is a ScheduleFree optimizer that requires train()/eval() calls."""
    # ScheduleFree optimizers have 'train_mode' in their param_groups
    return (hasattr(optimizer, 'train') and hasattr(optimizer, 'eval') and
            len(optimizer.param_groups) > 0 and 'train_mode' in optimizer.param_groups[0])

import torch
from heavyball import ForeachMuon

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
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
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
    else:
        raise ValueError(f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}")

    # # Load optimizer state if available
    # if trainer_state and "optimizer_state" in trainer_state:
    #     try:
    #         optimizer.load_state_dict(trainer_state["optimizer_state"])
    #         logger.info("Successfully loaded optimizer state from checkpoint")
    #     except ValueError:
    #         logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

    return optimizer

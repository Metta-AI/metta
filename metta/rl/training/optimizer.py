import logging

from typing import Any, Callable

import schedulefree
import torch
from heavyball import ForeachMuon, ForeachPSGDKron
from heavyball import utils as heavyball_utils

from metta.agent.policy import Policy
from metta.rl.trainer_config import OptimizerConfig

logger = logging.getLogger(__name__)


def _patch_heavyball_control_flow() -> None:
    # HeavyBall 1.7.2 ships broken `cond` / `while_loop` wrappers (wrong call signature),
    # which breaks PSGD Kron under `torch.compile`.
    def cond(pred: torch.Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> Any:
        if torch.compiler.is_compiling():
            return torch.cond(pred, true_fn, false_fn)
        if pred.item():
            return true_fn()
        return false_fn()

    def while_loop(cond_fn: Callable[..., torch.Tensor], body_fn: Callable[..., Any], carried_inputs: Any) -> Any:
        if torch.compiler.is_compiling():
            return torch.while_loop(cond_fn, body_fn, carried_inputs)
        while cond_fn(*carried_inputs).item():
            carried_inputs = body_fn(*carried_inputs)
        return carried_inputs

    heavyball_utils.cond = cond
    heavyball_utils.while_loop = while_loop


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
    elif optimizer_type == "kron":
        _patch_heavyball_control_flow()
        optimizer = ForeachPSGDKron(
            policy.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            preconditioner_update_probability=cfg.preconditioner_update_probability,
            max_size_triangular=cfg.max_size_triangular,
            min_ndim_triangular=cfg.min_ndim_triangular,
            memory_save_mode=cfg.memory_save_mode,
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
            lr=cfg.learning_rate,
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
        allowed_types = ("adam", "kron", "muon", "adamw_schedulefree", "sgd_schedulefree")
        raise ValueError(f"Optimizer type must be one of {allowed_types}, got {optimizer_type}")

    # # Load optimizer state if available
    # if trainer_state and "optimizer_state" in trainer_state:
    #     try:
    #         optimizer.load_state_dict(trainer_state["optimizer_state"])
    #         logger.info("Successfully loaded optimizer state from checkpoint")
    #     except ValueError:
    #         logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

    # Note: For ScheduleFree optimizers, we don't call train() here.
    # The trainer will call train() before the first training phase.
    # Calling train() too early can interfere with optimizer state initialization.

    return optimizer


def is_schedulefree_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    """Check if optimizer is a ScheduleFree optimizer that requires train()/eval() calls."""
    # ScheduleFree optimizers have train()/eval() methods and train_mode either as:
    if not (hasattr(optimizer, "train") and hasattr(optimizer, "eval")):
        return False

    if hasattr(optimizer, "train_mode"):
        return True

    if len(optimizer.param_groups) > 0 and "train_mode" in optimizer.param_groups[0]:
        return True

    return False

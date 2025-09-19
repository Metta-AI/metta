"""Hyperparameter scheduling management."""

import logging
import math

from metta.rl.training.component import TrainerComponent
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class HyperparameterSchedulerConfig(Config):
    """Scheduler settings applied during training."""

    enabled: bool = False
    schedule_type: str = "exponential"
    learning_rate_decay: float = 1.0
    ppo_clip_decay: float = 1.0
    ppo_ent_coef_decay: float = 1.0


class SchedulerConfig(Config):
    """Component-specific scheduling configuration."""

    interval: int = 1
    """How often to update hyperparameters (in epochs)."""


def _decay_value(initial: float, decay_rate: float, progress: float, schedule_type: str = "exponential") -> float:
    if schedule_type == "cosine":
        return initial * (1 + math.cos(math.pi * progress)) / 2
    if schedule_type == "linear":
        return initial * (1 - progress)
    return initial * (decay_rate**progress)


def step_hyperparameters(trainer_cfg, optimizer, current_step: int, total_timesteps: int, log) -> dict[str, float]:
    cfg = trainer_cfg.hyperparameter_scheduler

    if not getattr(cfg, "enabled", False):
        return {}

    progress = min(current_step / max(total_timesteps, 1), 1.0)
    updates: dict[str, float] = {}

    if cfg.learning_rate_decay < 1.0:
        new_lr = _decay_value(
            trainer_cfg.optimizer.learning_rate,
            cfg.learning_rate_decay,
            progress,
            cfg.schedule_type,
        )
        optimizer.param_groups[0]["lr"] = new_lr
        updates["learning_rate"] = new_lr

    if cfg.ppo_clip_decay < 1.0:
        updates["ppo_clip_coef"] = _decay_value(
            trainer_cfg.ppo.clip_coef,
            cfg.ppo_clip_decay,
            progress,
            cfg.schedule_type,
        )

    if cfg.ppo_ent_coef_decay < 1.0:
        updates["ppo_ent_coef"] = _decay_value(
            trainer_cfg.ppo.ent_coef,
            cfg.ppo_ent_coef_decay,
            progress,
            cfg.schedule_type,
        )

    if updates and current_step % 10000 == 0:
        params = ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
        log.info("Hyperparameter updates at step %s: %s", current_step, params)

    return updates


class Scheduler(TrainerComponent):
    """Manages hyperparameter scheduling."""

    def __init__(self, config: SchedulerConfig):
        """Initialize scheduler component.

        Args:
            config: Scheduler configuration.
        """
        super().__init__(epoch_interval=config.interval)

    def on_epoch_end(self, epoch: int) -> None:
        """Update hyperparameters for the current training epoch."""
        context = self.context
        trainer_cfg = context.config

        if not getattr(trainer_cfg.hyperparameter_scheduler, "enabled", False):
            return

        # Update learning rate and other hyperparameters across ranks
        step_hyperparameters(
            trainer_cfg,
            context.optimizer,
            context.agent_step,
            trainer_cfg.total_timesteps,
            logger,
        )

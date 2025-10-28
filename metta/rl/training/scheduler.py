"""Hyperparameter scheduling management."""

import logging
import math

from metta.rl.training import TrainerComponent
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
        self._step_hyperparameters(
            trainer_cfg,
            context.optimizer,
            context.agent_step,
            trainer_cfg.total_timesteps,
            logger,
        )

    @staticmethod
    def _decay_value(initial: float, decay_rate: float, progress: float, schedule_type: str) -> float:
        if schedule_type == "cosine":
            return initial * (1 + math.cos(math.pi * progress)) / 2
        if schedule_type == "linear":
            return initial * (1 - progress)
        return initial * (decay_rate**progress)

    @staticmethod
    def _get_ppo_config(trainer_cfg):
        losses_cfg = getattr(trainer_cfg, "losses", None)
        loss_configs = getattr(losses_cfg, "loss_configs", None)
        if isinstance(loss_configs, dict):
            return loss_configs.get("ppo")
        return None

    @classmethod
    def _step_hyperparameters(
        cls, trainer_cfg, optimizer, current_step: int, total_timesteps: int, log
    ) -> dict[str, float]:
        cfg = trainer_cfg.hyperparameter_scheduler

        if not getattr(cfg, "enabled", False):
            return {}

        progress = min(current_step / max(total_timesteps, 1), 1.0)
        updates: dict[str, float] = {}

        if cfg.learning_rate_decay < 1.0:
            base_lr = getattr(cfg, "_base_learning_rate", None)
            if base_lr is None:
                base_lr = trainer_cfg.optimizer.learning_rate
                cfg._base_learning_rate = base_lr
            new_lr = cls._decay_value(base_lr, cfg.learning_rate_decay, progress, cfg.schedule_type)
            optimizer.param_groups[0]["lr"] = new_lr
            trainer_cfg.optimizer.learning_rate = new_lr
            updates["learning_rate"] = new_lr

        ppo_cfg = cls._get_ppo_config(trainer_cfg)
        if ppo_cfg is None and (cfg.ppo_clip_decay < 1.0 or cfg.ppo_ent_coef_decay < 1.0):
            log.debug("Hyperparameter scheduler could not locate PPO config; skipping PPO decay updates")

        if ppo_cfg is not None and cfg.ppo_clip_decay < 1.0 and hasattr(ppo_cfg, "clip_coef"):
            base_clip = getattr(cfg, "_base_ppo_clip_coef", None)
            if base_clip is None:
                base_clip = ppo_cfg.clip_coef
                cfg._base_ppo_clip_coef = base_clip
            new_clip = cls._decay_value(base_clip, cfg.ppo_clip_decay, progress, cfg.schedule_type)
            ppo_cfg.clip_coef = new_clip
            updates["ppo_clip_coef"] = new_clip

        if ppo_cfg is not None and cfg.ppo_ent_coef_decay < 1.0 and hasattr(ppo_cfg, "ent_coef"):
            base_ent = getattr(cfg, "_base_ppo_ent_coef", None)
            if base_ent is None:
                base_ent = ppo_cfg.ent_coef
                cfg._base_ppo_ent_coef = base_ent
            new_ent = cls._decay_value(base_ent, cfg.ppo_ent_coef_decay, progress, cfg.schedule_type)
            ppo_cfg.ent_coef = new_ent
            updates["ppo_ent_coef"] = new_ent

        if updates and current_step % 10000 == 0:
            params = ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
            log.info("Hyperparameter updates at step %s: %s", current_step, params)

        return updates

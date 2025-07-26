import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import hydra
from omegaconf import DictConfig


class BaseSchedule(ABC):
    """Base class for scheduling strategies."""

    def __init__(
        self, initial_value: float, min_value: Optional[float] = None, max_value: Optional[float] = None, **kwargs
    ):
        self.initial_value = initial_value
        self.min_value = min_value if min_value is not None else initial_value * 0.1
        self.max_value = max_value if max_value is not None else initial_value

    @abstractmethod
    def __call__(self, progress: float) -> float:
        """Compute scheduled value given progress (0.0 to 1.0)."""
        pass


class ConstantSchedule(BaseSchedule):
    """Constant schedule that returns the initial value."""

    def __call__(self, progress: float) -> float:
        return self.initial_value


class LinearSchedule(BaseSchedule):
    def __call__(self, progress: float) -> float:
        return self.initial_value + (self.min_value - self.initial_value) * progress


class CosineSchedule(BaseSchedule):
    def __call__(self, progress: float) -> float:
        return self.min_value + (self.initial_value - self.min_value) * (1 + math.cos(math.pi * progress)) / 2


class ExponentialSchedule(BaseSchedule):
    def __init__(self, initial_value: float, decay_rate: float = 0.95, **kwargs):
        super().__init__(initial_value, **kwargs)
        self.decay_rate = decay_rate

    def __call__(self, progress: float) -> float:
        value = self.initial_value * (self.decay_rate**progress)
        return max(value, self.min_value)


class LogarithmicSchedule(BaseSchedule):
    def __init__(self, initial_value: float, decay_rate: float = 0.1, **kwargs):
        super().__init__(initial_value, **kwargs)
        self.decay_rate = decay_rate

    def __call__(self, progress: float) -> float:
        if progress == 0:
            return self.initial_value
        log_progress = math.log1p(self.decay_rate * progress) / math.log1p(self.decay_rate)
        return self.initial_value + (self.min_value - self.initial_value) * log_progress


class HyperparameterScheduler:
    def __init__(
        self,
        initial_values: dict[str, float],
        schedule_configs: dict[str, dict | None],
        optimizer,
        total_timesteps: int,
        logger=None,
    ):
        """Initialize the hyperparameter scheduler with explicit parameters.

        Args:
            initial_values: Dict mapping parameter names to their initial values
            schedule_configs: Dict mapping parameter names to their schedule configs (or None for constant)
            optimizer: The optimizer to update learning rate on
            total_timesteps: Total timesteps for progress calculation
            logger: Logger instance (optional)
        """
        self.initial_values = initial_values
        self.optimizer = optimizer
        self.total_timesteps = total_timesteps
        self.logger = logger or logging.getLogger(__name__)

        # Map from parameter name to schedule config key
        self.schedule_key_mapping = {
            "learning_rate": "learning_rate_schedule",
            "ppo_clip_coef": "ppo_clip_schedule",
            "ppo_vf_clip_coef": "ppo_vf_clip_schedule",
            "ppo_ent_coef": "ppo_ent_coef_schedule",
            "ppo_l2_reg_loss_coef": "ppo_l2_reg_loss_schedule",
            "ppo_l2_init_loss_coef": "ppo_l2_init_loss_schedule",
        }

        self.schedulers = {}

        for param_name, initial_value in initial_values.items():
            schedule_config = schedule_configs.get(param_name)
            if schedule_config is not None:
                self.logger.info(f"Initializing scheduler for: {param_name}")
                self.schedulers[param_name] = hydra.utils.instantiate(schedule_config)
            else:
                self.schedulers[param_name] = ConstantSchedule(initial_value)

    @staticmethod
    def from_trainer_config(trainer_cfg: DictConfig, optimizer, total_timesteps: int, logger=None):
        """Factory method to create HyperparameterScheduler from trainer config.

        Args:
            trainer_cfg: The trainer configuration
            optimizer: The optimizer to update learning rate on
            total_timesteps: Total timesteps for progress calculation
            logger: Logger instance (optional)
        """
        # Extract initial values from trainer config
        initial_values = {
            "learning_rate": trainer_cfg.optimizer.learning_rate,
            "ppo_clip_coef": trainer_cfg.ppo.clip_coef,
            "ppo_vf_clip_coef": trainer_cfg.ppo.vf_clip_coef,
            "ppo_ent_coef": trainer_cfg.ppo.ent_coef,
            "ppo_l2_reg_loss_coef": trainer_cfg.ppo.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": trainer_cfg.ppo.l2_init_loss_coef,
        }

        # Extract schedule configs
        scheduler_cfg = trainer_cfg.hyperparameter_scheduler
        schedule_configs = {}

        # Map parameter names to their schedule config keys
        schedule_key_mapping = {
            "learning_rate": "learning_rate_schedule",
            "ppo_clip_coef": "ppo_clip_schedule",
            "ppo_vf_clip_coef": "ppo_vf_clip_schedule",
            "ppo_ent_coef": "ppo_ent_coef_schedule",
            "ppo_l2_reg_loss_coef": "ppo_l2_reg_loss_schedule",
            "ppo_l2_init_loss_coef": "ppo_l2_init_loss_schedule",
        }

        for param_name, schedule_key in schedule_key_mapping.items():
            schedule_config = getattr(scheduler_cfg, schedule_key, None)
            schedule_configs[param_name] = schedule_config

        return HyperparameterScheduler(
            initial_values=initial_values,
            schedule_configs=schedule_configs,
            optimizer=optimizer,
            total_timesteps=total_timesteps,
            logger=logger,
        )

    def _compute_scheduled_value(self, param_name: str, current_step: int) -> float:
        """Compute the scheduled value for a hyperparameter at the current step."""
        schedule_fn = self.schedulers[param_name]

        progress = min(current_step / self.total_timesteps, 1.0)
        return schedule_fn(progress)

    def step(self, current_step: int, update_callbacks: dict[str, callable] = None) -> dict[str, float]:
        """Update hyperparameters for the current step.

        Args:
            current_step: Current training step
            update_callbacks: Optional dict of callbacks to update values (e.g., trainer_cfg setters)

        Returns:
            Dict of updated parameter values
        """
        if not self.schedulers:
            return {}  # No schedulers configured

        updates = {}
        for param_name in self.schedulers.keys():
            new_value = self._compute_scheduled_value(param_name, current_step)
            updates[param_name] = new_value

        # Update optimizer learning rate directly
        if "learning_rate" in updates:
            self.optimizer.param_groups[0]["lr"] = updates["learning_rate"]

        # Call update callbacks if provided
        if update_callbacks:
            for param_name, new_value in updates.items():
                if param_name in update_callbacks:
                    update_callbacks[param_name](new_value)

        if current_step % 10000 == 0 and updates:
            self.logger.info(
                f"Step {current_step}: Updated hyperparameters: "
                + ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
            )

        return updates

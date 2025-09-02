import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional


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
        schedule_configs: dict[str, Any | None],
        optimizer,
        total_timesteps: int,
        logger=None,
    ):
        self.optimizer = optimizer
        self.total_timesteps = total_timesteps
        self.logger = logger or logging.getLogger(__name__)

        self.enabled = any(config is not None for config in schedule_configs.values())
        if not self.enabled:
            self.logger.info("Hyperparameter scheduling disabled")
            return

        self.schedulers = {}
        schedule_classes = {
            "constant": ConstantSchedule,
            "linear": LinearSchedule,
            "cosine": CosineSchedule,
            "exponential": ExponentialSchedule,
            "logarithmic": LogarithmicSchedule,
        }

        for param_name, initial_value in initial_values.items():
            config = schedule_configs.get(param_name)
            if config is None:
                continue

            schedule_cls = schedule_classes.get(config.type, ConstantSchedule)
            if config.type in ["exponential", "logarithmic"]:
                scheduler = schedule_cls(initial_value, decay_rate=config.decay_rate, min_value=config.min_value)
            elif config.type in ["linear", "cosine"]:
                scheduler = schedule_cls(initial_value, min_value=config.min_value)
            else:
                scheduler = schedule_cls(initial_value)

            self.schedulers[param_name] = scheduler

        if self.schedulers:
            self.logger.info(f"Enabled scheduling: {', '.join(self.schedulers.keys())}")

    @staticmethod
    def from_trainer_config(trainer_cfg, optimizer, total_timesteps: int, logger=None):
        initial_values = {
            "learning_rate": trainer_cfg.optimizer.learning_rate,
            "ppo_clip_coef": trainer_cfg.ppo.clip_coef,
            "ppo_vf_clip_coef": trainer_cfg.ppo.vf_clip_coef,
            "ppo_ent_coef": trainer_cfg.ppo.ent_coef,
            "ppo_l2_reg_loss_coef": trainer_cfg.ppo.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": trainer_cfg.ppo.l2_init_loss_coef,
        }

        cfg = trainer_cfg.hyperparameter_scheduler
        schedule_configs = {
            "learning_rate": cfg.learning_rate_schedule,
            "ppo_clip_coef": cfg.ppo_clip_schedule,
            "ppo_vf_clip_coef": cfg.ppo_vf_clip_schedule,
            "ppo_ent_coef": cfg.ppo_ent_coef_schedule,
            "ppo_l2_reg_loss_coef": cfg.ppo_l2_reg_loss_schedule,
            "ppo_l2_init_loss_coef": cfg.ppo_l2_init_loss_schedule,
        }

        return HyperparameterScheduler(initial_values, schedule_configs, optimizer, total_timesteps, logger)

    def _compute_scheduled_value(self, param_name: str, current_step: int) -> float:
        progress = min(current_step / max(self.total_timesteps, 1), 1.0)
        return self.schedulers[param_name](progress)

    def step(self, current_step: int, update_callbacks: dict[str, callable] = None) -> dict[str, float]:
        if not self.enabled:
            return {}

        updates = {name: self._compute_scheduled_value(name, current_step) for name in self.schedulers}

        if "learning_rate" in updates:
            self.optimizer.param_groups[0]["lr"] = updates["learning_rate"]

        if update_callbacks:
            for param_name, new_value in updates.items():
                if param_name in update_callbacks:
                    update_callbacks[param_name](new_value)

        if current_step % 10000 == 0 and updates:
            params_str = ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
            self.logger.info(f"Step {current_step}: {params_str}")

        return updates

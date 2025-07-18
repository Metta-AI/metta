import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

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
    scheduler_registry: Dict[str, Callable[..., BaseSchedule]] = {
        "constant": ConstantSchedule,
        "linear": LinearSchedule,
        "cosine": CosineSchedule,
        "exponential": ExponentialSchedule,
        "logarithmic": LogarithmicSchedule,
    }

    def __init__(self, trainer_cfg: DictConfig, trainer, total_timesteps: int, logging):
        """Initialize the hyperparameter scheduler with configuration and total timesteps."""
        self.trainer_cfg = trainer_cfg
        self.trainer = trainer
        self.total_timesteps = total_timesteps
        self.logger = logging.getLogger(__name__)

        # key: attribute in trainer_cfg.hyperparameter_scheduler
        self.schedule_keys = {
            "learning_rate": "learning_rate_schedule",
            "ppo_clip_coef": "ppo_clip_schedule",
            "ppo_vf_clip_coef": "ppo_vf_clip_schedule",
            "ppo_ent_coef": "ppo_ent_coef_schedule",
            "ppo_l2_reg_loss_coef": "ppo_l2_reg_loss_schedule",
            "ppo_l2_init_loss_coef": "ppo_l2_init_loss_schedule",
        }

        self.schedulers = {}

        scheduler_cfg = trainer_cfg.hyperparameter_scheduler

        for param_name, key in self.schedule_keys.items():
            schedule_cfg = getattr(scheduler_cfg, key, None)
            if schedule_cfg is not None:
                self.logger.info(f"Initializing scheduler for: {param_name}")
                self.schedulers[param_name] = hydra.utils.instantiate(schedule_cfg)
            else:
                initial_value = self.trainer.hyperparameters[param_name]
                self.schedulers[param_name] = ConstantSchedule(initial_value)

    def _compute_scheduled_value(self, param_name: str, current_step: int) -> float:
        """Compute the scheduled value for a hyperparameter at the current step."""
        schedule_fn = self.schedulers[param_name]

        progress = min(current_step / self.total_timesteps, 1.0)
        return schedule_fn(progress)

    def step(self, current_step: int) -> None:
        """Update trainer hyperparameters for the current step."""
        if not self.schedulers:
            return  # No schedulers configured

        updates = {}
        trainer = self.trainer
        for param_name in self.schedulers.keys():
            new_value = self._compute_scheduled_value(param_name, current_step)
            updates[param_name] = new_value

        # Update only the parameters that have schedulers
        if "learning_rate" in updates:
            trainer.optimizer.param_groups[0]["lr"] = updates["learning_rate"]
        if "ppo_clip_coef" in updates:
            trainer.trainer_cfg.ppo.clip_coef = updates["ppo_clip_coef"]
        if "ppo_vf_clip_coef" in updates:
            trainer.trainer_cfg.ppo.vf_clip_coef = updates["ppo_vf_clip_coef"]
        if "ppo_ent_coef" in updates:
            trainer.trainer_cfg.ppo.ent_coef = updates["ppo_ent_coef"]
        if "ppo_l2_reg_loss_coef" in updates:
            trainer.trainer_cfg.ppo.l2_reg_loss_coef = updates["ppo_l2_reg_loss_coef"]
        if "ppo_l2_init_loss_coef" in updates:
            trainer.trainer_cfg.ppo.l2_init_loss_coef = updates["ppo_l2_init_loss_coef"]

        if current_step % 1000 == 0 and updates:
            self.logger.info(
                f"Step {current_step}: Updated hyperparameters: "
                + ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
            )

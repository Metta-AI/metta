import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

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

    def __init__(self, trainer_cfg: DictConfig, total_timesteps: int, logging):
        """Initialize the hyperparameter scheduler with configuration and total timesteps."""
        self.trainer_cfg = trainer_cfg
        self.total_timesteps = total_timesteps
        self.logger = logging.getLogger(__name__)

        self.initial_values = {
            "learning_rate": trainer_cfg.optimizer.learning_rate,
            "ppo_clip_coef": trainer_cfg.ppo.clip_coef,
            "ppo_vf_clip_coef": trainer_cfg.ppo.vf_clip_coef,
            "ppo_ent_coef": trainer_cfg.ppo.ent_coef,
            "ppo_l2_reg_loss_coef": trainer_cfg.ppo.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": trainer_cfg.ppo.l2_init_loss_coef,
        }
        self.schedules = {
            "learning_rate": self._get_schedule_config(trainer_cfg, "learning_rate_schedule", "cosine"),
            "ppo_clip_coef": self._get_schedule_config(trainer_cfg, "ppo_clip_schedule", "logarithmic"),
            "ppo_vf_clip_coef": self._get_schedule_config(trainer_cfg, "ppo_vf_clip_schedule", "linear"),
            "ppo_ent_coef": self._get_schedule_config(trainer_cfg, "ppo_ent_coef_schedule", "linear"),
            "ppo_l2_reg_loss_coef": self._get_schedule_config(trainer_cfg, "ppo_l2_reg_loss_schedule", "linear"),
            "ppo_l2_init_loss_coef": self._get_schedule_config(trainer_cfg, "ppo_l2_init_loss_schedule", "linear"),
        }

    def _get_schedule_config(self, cfg: DictConfig, schedule_key: str, default_type: str) -> Dict[str, Any]:
        """Extract schedule configuration from trainer_cfg or return default."""
        schedule = getattr(cfg, schedule_key, None)
        if schedule is None:
            return {"type": default_type}
        return {
            "type": schedule.get("type", default_type),
            "min_value": schedule.get("min_value", None),
            "max_value": schedule.get("max_value", None),
            "decay_rate": schedule.get("decay_rate", 0.1),
        }

    def _compute_scheduled_value(self, param_name: str, current_step: int) -> float:
        """Compute the scheduled value for a hyperparameter at the current step."""
        initial_value = self.initial_values[param_name]
        schedule = self.schedules[param_name]
        schedule_type = schedule["type"]

        progress = min(current_step / self.total_timesteps, 1.0)

        schedule_class = self.scheduler_registry.get(schedule_type)
        if not schedule_class:
            self.logger.warning(f"Unknown schedule type {schedule_type} for {param_name}, using constant")
            return initial_value

        scheduled_value = schedule_class(
            initial_value=initial_value,
            min_value=schedule.get("min_value"),
            max_value=schedule.get("max_value"),
            decay_rate=schedule.get("decay_rate", 0.1),
        )(progress)

        return scheduled_value

    def step(self, trainer, current_step: int) -> None:
        """Update trainer hyperparameters for the current step."""
        updates = {}

        for param_name in self.initial_values:
            new_value = self._compute_scheduled_value(param_name, current_step)
            updates[param_name] = new_value

        trainer.optimizer.param_groups[0]["lr"] = updates["learning_rate"]
        trainer.trainer_cfg.ppo.clip_coef = updates["ppo_clip_coef"]
        trainer.trainer_cfg.ppo.vf_clip_coef = updates["ppo_vf_clip_coef"]
        trainer.trainer_cfg.ppo.ent_coef = updates["ppo_ent_coef"]
        trainer.trainer_cfg.ppo.l2_reg_loss_coef = updates["ppo_l2_reg_loss_coef"]
        trainer.trainer_cfg.ppo.l2_init_loss_coef = updates["ppo_l2_init_loss_coef"]

        if current_step % 1000 == 0:
            self.logger.info(
                f"Step {current_step}: Updated hyperparameters: "
                + ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
            )

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


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

        self.schedulers = {}

        for param_name, initial_value in initial_values.items():
            schedule_config = schedule_configs.get(param_name)
            if schedule_config is not None:
                self.logger.info(f"Initializing scheduler for: {param_name}")
                self.schedulers[param_name] = self._create_scheduler(schedule_config, initial_value)
            else:
                self.schedulers[param_name] = ConstantSchedule(initial_value)

    def _create_scheduler(self, config: dict, initial_value: float) -> BaseSchedule:
        """Create scheduler from configuration dict."""
        schedule_type = config.get("_target_", "").split(".")[-1]

        if schedule_type == "CosineSchedule":
            return CosineSchedule(initial_value, min_value=config.get("min_value"))
        elif schedule_type == "LinearSchedule":
            return LinearSchedule(initial_value, min_value=config.get("min_value"))
        elif schedule_type == "LogarithmicSchedule":
            return LogarithmicSchedule(
                initial_value, min_value=config.get("min_value"), decay_rate=config.get("decay_rate", 0.1)
            )
        elif schedule_type == "ExponentialSchedule":
            return ExponentialSchedule(
                initial_value, min_value=config.get("min_value"), decay_rate=config.get("decay_rate", 0.95)
            )
        else:
            return ConstantSchedule(initial_value)

    @staticmethod
    def from_trainer_config(trainer_cfg: "TrainerConfig", optimizer, total_timesteps: int, logger=None):
        """Factory method to create HyperparameterScheduler from trainer config.

        Args:
            trainer_cfg: The trainer configuration (Pydantic)
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

        # Extract schedule configs from hyperparameter_scheduler
        scheduler_cfg = trainer_cfg.hyperparameter_scheduler
        schedule_configs = {}

        if scheduler_cfg:
            schedule_configs = {
                "learning_rate": scheduler_cfg.learning_rate_schedule,
                "ppo_clip_coef": scheduler_cfg.ppo_clip_schedule,
                "ppo_vf_clip_coef": scheduler_cfg.ppo_vf_clip_schedule,
                "ppo_ent_coef": scheduler_cfg.ppo_ent_coef_schedule,
                "ppo_l2_reg_loss_coef": scheduler_cfg.ppo_l2_reg_loss_schedule,
                "ppo_l2_init_loss_coef": scheduler_cfg.ppo_l2_init_loss_schedule,
            }

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
        if "learning_rate" in updates and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = updates["learning_rate"]

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

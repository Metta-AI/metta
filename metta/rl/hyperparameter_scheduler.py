import math
from typing import Any, Dict

from omegaconf import DictConfig


class HyperparameterScheduler:
    def __init__(self, trainer_cfg: DictConfig, total_timesteps: int, logging):
        """Initialize the hyperparameter scheduler with configuration and total timesteps."""
        self.trainer_cfg = trainer_cfg
        self.total_timesteps = total_timesteps  # This should be actual timesteps, not epochs
        self.logger = logging.getLogger(__name__)

        # Store initial values
        self.initial_values = {
            "learning_rate": trainer_cfg.optimizer.learning_rate,
            "ppo_clip_coef": trainer_cfg.ppo.clip_coef,
            "ppo_vf_clip_coef": trainer_cfg.ppo.vf_clip_coef,
            "ppo_ent_coef": trainer_cfg.ppo.ent_coef,
            "ppo_l2_reg_loss_coef": trainer_cfg.ppo.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": trainer_cfg.ppo.l2_init_loss_coef,
        }

        # Define scheduling strategies
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

        if schedule_type == "constant":
            return initial_value

        progress = min(current_step / self.total_timesteps, 1.0)

        if schedule_type == "linear":
            min_value = schedule.get("min_value", initial_value * 0.1)
            return initial_value + (min_value - initial_value) * progress

        elif schedule_type == "cosine":
            min_value = schedule.get("min_value", initial_value * 0.1)
            return min_value + (initial_value - min_value) * (1 + math.cos(math.pi * progress)) / 2

        elif schedule_type == "logarithmic":
            decay_rate = schedule.get("decay_rate", 0.1)
            min_value = schedule.get("min_value", initial_value * 0.01)
            if progress == 0:
                return initial_value
            log_progress = math.log1p(decay_rate * progress) / math.log1p(decay_rate)
            return initial_value + (min_value - initial_value) * log_progress

        else:
            self.logger.warning(f"Unknown schedule type {schedule_type} for {param_name}, using constant")
            return initial_value

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

import logging
import math


def _compute_scheduled_value(schedule_type: str, initial_value: float, progress: float, **kwargs) -> float:
    """Compute scheduled value for a hyperparameter given progress (0.0 to 1.0)."""
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

    if schedule_type == "linear":
        min_value = kwargs.get("min_value", initial_value * 0.1)
        return initial_value + (min_value - initial_value) * progress
    elif schedule_type == "cosine":
        min_value = kwargs.get("min_value", initial_value * 0.1)
        return min_value + (initial_value - min_value) * (1 + math.cos(math.pi * progress)) / 2
    elif schedule_type == "exponential":
        decay_rate = kwargs.get("decay_rate", 0.95)
        min_value = kwargs.get("min_value", initial_value * 0.1)
        return max(initial_value * (decay_rate**progress), min_value)
    else:  # constant or unknown
        return initial_value


def step_hyperparameters(
    trainer_cfg, optimizer, current_step: int, total_timesteps: int, logger=None
) -> dict[str, float]:
    """Update hyperparameters for current training step. Returns dict of updated values."""
    logger = logger or logging.getLogger(__name__)
    scheduler_cfg = trainer_cfg.hyperparameter_scheduler

    # Check if scheduling is enabled
    if not getattr(scheduler_cfg, "enabled", False):
        return {}

    progress = min(current_step / max(total_timesteps, 1), 1.0)
    updates = {}

    # Learning rate
    if hasattr(scheduler_cfg, "learning_rate_decay") and scheduler_cfg.learning_rate_decay < 1.0:
        new_lr = trainer_cfg.optimizer.learning_rate * (scheduler_cfg.learning_rate_decay**progress)
        optimizer.param_groups[0]["lr"] = new_lr
        updates["learning_rate"] = new_lr

    # PPO clip coefficient
    if hasattr(scheduler_cfg, "ppo_clip_decay") and scheduler_cfg.ppo_clip_decay < 1.0:
        new_clip = trainer_cfg.ppo.clip_coef * (scheduler_cfg.ppo_clip_decay**progress)
        updates["ppo_clip_coef"] = new_clip

    # Log periodically
    if current_step % 10000 == 0 and updates:
        params_str = ", ".join(f"{k}={v:.6f}" for k, v in updates.items())
        logger.info(f"Step {current_step}: {params_str}")

    return updates
